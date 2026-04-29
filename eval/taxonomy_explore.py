"""Sprint 6: explore the natural taxonomy of OEIS sequences via EVoC + Toponymy.

Runs Toponymy with an EVoCClusterer-on-full-512-dim-embeddings (NOT on
the 2D UMAP coords used by `pipeline/06_label.py`) to discover semantic
structure that the prescriptive Sprint-3 4-enum taxonomy may have missed.
Writes a human-reviewable markdown artifact to
``data/eval/taxonomy_exploration.md`` for the user to read before
redesigning ``pipeline/enrichment.py`` and re-running stage 03.

Reads:
- ``data/enriched.parquet``     — id, name, comments, formulas, values_preview_str, keywords
- ``data/embeddings.npz``       — float32 (N, 512)
- ``data/embeddings_index.npy`` — id strings, used to verify alignment

Writes:
- ``data/eval/taxonomy_exploration.md``

Cost: each cluster gets one Sonnet naming call. EVoC params are bounded
(``base_min_cluster_size=20, max_layers=5``) to keep total Sonnet spend
under ~$25 even on the curated 25k corpus.

Ported from ``~/repos/claude-code-changelog-analysis/scripts/explore_taxonomy.py``
with three OEIS-specific deviations:
1. The numpy 2.x / fast_hdbscan compat patches from ``pipeline/06_label.py``
   replace the reference's simpler ones (which crash on our dep matrix).
2. Document template includes ``keywords`` and ``formulas`` (strong OEIS
   categorical signal absent from the changelog corpus).
3. Bugfix-direction-removal branch is dropped (no analogous dominant axis
   in OEIS embeddings).
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

# ── Path setup so `from pipeline.x import …` works from eval/ ────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))  # for `import config` style

# ── numpy 2.x + fast_hdbscan 0.3 compat patches (must come BEFORE toponymy) ──
import fast_hdbscan.boruvka as _boruvka  # noqa: E402
import fast_hdbscan.numba_kdtree as _nkd  # noqa: E402
import numpy as np  # noqa: E402


def _kdtree_to_numba_patched(sklearn_kdtree):
    """Convert a sklearn KDTree → fast_hdbscan NumbaKDTree.

    Two compat fixes layered together:
    1. numpy 2.x removed structured-array attribute access — use item access.
    2. fast_hdbscan's `NumbaKDTreeType` declares float32 / bool fields, but
       modern sklearn returns float64 / int64; cast every field to the
       declared dtype and force C-contiguous layout.
    """
    data, idx_array, node_data, node_bounds = sklearn_kdtree.get_arrays()
    return _nkd.NumbaKDTree(
        np.ascontiguousarray(data, dtype=np.float32),
        np.ascontiguousarray(idx_array),
        np.ascontiguousarray(node_data["idx_start"]),
        np.ascontiguousarray(node_data["idx_end"]),
        np.ascontiguousarray(node_data["radius"], dtype=np.float32),
        np.ascontiguousarray(node_data["is_leaf"], dtype=np.bool_),
        np.ascontiguousarray(node_bounds, dtype=np.float32),
    )


_nkd.kdtree_to_numba = _kdtree_to_numba_patched

_orig_boruvka = _boruvka.parallel_boruvka
_DEFAULT_SAMPLE_WEIGHTS = np.zeros(1, dtype=np.float32)


def _boruvka_patched(tree, n_threads=1, min_samples=10, sample_weights=None, reproducible=False):
    """Forward all 5 args explicitly — the @njit overload requires concrete types,
    not the Python defaults that toponymy 0.4 leaves omitted."""
    if sample_weights is None:
        sample_weights = _DEFAULT_SAMPLE_WEIGHTS
    return _orig_boruvka(tree, n_threads, min_samples, sample_weights, reproducible)


import toponymy.clustering as _tc  # noqa: E402

_tc.parallel_boruvka = _boruvka_patched

# ── Now safe to import the rest ──────────────────────────────────────────────
import evoc  # noqa: E402
import nest_asyncio  # noqa: E402
import pandas as pd  # noqa: E402
from toponymy import Toponymy  # noqa: E402
from toponymy.cluster_layer import ClusterLayerText  # noqa: E402
from toponymy.clustering import Clusterer, build_cluster_tree, centroids_from_labels  # noqa: E402
from toponymy.embedding_wrappers import CohereEmbedder  # noqa: E402
from toponymy.llm_wrappers import AsyncAnthropicNamer  # noqa: E402

from pipeline.config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL_NAMER,
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    DATA_DIR,
    EMBEDDINGS_INDEX_NPY,
    EMBEDDINGS_NPZ,
    ENRICHED_PARQUET,
)
from pipeline.enrichment import estimate_cost  # noqa: E402

nest_asyncio.apply()

# ── CohereEmbedder.encode tolerance patch ────────────────────────────────────
# At scale, toponymy's LLM namer occasionally fails on a cluster and leaves
# topic_names with None or "" entries. The downstream `disambiguate_topics`
# step then calls `encode(topic_names)` which 400s on empty strings. Substitute
# a placeholder so embedding succeeds. Sprint 5 confirmed this is needed at 25k.
_UNNAMED_PLACEHOLDER = "unnamed topic"
_orig_cohere_encode = CohereEmbedder.encode


def _cohere_encode_tolerant(self, texts, verbose=None, show_progress_bar=None):
    sanitized = [t if (isinstance(t, str) and t.strip()) else _UNNAMED_PLACEHOLDER for t in texts]
    return _orig_cohere_encode(self, sanitized, verbose=verbose, show_progress_bar=show_progress_bar)


CohereEmbedder.encode = _cohere_encode_tolerant


# ── Cost-tracking namer ──────────────────────────────────────────────────────


class CostTrackingNamer(AsyncAnthropicNamer):
    """AsyncAnthropicNamer subclass that accumulates input/output token usage.

    Toponymy's base namer extracts only `response.content[0].text` and
    discards `response.usage`, so we override the two API-call methods to
    capture usage before extracting the text. The body of each method
    mirrors the parent's verbatim except for the two accumulator lines.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.n_calls = 0

    async def _call_single_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        async with self.semaphore:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt + self.extra_prompting}],
                temperature=temperature,
            )
            self.total_input_tokens += int(getattr(response.usage, "input_tokens", 0) or 0)
            self.total_output_tokens += int(getattr(response.usage, "output_tokens", 0) or 0)
            self.n_calls += 1
            return response.content[0].text

    async def _call_single_llm_with_system(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        async with self.semaphore:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + self.extra_prompting},
                ],
                temperature=temperature,
            )
            self.total_input_tokens += int(getattr(response.usage, "input_tokens", 0) or 0)
            self.total_output_tokens += int(getattr(response.usage, "output_tokens", 0) or 0)
            self.n_calls += 1
            return response.content[0].text


# ── EVoC clusterer ───────────────────────────────────────────────────────────


class EVoCClusterer(Clusterer):
    """Toponymy `Clusterer` subclass that runs EVoC on the full embedding vectors.

    Unlike `ToponymyClusterer` (used by `pipeline/06_label.py`), this ignores
    `clusterable_vectors` (typically a 2D UMAP projection) and clusters
    directly on the high-dim `embedding_vectors`, which preserves more of the
    semantic structure UMAP flattens out. The hierarchical layers come from
    EVoC's own `cluster_layers_` attribute.
    """

    def __init__(self, base_min_cluster_size: int = 20, noise_level: float = 0.3, max_layers: int = 5):
        super().__init__()
        self.evoc_model = evoc.EVoC(
            base_min_cluster_size=base_min_cluster_size,
            noise_level=noise_level,
            max_layers=max_layers,
        )

    def fit(self, clusterable_vectors, embedding_vectors, layer_class=ClusterLayerText, **kwargs):
        self.evoc_model.fit(embedding_vectors)
        cluster_labels = self.evoc_model.cluster_layers_
        self.cluster_tree_ = build_cluster_tree(cluster_labels)
        self.cluster_layers_ = [
            layer_class(
                labels,
                centroids_from_labels(labels, embedding_vectors),
                layer_id=i,
            )
            for i, labels in enumerate(cluster_labels)
        ]
        return self

    def fit_predict(self, clusterable_vectors, embedding_vectors, layer_class=ClusterLayerText, **kwargs):
        self.fit(clusterable_vectors, embedding_vectors, layer_class=layer_class, **kwargs)
        return self.cluster_layers_, self.cluster_tree_


# ── Document builder ─────────────────────────────────────────────────────────


def build_document(row) -> str:
    """OEIS-tuned exemplar text for taxonomy design.

    DIVERGES from `pipeline/06_label.py:build_document` — that one is the
    production stage 06 template; this one is for one-off taxonomy
    exploration. We include `keywords` and `formulas` because both carry
    strong OEIS categorical signal that name+comments alone dilute.
    """
    name = (row["name"] or "").strip() if row["name"] is not None else ""
    values = row["values_preview_str"] or ""
    comments_raw = row["comments"] if row["comments"] is not None else ""
    formulas_raw = row["formulas"] if row["formulas"] is not None else ""
    comments = comments_raw[:150]
    formulas = formulas_raw[:150]

    kws_raw = row.get("keywords")
    if kws_raw is None:
        kws = ""
    else:
        try:
            kws = ",".join(str(k) for k in list(kws_raw)[:6])
        except TypeError:
            kws = ""

    return f"{name} [{kws}] {values}\n{comments}\n{formulas}"


# ── Markdown formatter ───────────────────────────────────────────────────────


def format_layers(layers: list) -> str:
    """Format Toponymy cluster layers into readable markdown.

    Layers are reversed so the deepest (most-fine) layer renders first.
    Within a layer, topics are sorted by entry count descending. Each
    topic shows up to 3 exemplars truncated to 120 chars.
    """
    lines: list[str] = []

    for i, layer in enumerate(reversed(layers)):
        topic_names = list(layer.topic_names)
        n_topics = len(topic_names)
        cluster_labels = np.asarray(layer.cluster_labels)
        n_entries = len(cluster_labels)
        n_noise = int(np.sum(cluster_labels == -1))
        n_clustered = n_entries - n_noise

        lines.append(f"## Layer {i} ({n_topics} topics, {n_clustered} clustered, {n_noise} noise)")
        lines.append("")

        # Count entries per topic
        topic_counts: dict[int, int] = {}
        for label_idx in range(n_topics):
            topic_counts[label_idx] = int(np.sum(cluster_labels == label_idx))

        # Sort by count descending
        sorted_topics = sorted(topic_counts.items(), key=lambda x: -x[1])

        for topic_idx, count in sorted_topics:
            name = topic_names[topic_idx] if topic_idx < len(topic_names) else "(unnamed)"
            if name is None or not str(name).strip():
                name = "(unnamed)"
            lines.append(f"- **{name}** ({count} entries)")

            # Add exemplars if available
            if hasattr(layer, "exemplars") and layer.exemplars is not None:
                try:
                    exemplars = layer.exemplars[topic_idx]
                except (IndexError, KeyError):
                    exemplars = []
                for ex in list(exemplars)[:3]:
                    ex_text = str(ex).strip().replace("\n", " ")
                    if len(ex_text) > 120:
                        ex_text = ex_text[:117] + "..."
                    lines.append(f'  - "{ex_text}"')
        lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── Verify inputs ────────────────────────────────────────────────────────
    for label, path in (
        ("enriched.parquet", ENRICHED_PARQUET),
        ("embeddings.npz", EMBEDDINGS_NPZ),
        ("embeddings_index.npy", EMBEDDINGS_INDEX_NPY),
    ):
        if not path.exists():
            raise SystemExit(f"{label} not found at {path} — run earlier pipeline stages first")

    if not ANTHROPIC_API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY missing — set it in ~/.config/data-apis/.env or .env")
    if not CO_API_KEY:
        raise SystemExit("CO_API_KEY missing — set it in ~/.config/data-apis/.env or .env")

    print(f"Reading {ENRICHED_PARQUET.name}…")
    df = pd.read_parquet(ENRICHED_PARQUET)
    print(f"  {len(df):,} rows loaded")

    print(f"Reading {EMBEDDINGS_NPZ.name}, {EMBEDDINGS_INDEX_NPY.name}…")
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"].astype(np.float32)
    emb_ids = np.load(EMBEDDINGS_INDEX_NPY)
    print(f"  embeddings: {embeddings.shape} {embeddings.dtype}")

    if not (len(embeddings) == len(emb_ids) == len(df)):
        raise SystemExit(f"Length mismatch: embs={len(embeddings)}, ids={len(emb_ids)}, parquet={len(df)}")
    if list(df["id"]) != [str(x) for x in emb_ids]:
        raise SystemExit("embeddings_index.npy ids do not match enriched.parquet row order — re-run stage 04")

    # ── Build documents ──────────────────────────────────────────────────────
    print("\nBuilding document texts (OEIS taxonomy-design template)…")
    documents = [build_document(row) for _, row in df.iterrows()]
    print(f"  {len(documents):,} documents built")
    print(f"  sample[0]: {documents[0][:200]!r}")

    np.random.seed(42)

    # ── Wire Toponymy ────────────────────────────────────────────────────────
    print(f"\nInitializing Toponymy (namer={ANTHROPIC_MODEL_NAMER}, embedder={COHERE_EMBED_MODEL})…")
    namer = CostTrackingNamer(api_key=ANTHROPIC_API_KEY, model=ANTHROPIC_MODEL_NAMER)
    embedder = CohereEmbedder(api_key=CO_API_KEY, model=COHERE_EMBED_MODEL)
    clusterer = EVoCClusterer(base_min_cluster_size=20, noise_level=0.3, max_layers=5)

    topic_model = Toponymy(
        llm_wrapper=namer,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="OEIS integer sequences",
        corpus_description="curated 25k subset of the Online Encyclopedia of Integer Sequences",
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )

    # ── Fit (issues Sonnet calls) ────────────────────────────────────────────
    print("Fitting Toponymy (issues Sonnet calls per cluster — be patient)…")
    t0 = time.monotonic()
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            topic_model.fit(
                objects=documents,
                embedding_vectors=embeddings,
                clusterable_vectors=embeddings,  # ignored by EVoCClusterer but required by API
            )
            last_exc = None
            break
        except Exception as e:
            last_exc = e
            if "overloaded" in str(e).lower() and attempt < 4:
                wait = 2**attempt * 5
                print(f"  API overloaded, retrying in {wait}s ({attempt + 1}/5)…")
                time.sleep(wait)
            else:
                raise
    if last_exc is not None:
        raise last_exc
    elapsed = time.monotonic() - t0

    n_layers = len(topic_model.cluster_layers_)
    if n_layers == 0:
        raise SystemExit("EVoC produced 0 cluster layers — try lowering base_min_cluster_size")
    print(f"\nProduced {n_layers} cluster layer(s) in {elapsed:.1f}s")
    for i, layer in enumerate(topic_model.cluster_layers_):
        n_topics = len(layer.topic_names)
        n_noise = int(np.sum(np.asarray(layer.cluster_labels) == -1))
        print(f"  layer {i}: {n_topics:>3d} topics, {n_noise:>5d} noise")

    # ── Cost report ──────────────────────────────────────────────────────────
    cost = estimate_cost(ANTHROPIC_MODEL_NAMER, namer.total_input_tokens, namer.total_output_tokens)
    print(
        f"\nSonnet usage: {namer.n_calls} calls, "
        f"{namer.total_input_tokens:,} input + {namer.total_output_tokens:,} output tokens "
        f"→ ${cost:.4f}"
    )

    # ── Write markdown ───────────────────────────────────────────────────────
    output_path = DATA_DIR / "eval" / "taxonomy_exploration.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# OEIS Taxonomy Exploration (Sprint 6)\n\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Sequences: {len(df):,}\n"
        f"Layers: {n_layers}\n"
        "EVoC params: base_min_cluster_size=20, noise_level=0.3, max_layers=5\n"
        "Document template: name + keywords + values_preview + comments[:150] + formulas[:150]\n\n"
        f"Sonnet model: {ANTHROPIC_MODEL_NAMER}\n"
        f"Sonnet usage: {namer.n_calls} calls, "
        f"{namer.total_input_tokens:,} in + {namer.total_output_tokens:,} out tokens "
        f"→ **${cost:.2f}**\n"
        f"Wall time: {elapsed:.1f}s\n\n"
        "Layer ordering: coarsest (Layer 0) → finest. "
        "Within each layer, topics are sorted by entry count descending; "
        "each topic shows up to 3 exemplars (truncated to 120 chars).\n\n"
        "---\n\n"
    )
    body = format_layers(topic_model.cluster_layers_)

    tmp = output_path.with_suffix(".md.tmp")
    tmp.write_text(header + body)
    tmp.replace(output_path)
    size_kb = output_path.stat().st_size / 1_000
    print(f"\nWrote {output_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
