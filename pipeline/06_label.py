"""Cluster the OEIS map and generate hierarchical topic labels via Toponymy.

Toponymy fits a hierarchy of clusters on the UMAP coords and asks Claude
Sonnet to name each cluster from a few exemplar sequences. The result is a
parquet of `id, label_layer_0, label_layer_1, …` columns where layer 0 is
the **coarsest** layer (DataMapPlot wants coarsest first).

Reads:
- ``data/enriched.parquet``         — id + name + comments + values_preview_str
- ``data/embeddings.npz``           — float32 (N, 512)
- ``data/embeddings_index.npy``     — id strings, used to align with parquet
- ``data/umap_coords.npz``          — float32 (N, 2)

Writes:
- ``data/labels.parquet``           — id + label_layer_*
- ``data/toponymy_model.joblib``    — fitted model (skipped if unpicklable)

The numpy 2.x / fast_hdbscan compat patches at the top MUST be applied
before importing anything from `toponymy.*` — they fix two bugs that
otherwise crash the clusterer:

1. `fast_hdbscan.numba_kdtree.kdtree_to_numba` accesses structured-array
   attributes via attribute syntax (`arr.idx_start`), which numpy 2.0
   removed; the patch uses item syntax (`arr["idx_start"]`).

2. `fast_hdbscan 0.3` made `n_threads` a required positional arg of
   `parallel_boruvka`, but `toponymy 0.4` calls it without that arg.

Cost: Toponymy issues one Sonnet call per cluster, not per row. For 183
core sequences expect ~4–8 clusters per layer × 1–2 layers ≈ 5–15 calls,
≈ $0.30–1.00.
"""

from __future__ import annotations

# ── numpy 2.x + fast_hdbscan 0.3 compat patches (must come first) ───────────
import fast_hdbscan.boruvka as _boruvka  # noqa: E402
import fast_hdbscan.numba_kdtree as _nkd  # noqa: E402
import numpy as np  # noqa: E402


def _kdtree_to_numba_patched(sklearn_kdtree):
    """Convert a sklearn KDTree into a fast_hdbscan NumbaKDTree.

    Two compat fixes layered together:
    1. numpy 2.x removed structured-array attribute access — use item access.
    2. fast_hdbscan's `NumbaKDTreeType` declares float32 / bool fields, but
       modern sklearn returns float64 / int64; cast every field to the
       declared dtype and force C-contiguous layout.
    """
    data, idx_array, node_data, node_bounds = sklearn_kdtree.get_arrays()
    return _nkd.NumbaKDTree(
        np.ascontiguousarray(data, dtype=np.float32),
        np.ascontiguousarray(idx_array),  # intp = int64 on 64-bit platforms
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

# ── now safe to import the rest ──────────────────────────────────────────────
import joblib  # noqa: E402
import nest_asyncio  # noqa: E402
import pandas as pd  # noqa: E402
from config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL_NAMER,
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    EMBEDDINGS_INDEX_NPY,
    EMBEDDINGS_NPZ,
    ENRICHED_PARQUET,
    LABELS_PARQUET,
    TOPONYMY_MODEL_JOBLIB,
    UMAP_COORDS_NPZ,
)
from toponymy import Toponymy, ToponymyClusterer  # noqa: E402
from toponymy.embedding_wrappers import CohereEmbedder  # noqa: E402
from toponymy.llm_wrappers import AsyncAnthropicNamer  # noqa: E402

nest_asyncio.apply()


# Patch CohereEmbedder.encode to tolerate None / empty topic names.
# At scale, toponymy's LLM namer occasionally fails on a cluster and leaves
# topic_names with None or "" entries. The downstream `disambiguate_topics`
# step then calls `encode(topic_names)` which blindly passes the list to
# Cohere, and Cohere 400s with "one of texts, images, or inputs must be
# specified" because the slice contains no valid strings. Substitute a
# placeholder so embedding succeeds and disambiguation can proceed.
_UNNAMED_PLACEHOLDER = "unnamed topic"
_orig_cohere_encode = CohereEmbedder.encode


def _cohere_encode_tolerant(self, texts, verbose=None, show_progress_bar=None):
    sanitized = [t if (isinstance(t, str) and t.strip()) else _UNNAMED_PLACEHOLDER for t in texts]
    return _orig_cohere_encode(self, sanitized, verbose=verbose, show_progress_bar=show_progress_bar)


CohereEmbedder.encode = _cohere_encode_tolerant


# ── Document builder ─────────────────────────────────────────────────────────


def build_document(row) -> str:
    """Per-sequence document text fed to Toponymy's keyphrase / exemplar layer."""
    name = (row["name"] or "").strip()
    values = row["values_preview_str"] or ""
    comments = (row["comments"] or "")[:200]
    return f"{name} — first values {values}\n{comments}"


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    for label, path in (
        ("enriched.parquet", ENRICHED_PARQUET),
        ("embeddings.npz", EMBEDDINGS_NPZ),
        ("embeddings_index.npy", EMBEDDINGS_INDEX_NPY),
        ("umap_coords.npz", UMAP_COORDS_NPZ),
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

    print(f"Reading {EMBEDDINGS_NPZ.name}, {EMBEDDINGS_INDEX_NPY.name}, {UMAP_COORDS_NPZ.name}…")
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"].astype(np.float32)
    emb_ids = np.load(EMBEDDINGS_INDEX_NPY)
    coords = np.load(UMAP_COORDS_NPZ)["coords"].astype(np.float32)

    if not (len(embeddings) == len(emb_ids) == len(coords) == len(df)):
        raise SystemExit(
            f"Length mismatch: embs={len(embeddings)}, ids={len(emb_ids)}, coords={len(coords)}, parquet={len(df)}"
        )

    # Stage 04 already stitches outputs to the parquet row order, but verify defensively.
    if list(df["id"]) != [str(x) for x in emb_ids]:
        raise SystemExit("embeddings_index.npy ids do not match enriched.parquet row order — re-run stage 04")

    print("\nBuilding document texts…")
    documents = [build_document(row) for _, row in df.iterrows()]
    print(f"  {len(documents):,} documents built")

    # Seed numpy for reproducible exemplar selection within Toponymy.
    # (Sonnet topic-naming is still non-deterministic.)
    np.random.seed(42)

    print("\nFitting clusterer (ToponymyClusterer)…")
    clusterer = ToponymyClusterer(min_clusters=4)
    clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)
    print(f"  → {len(clusterer.cluster_layers_)} cluster layer(s)")

    print(f"\nInitializing Toponymy (namer={ANTHROPIC_MODEL_NAMER}, embedder={COHERE_EMBED_MODEL})…")
    namer = AsyncAnthropicNamer(api_key=ANTHROPIC_API_KEY, model=ANTHROPIC_MODEL_NAMER)
    embedder = CohereEmbedder(api_key=CO_API_KEY, model=COHERE_EMBED_MODEL)

    topic_model = Toponymy(
        llm_wrapper=namer,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="OEIS integer sequences",
        corpus_description="selection of integer sequences from the Online Encyclopedia of Integer Sequences",
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )

    print("Fitting Toponymy model (this issues Claude calls per cluster)…")
    topic_model.fit(
        objects=documents,
        embedding_vectors=embeddings,
        clusterable_vectors=coords,
    )

    n_layers = len(topic_model.cluster_layers_)
    if n_layers == 0:
        raise SystemExit("Toponymy produced 0 cluster layers — try reducing min_clusters")
    print(f"\nToponymy produced {n_layers} cluster layer(s)")

    # Toponymy returns layers finest → coarsest; DataMapPlot wants coarsest first.
    labels_dict: dict[str, list] = {"id": df["id"].tolist()}
    for i, layer in enumerate(reversed(topic_model.cluster_layers_)):
        labels_dict[f"label_layer_{i}"] = list(layer.topic_name_vector)
        unique_labels = sorted(set(labels_dict[f"label_layer_{i}"]))
        print(f"  layer_{i}: {len(unique_labels)} unique label(s)")
        for label_str in unique_labels[:8]:
            n = labels_dict[f"label_layer_{i}"].count(label_str)
            print(f"    {n:>4d}  {label_str[:80]}")
        if len(unique_labels) > 8:
            print(f"    …  ({len(unique_labels) - 8} more)")

    labels_df = pd.DataFrame(labels_dict)

    print(f"\nWriting {LABELS_PARQUET.name}…")
    tmp = LABELS_PARQUET.with_suffix(".parquet.tmp")
    labels_df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(LABELS_PARQUET)
    size_kb = LABELS_PARQUET.stat().st_size / 1_000
    print(f"  → {LABELS_PARQUET} ({size_kb:.1f} KB, {len(labels_df):,} rows × {len(labels_df.columns)} cols)")

    print(f"\nSaving fitted model to {TOPONYMY_MODEL_JOBLIB.name}…")
    try:
        joblib.dump(topic_model, TOPONYMY_MODEL_JOBLIB)
        size_mb = TOPONYMY_MODEL_JOBLIB.stat().st_size / 1_000_000
        print(f"  → {TOPONYMY_MODEL_JOBLIB} ({size_mb:.2f} MB)")
    except (TypeError, AttributeError) as e:
        print(f"  Skipped (async client not picklable): {type(e).__name__}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
