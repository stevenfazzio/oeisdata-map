"""Embed selected sequences with Cohere embed-v4.0.

Reads `data/enriched.parquet` and writes:

- `data/embeddings.npz`        — float32 (N, 512) array under key `embeddings`,
                                 plus a `text_version` string under that key
- `data/embeddings_index.npy`  — string array of OEIS ids in the same row order

The composite text per sequence (see `build_embed_text`) is:

    Name: <name>
    Keywords: <editorial keywords only, comma-joined>
    First values: <values_preview_str>
    Description:
    <comments truncated to MAX_COMMENT_CHARS>
    Formula: <formulas truncated to MAX_FORMULA_CHARS>
    Examples:
    <examples truncated to MAX_EXAMPLE_CHARS>

Two design choices the text composition rests on:

- **Content keywords are removed.** OEIS keywords split into content
  (``tabl``, ``mult``, ``sign``, …) and editorial (``nice``, ``easy``,
  ``hard``). Content keywords are independently used as an embedding-quality
  eval signal (stage 04b silhouette), so feeding them into the embedder
  would make that eval circular: we'd be measuring whether Cohere copied
  the literal token rather than whether it captured the property from
  prose. See pipeline/keywords.py for the split.

- **The `First values` line is the killer signal** — Cohere recognizes
  Fibonacci from ``0, 1, 1, 2, 3, 5, 8…`` directly even when the prose is
  generic.

- **Examples** (``%e``) are content-rich and weren't previously embedded.
  Often more discriminative than comments alone, especially for
  combinatorial sequences whose worked examples reveal structure.

**Resume:** if `embeddings.npz` + `embeddings_index.npy` already exist AND
the cached `text_version` matches the current EMBED_TEXT_VERSION, only rows
whose id is missing from the cached index get re-embedded; the rest are
reused. If the version differs, the whole cache is invalidated and every
row is re-embedded (since the meaning of every cached vector has changed).
The final output is always re-stitched to match the row order of
`enriched.parquet` so downstream stages can rely on positional alignment
with the parquet.
"""

from __future__ import annotations

import time
from pathlib import Path

import cohere
import numpy as np
import pandas as pd
from config import (
    CO_API_KEY,
    COHERE_BATCH_SIZE,
    COHERE_EMBED_DIMENSION,
    COHERE_EMBED_MODEL,
    EMBEDDINGS_INDEX_NPY,
    EMBEDDINGS_NPZ,
    ENRICHED_PARQUET,
    MAX_COMMENT_CHARS,
    MAX_EXAMPLE_CHARS,
    MAX_FORMULA_CHARS,
    MAX_NAME_CHARS,
)
from keywords import EMBED_TEXT_DROP
from tqdm import tqdm

# Bump this whenever build_embed_text's output changes. Cached embeddings
# from an older version are invalidated automatically (see _load_existing).
# v1: name + keywords (all) + values + comments + formula
# v2: name + keywords (editorial only, content+universal dropped) + values
#     + comments + formula + examples
EMBED_TEXT_VERSION = "v2"


# ── Embed text builder ───────────────────────────────────────────────────────


def build_embed_text(row) -> str:
    """Composite text per sequence; values preview is the killer signal.

    Content keywords are dropped (so the keyword-silhouette eval in stage
    04b stays orthogonal to the embedding); editorial keywords (nice, easy,
    hard, …) are kept since they reflect editorial attention not under
    eval. Examples are included for sequences that have them.
    """
    name = (row["name"] or "").strip()[:MAX_NAME_CHARS]
    comments = (row["comments"] or "")[:MAX_COMMENT_CHARS]
    formulas = (row["formulas"] or "")[:MAX_FORMULA_CHARS]
    examples = (row["examples"] or "")[:MAX_EXAMPLE_CHARS]
    kw = row["keywords"]
    if kw is not None:
        editorial_kws = [k for k in kw if str(k) not in EMBED_TEXT_DROP]
    else:
        editorial_kws = []
    keywords = ", ".join(editorial_kws)
    values = row["values_preview_str"] or ""
    return (
        f"Name: {name}\n"
        f"Keywords: {keywords}\n"
        f"First values: {values}\n"
        f"Description:\n{comments}\n"
        f"Formula: {formulas}\n"
        f"Examples:\n{examples}"
    )


# ── Cohere call with backoff ─────────────────────────────────────────────────


def _embed_batch(co: cohere.ClientV2, texts: list[str], *, retries: int = 5) -> np.ndarray:
    """Embed a batch with exponential backoff on transient failures."""
    for attempt in range(retries):
        try:
            resp = co.embed(
                model=COHERE_EMBED_MODEL,
                texts=texts,
                input_type="clustering",
                embedding_types=["float"],
                output_dimension=COHERE_EMBED_DIMENSION,
            )
            return np.asarray(resp.embeddings.float_, dtype=np.float32)
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2**attempt * 0.5
            print(f"    Retry {attempt + 1}/{retries} after {wait:.1f}s: {type(e).__name__}: {e}")
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


# ── Atomic .npz / .npy save helpers ──────────────────────────────────────────


def _atomic_savez(path: Path, **arrays) -> None:
    """`np.savez` via temp+rename. Passes a file handle so numpy doesn't auto-append `.npz`."""
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
    tmp_path.replace(path)


def _atomic_savenpy(path: Path, arr: np.ndarray) -> None:
    """`np.save` via temp+rename."""
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.save(f, arr)
    tmp_path.replace(path)


# ── Resume helper ────────────────────────────────────────────────────────────


def _load_existing() -> dict[str, np.ndarray]:
    """Return an `id → embedding` map for any previously cached embeddings (or empty).

    The cache is invalidated (returns empty) when the stored ``text_version``
    does not match the current ``EMBED_TEXT_VERSION`` — every cached vector
    represents a different text composition and is no longer trustworthy.
    """
    if not (EMBEDDINGS_NPZ.exists() and EMBEDDINGS_INDEX_NPY.exists()):
        return {}
    npz = np.load(EMBEDDINGS_NPZ, allow_pickle=False)
    prev_embs = npz["embeddings"]
    cached_version = (
        str(npz["text_version"].item()) if "text_version" in npz.files else "v1"  # legacy: pre-versioning
    )
    if cached_version != EMBED_TEXT_VERSION:
        print(
            f"Cache invalidated: {EMBEDDINGS_NPZ.name} has text_version={cached_version!r}, "
            f"current is {EMBED_TEXT_VERSION!r}. Re-embedding all rows."
        )
        return {}
    prev_ids = np.load(EMBEDDINGS_INDEX_NPY)
    if len(prev_embs) != len(prev_ids):
        raise SystemExit(
            f"Mismatch: {EMBEDDINGS_NPZ.name} has {len(prev_embs)} rows but "
            f"{EMBEDDINGS_INDEX_NPY.name} has {len(prev_ids)} — delete both and re-embed."
        )
    return {str(prev_ids[i]): prev_embs[i] for i in range(len(prev_ids))}


def _save_outputs(ordered_embs: np.ndarray, ordered_ids: list[str]) -> None:
    print(f"\nWriting {EMBEDDINGS_NPZ.name} ({ordered_embs.shape}, {ordered_embs.dtype})…")
    _atomic_savez(
        EMBEDDINGS_NPZ,
        embeddings=ordered_embs,
        text_version=np.asarray(EMBED_TEXT_VERSION),
    )

    print(f"Writing {EMBEDDINGS_INDEX_NPY.name} ({len(ordered_ids):,} ids)…")
    _atomic_savenpy(EMBEDDINGS_INDEX_NPY, np.asarray(ordered_ids))

    npz_size = EMBEDDINGS_NPZ.stat().st_size / 1_000_000
    npy_size = EMBEDDINGS_INDEX_NPY.stat().st_size / 1_000_000
    print(
        f"Done. {EMBEDDINGS_NPZ.name} ({npz_size:.2f} MB, text_version={EMBED_TEXT_VERSION!r}), "
        f"{EMBEDDINGS_INDEX_NPY.name} ({npy_size:.2f} MB)"
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    if not ENRICHED_PARQUET.exists():
        raise SystemExit(f"{ENRICHED_PARQUET} not found — run `uv run python pipeline/03_enrich.py` first")
    if not CO_API_KEY:
        raise SystemExit("CO_API_KEY missing — set it in ~/.config/data-apis/.env or .env")

    print(f"Reading {ENRICHED_PARQUET.name}…")
    df = pd.read_parquet(ENRICHED_PARQUET)
    print(f"  {len(df):,} rows loaded\n")

    target_ids: list[str] = df["id"].tolist()

    existing = _load_existing()
    if existing:
        print(f"Resume: found {len(existing):,} previously embedded sequences")

    needs_embed_mask = ~df["id"].isin(existing)
    n_new = int(needs_embed_mask.sum())
    print(f"  {n_new:,} rows need embedding ({len(df) - n_new:,} already cached)\n")

    if n_new > 0:
        new_df = df.loc[needs_embed_mask].reset_index(drop=True)
        texts = [build_embed_text(row) for _, row in new_df.iterrows()]
        print(f"Calling Cohere embed-v4.0 ({len(texts):,} texts, batch size {COHERE_BATCH_SIZE})…")

        co = cohere.ClientV2(api_key=CO_API_KEY)
        chunks: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), COHERE_BATCH_SIZE), desc="Embedding"):
            batch = texts[start : start + COHERE_BATCH_SIZE]
            chunks.append(_embed_batch(co, batch))
        new_embs = np.concatenate(chunks, axis=0).astype(np.float32)

        for i, sid in enumerate(new_df["id"].tolist()):
            existing[sid] = new_embs[i]
    else:
        print("All rows already embedded — re-stitching outputs to match input row order.")

    # Re-stitch into input row order so downstream stages can rely on positional alignment.
    ordered_embs = np.stack([existing[sid] for sid in target_ids], axis=0).astype(np.float32)

    _save_outputs(ordered_embs, target_ids)


if __name__ == "__main__":
    main()
