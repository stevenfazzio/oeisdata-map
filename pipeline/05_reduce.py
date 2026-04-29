"""Reduce 512-dim Cohere embeddings to 2D with UMAP.

Reads `data/embeddings.npz` (key `embeddings`, float32 (N, 512)) and writes
`data/umap_coords.npz` (key `coords`, float32 (N, 2)) preserving row order.

Parameters are copied verbatim from `semantic-github-map/pipeline/05_reduce_umap.py`:

    UMAP(n_components=2, n_neighbors=15, min_dist=0.05, metric="cosine",
         random_state=42)

UMAP is deterministic with `random_state=42`, so re-running is cheap and
produces identical coords — no incremental resume logic is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import umap
from config import EMBEDDINGS_NPZ, UMAP_COORDS_NPZ

# ── Atomic .npz save helper ──────────────────────────────────────────────────


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    """`np.savez` via temp+rename. Pass a file handle so numpy doesn't auto-append `.npz`."""
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
    tmp_path.replace(path)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    if not EMBEDDINGS_NPZ.exists():
        raise SystemExit(f"{EMBEDDINGS_NPZ} not found — run `uv run python pipeline/04_embed.py` first")

    print(f"Loading {EMBEDDINGS_NPZ.name}…")
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"]
    print(f"  {embeddings.shape} {embeddings.dtype}\n")

    print("Fitting UMAP (n_components=2, n_neighbors=15, min_dist=0.05, metric=cosine, random_state=42)…")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.05,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings).astype(np.float32)
    print(f"  → {coords.shape} {coords.dtype}")
    print(f"  x range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(f"  y range: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")

    print(f"\nWriting {UMAP_COORDS_NPZ.name}…")
    _atomic_savez(UMAP_COORDS_NPZ, coords=coords)
    size_kb = UMAP_COORDS_NPZ.stat().st_size / 1_000
    print(f"Done. {UMAP_COORDS_NPZ} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
