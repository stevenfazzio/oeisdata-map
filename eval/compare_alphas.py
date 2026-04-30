"""Render three retrofit-α variants of the v2 map for side-by-side comparison.

Generates three full pipeline outputs (stages 05+06+07) on the v2 embeddings,
one per α value, and saves each rendered HTML to docs/compare/{label}.html so
they can be viewed and compared.

Variants:
  - baseline: α=0 (no retrofit; raw Cohere embeddings)
  - alpha_02: α=0.2, n_iter=10 (gentle retrofit, knee of the Pareto frontier)
  - alpha_08: α=0.8, n_iter=5  (current published winner, max Hits@10 in v2)

Cost note: uses Haiku 4.5 as the Toponymy namer (~3× cheaper than Sonnet 4.5)
because what we care about is the *relative* label coherence across α, not
absolute label quality. Once the user picks a winner, that one variant can
be re-rendered with Sonnet for the production map.

Each run takes ~10-20 minutes (Toponymy + Anthropic naming dominates). Three
runs ≈ 30-60 min wall time, ~$15-30 in Haiku spend. Within the per-run cost cap.

Run from repo root:

    OEIS_SCOPE=curated uv run python eval/compare_alphas.py

Outputs:
  - docs/compare/baseline.html      α=0
  - docs/compare/alpha_02.html      α=0.2
  - docs/compare/alpha_08.html      α=0.8
  - data/embeddings_compare_*.npz   per-α vectors (preserved for re-runs)
  - data/labels_compare_*.parquet   per-α Toponymy outputs
  - data/umap_compare_*.npz         per-α UMAP coords

After the script finishes, the main pipeline files (data/embeddings.npz,
data/umap_coords.npz, data/labels.parquet, docs/index.html) reflect whichever
variant ran LAST. Re-run pipeline/05_reduce.py + 06_label.py + 07_visualize.py
afterward to restore a specific variant as the canonical pipeline state.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

# Import retrofit-math helpers without triggering config dotenv load twice.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("retrofit_mod", REPO_ROOT / "pipeline" / "04b_retrofit.py")
_retrofit_mod = importlib.util.module_from_spec(_spec)
sys.modules["retrofit_mod"] = _retrofit_mod
_spec.loader.exec_module(_retrofit_mod)

build_adjacency = _retrofit_mod.build_adjacency
retrofit_fn = _retrofit_mod.retrofit

import pandas as pd  # noqa: E402

DATA = REPO_ROOT / "data"
DOCS = REPO_ROOT / "docs"
COMPARE_DIR = DOCS / "compare"

# (α, label, n_iter) — n_iter chosen per the v2 grid winner for that α.
# n_iter sensitivity is empirically near-zero on this graph (<0.001 Hits@10
# between iter=5 and iter=10), so picking either is fine — we just pick the
# v2 winner for that α to match what the eval reported.
VARIANTS: list[tuple[float, str, int]] = [
    (0.0, "baseline", 0),
    (0.2, "alpha_02", 10),
    (0.8, "alpha_08", 5),
]


def _atomic_savez(path: Path, **arrays) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
    tmp_path.replace(path)


def _build_alpha_embeddings(alpha: float, n_iter: int, label: str) -> Path:
    """Generate or reuse the embeddings file for this α and return its path.

    α=0 reuses the baseline embeddings.npz directly (no copy needed since
    OEIS_EMBEDDINGS_FILE just names the file inside data/). α>0 computes the
    Faruqui retrofit and writes embeddings_compare_{label}.npz.
    """
    out_name = f"embeddings_compare_{label}.npz"
    out_path = DATA / out_name

    if alpha == 0.0:
        # Baseline = raw Cohere embeddings. The pipeline reads the file named
        # by OEIS_EMBEDDINGS_FILE, so we set it to "embeddings.npz" later.
        return Path("embeddings.npz")

    if out_path.exists():
        # Already computed for a previous run; reuse.
        print(f"  [{label}] reusing existing {out_name}")
        return Path(out_name)

    # Load v2 baseline + the Y-edge graph + isolated mask (from saved split
    # if it exists, else recomputed).
    print(f"  [{label}] computing α={alpha}, n_iter={n_iter} retrofit…")
    Q0 = np.load(DATA / "embeddings.npz")["embeddings"].astype(np.float32)
    norms = np.linalg.norm(Q0, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Q0 = (Q0 / norms).astype(np.float32)

    # Use the train edges from y_edges_split.parquet (produced by stage 04b).
    edges_split = pd.read_parquet(DATA / "y_edges_split.parquet")
    train = edges_split[edges_split["split"] == "train"]
    enriched = pd.read_parquet(DATA / "enriched.parquet", columns=["id"])
    id_to_idx = {sid: i for i, sid in enumerate(enriched["id"].tolist())}
    n = len(enriched)
    edges_arr = np.column_stack(
        [
            train["from_id"].map(id_to_idx).to_numpy(dtype=np.int32),
            train["to_id"].map(id_to_idx).to_numpy(dtype=np.int32),
        ]
    )
    adj_norm = build_adjacency(n, edges_arr)
    train_deg = np.asarray((adj_norm > 0).sum(axis=1)).ravel()
    isolated_mask = train_deg == 0

    Q = retrofit_fn(Q0, adj_norm, isolated_mask, alpha=alpha, n_iter=n_iter)

    # Save with the same v2 text_version stamp so stage 04 wouldn't try to
    # invalidate it (not that stage 04 reads this file, but defensively).
    _atomic_savez(out_path, embeddings=Q, text_version=np.asarray("v2"))
    size_mb = out_path.stat().st_size / 1_000_000
    print(f"    → {out_name} ({Q.shape}, {size_mb:.1f} MB)")
    return Path(out_name)


def _run_stage(stage_script: str, env: dict[str, str]) -> None:
    """Run a pipeline stage as a subprocess, streaming + capturing output.

    Stream-and-tee so the parent's log file gets real-time progress (instead
    of waiting for the whole subprocess to finish before flushing). PYTHONUNBUFFERED=1
    forces the child Python's stdout to flush per line.
    """
    cmd = ["uv", "run", "python", f"pipeline/{stage_script}"]
    child_env = {**env, "PYTHONUNBUFFERED": "1"}
    print(
        f"    $ {' '.join(cmd)}  (env: OEIS_EMBEDDINGS_FILE={env.get('OEIS_EMBEDDINGS_FILE')!r}, "
        f"ANTHROPIC_MODEL_NAMER={env.get('ANTHROPIC_MODEL_NAMER')!r})",
        flush=True,
    )
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        env=child_env,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        # Indent so the variant boundaries stay readable.
        sys.stdout.write("    > " + line)
        sys.stdout.flush()
        last_lines.append(line)
        # Bound memory: only keep the last 200 lines for tail-on-failure.
        if len(last_lines) > 200:
            last_lines = last_lines[-200:]
    rc = proc.wait()
    elapsed = time.time() - t0
    if rc != 0:
        tail = "".join(last_lines[-50:])
        raise SystemExit(f"Stage {stage_script} failed after {elapsed:.0f}s. Last 50 lines:\n{tail}")
    print(f"    ({elapsed:.0f}s) ✓ {stage_script}", flush=True)


def main() -> None:
    if os.environ.get("OEIS_SCOPE", "").lower() != "curated":
        raise SystemExit(
            "OEIS_SCOPE must be 'curated' for this comparison. "
            "Run as: OEIS_SCOPE=curated uv run python eval/compare_alphas.py"
        )
    if not (DATA / "embeddings.npz").exists():
        raise SystemExit("data/embeddings.npz missing — run stage 04 first")
    if not (DATA / "y_edges_split.parquet").exists():
        raise SystemExit(
            "data/y_edges_split.parquet missing — run pipeline/04b_retrofit.py first "
            "(it produces the train/holdout edge split this script reuses)"
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY missing — needed for stage 06 cluster naming")
    if not os.environ.get("CO_API_KEY"):
        raise SystemExit("CO_API_KEY missing — needed for stage 06 keyphrase Cohere calls")

    COMPARE_DIR.mkdir(exist_ok=True)

    # Build env shared across all variants. Haiku for naming (cheaper).
    base_env = {**os.environ, "ANTHROPIC_MODEL_NAMER": "claude-haiku-4-5"}

    print(f"=== Comparison run: {len(VARIANTS)} variants ===\n", flush=True)

    for alpha, label, n_iter in VARIANTS:
        print(f"\n=== Variant: {label} (α={alpha}, n_iter={n_iter}) ===", flush=True)
        emb_path = _build_alpha_embeddings(alpha, n_iter, label)
        env = {**base_env, "OEIS_EMBEDDINGS_FILE": str(emb_path)}

        for stage in ("05_reduce.py", "06_label.py", "07_visualize.py"):
            _run_stage(stage, env)

        # Snapshot variant outputs to compare/
        out_html = COMPARE_DIR / f"{label}.html"
        shutil.copy2(DOCS / "index.html", out_html)
        size_mb = out_html.stat().st_size / 1_000_000
        print(f"  → {out_html.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")

        # Also snapshot the labels and umap coords for later analysis
        shutil.copy2(DATA / "umap_coords.npz", DATA / f"umap_compare_{label}.npz")
        shutil.copy2(DATA / "labels.parquet", DATA / f"labels_compare_{label}.parquet")

    print("\n=== Done. Comparison HTMLs in docs/compare/ ===")
    for _, label, _ in VARIANTS:
        print(f"  - docs/compare/{label}.html")


if __name__ == "__main__":
    main()
