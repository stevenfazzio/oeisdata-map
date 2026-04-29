"""Run a Haiku-vs-Sonnet model comparison on a 200-row OEIS sample.

Materializes ``data/eval/sample.parquet`` (deterministic given ``--n`` and
``--seed``), then classifies the sample with each requested model and writes
per-model results to ``data/eval/enriched_{model}.parquet``. Incremental:
re-running with the same args skips rows already classified for each model.

Usage:

    python eval/run_models.py [--n 200] [--seed 42] [--models haiku,sonnet]

After this, run ``python eval/compare.py`` to compute per-field agreement and
generate the disagreement report.

**Re-runs after taxonomy revision.** If the system prompt or enums in
``pipeline/enrichment.py`` change, the cached parquets in ``data/eval/`` are
stale. Delete ``data/eval/enriched_*.parquet`` (and optionally
``data/eval/sample.parquet`` if you also changed ``--n``/``--seed``) before
re-running this script.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import anthropic

# Make pipeline.* importable when this script is run via `python eval/run_models.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Local sibling import (the script's directory is sys.path[0])
from _sample import load_sample  # noqa: E402

from pipeline.config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    ANTHROPIC_CONCURRENCY,
    DATA_DIR,
    ENRICH_BATCH_SIZE,
    RAW_PARQUET,
)
from pipeline.enrichment import (  # noqa: E402
    ENUM_COLS,
    enrich_dataframe,
    estimate_cost,
    safe_write_parquet,
)

# Eval-only model registry. All three are hardcoded so the eval harness can
# compare any subset regardless of which model production stage 03 currently
# uses. Production model lives in pipeline/config.py (ANTHROPIC_MODEL_ENRICH).
MODEL_IDS: dict[str, str] = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-5",
    "opus": "claude-opus-4-6",
}

EVAL_DIR = DATA_DIR / "eval"
SAMPLE_PARQUET = EVAL_DIR / "sample.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=200, help="number of rows to sample (default: 200)")
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducibility (default: 42)")
    p.add_argument(
        "--models",
        type=str,
        default="haiku,sonnet",
        help=f"comma-separated subset of {sorted(MODEL_IDS)} (default: haiku,sonnet)",
    )
    return p.parse_args()


def get_or_build_sample(n: int, seed: int):
    """Load data/eval/sample.parquet if it matches (n, seed); otherwise rebuild it.

    Stamps the file with `_meta_n` and `_meta_seed` columns for cheap detection
    of stale samples on later runs.
    """
    import pandas as pd

    if SAMPLE_PARQUET.exists():
        cached = pd.read_parquet(SAMPLE_PARQUET)
        cached_n = int(cached["_meta_n"].iloc[0]) if "_meta_n" in cached.columns and len(cached) else None
        cached_seed = int(cached["_meta_seed"].iloc[0]) if "_meta_seed" in cached.columns and len(cached) else None
        if cached_n == n and cached_seed == seed and len(cached) == n:
            print(f"Reusing existing sample at {SAMPLE_PARQUET} (n={n}, seed={seed})")
            return cached.drop(columns=[c for c in ("_meta_n", "_meta_seed") if c in cached.columns])
        print(
            f"Existing sample at {SAMPLE_PARQUET} has (n={cached_n}, seed={cached_seed}); "
            f"rebuilding for (n={n}, seed={seed})"
        )

    sample = load_sample(RAW_PARQUET, n=n, seed=seed)
    stamped = sample.copy()
    stamped["_meta_n"] = n
    stamped["_meta_seed"] = seed
    safe_write_parquet(stamped, SAMPLE_PARQUET)
    print(f"Wrote {SAMPLE_PARQUET} ({len(sample)} rows)")
    return sample


async def main() -> None:
    args = parse_args()
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in requested if m not in MODEL_IDS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; valid: {sorted(MODEL_IDS)}")

    if not ANTHROPIC_API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY not set; check ~/.config/data-apis/.env or .env")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    sample = get_or_build_sample(n=args.n, seed=args.seed)
    print(f"\nSample: {len(sample)} rows × {len(sample.columns)} cols")
    print(f"  cols: {list(sample.columns)}")

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    summary: list[tuple[str, str, int, int, float, float]] = []  # (short, full, in, out, cost, secs)
    grand_total_cost = 0.0

    for short in requested:
        model_id = MODEL_IDS[short]
        out_path = EVAL_DIR / f"enriched_{short}.parquet"
        print(f"\n=== {short} ({model_id}) → {out_path.name} ===")
        t0 = time.monotonic()
        enriched, in_tok, out_tok = await enrich_dataframe(
            sample,
            client=client,
            model=model_id,
            batch_size=ENRICH_BATCH_SIZE,
            concurrency=ANTHROPIC_CONCURRENCY,
            checkpoint_path=out_path,
            checkpoint_every=50,
        )
        elapsed = time.monotonic() - t0

        # Sanity: count any unfilled cells
        n_null = int(enriched[list(ENUM_COLS)].isna().any(axis=1).sum())
        if n_null:
            print(f"  WARNING: {n_null} rows with at least one null enum cell in {out_path.name}")

        cost = estimate_cost(model_id, in_tok, out_tok)
        grand_total_cost += cost
        summary.append((short, model_id, in_tok, out_tok, cost, elapsed))

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Eval run summary")
    print("=" * 64)
    print(f"  sample: {len(sample)} rows (n={args.n}, seed={args.seed})")
    print(f"  output: {EVAL_DIR}")
    print()
    print(f"  {'model':<10} {'in tokens':>12} {'out tokens':>12} {'cost USD':>12} {'wall':>10}")
    print(f"  {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 10}")
    for short, _full, in_tok, out_tok, cost, secs in summary:
        print(f"  {short:<10} {in_tok:>12,} {out_tok:>12,} ${cost:>10.4f} {secs:>9.1f}s")
    print(f"  {'TOTAL':<10} {'':>12} {'':>12} ${grand_total_cost:>10.4f}")
    print()
    print("Next: python eval/compare.py")


if __name__ == "__main__":
    asyncio.run(main())
