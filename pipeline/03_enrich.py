"""Stage 03: classify selected sequences with Claude Sonnet into 4 enum fields.

Reads ``data/selected.parquet`` (output of stage 02), runs each sequence
through ``pipeline.enrichment.enrich_dataframe`` (batched async tool-use),
and writes ``data/enriched.parquet`` with 4 new columns: ``math_domain``,
``sequence_type``, ``growth_class``, ``origin_era``.

**Model:** ``ANTHROPIC_MODEL_ENRICH`` in ``pipeline/config.py`` — Sonnet 4.5
since Sprint 6. Sprint 3-5 used Haiku 4.5; the Sprint 6 200-row eval showed
Haiku trailed Opus by 8-20pp on every enum (worst on origin_era at 58% vs
Sonnet's 77.5% Opus-agreement) so production switched to Sonnet for ~3x cost
in exchange for the accuracy gain.

**Hard skip at OEIS_SCOPE=all.** A full Sonnet enrichment of all 394k
sequences would cost ~$1900, well over the $100 cap in CLAUDE.md. When
``OEIS_SCOPE=all``, this stage copies ``selected.parquet`` to
``enriched.parquet`` unchanged so downstream stages have a consistent
input — they will simply omit the LLM-derived colormaps in the
visualization.

Incremental: re-running this script skips rows that already have all 4 enum
columns populated in ``enriched.parquet`` AND have all of ENUM_COLS as
columns. **Footgun:** if you tweak SYSTEM_PROMPT or the values inside an
existing enum without changing ENUM_COLS shape, the resume logic will treat
all rows as already-enriched and skip silently. Delete ``enriched.parquet``
before re-running in that case.

Usage:

    OEIS_SCOPE=core    uv run python pipeline/03_enrich.py   # v0  (~$1, ~1 min)
    OEIS_SCOPE=curated uv run python pipeline/03_enrich.py   # v1  (~$54, ~30-60 min)
    OEIS_SCOPE=all     uv run python pipeline/03_enrich.py   # skipped (cost cap)
"""

from __future__ import annotations

import asyncio
import shutil
import time

import anthropic
import pandas as pd
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_CONCURRENCY,
    ANTHROPIC_MODEL_ENRICH,
    ENRICH_BATCH_SIZE,
    ENRICHED_PARQUET,
    SCOPE,
    SELECTED_PARQUET,
    SKIP_ENRICH,
)
from enrichment import ENUM_COLS, enrich_dataframe, estimate_cost


async def _run() -> None:
    if not SELECTED_PARQUET.exists():
        raise SystemExit(f"{SELECTED_PARQUET} not found — run `uv run python pipeline/02_select.py` first")

    if SCOPE == "all" or SKIP_ENRICH:
        # Hard skip at SCOPE=all per the CLAUDE.md cost cap (~$650).
        # Also honored when OEIS_SKIP_ENRICH=1 for iteration-without-enrichment.
        reason = "OEIS_SCOPE=all (cost cap)" if SCOPE == "all" else "OEIS_SKIP_ENRICH=1"
        print(f"Stage 03 skipped — {reason}")
        print(f"  copying {SELECTED_PARQUET.name} → {ENRICHED_PARQUET.name} unchanged")
        shutil.copy(SELECTED_PARQUET, ENRICHED_PARQUET)
        size_mb = ENRICHED_PARQUET.stat().st_size / 1_000_000
        print(f"  done ({size_mb:.1f} MB)")
        return

    if not ANTHROPIC_API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY not set; check ~/.config/data-apis/.env or .env")

    print(f"Reading {SELECTED_PARQUET.name}…")
    df = pd.read_parquet(SELECTED_PARQUET)
    print(f"  {len(df):,} rows loaded (scope={SCOPE})\n")

    print(f"Enriching with {ANTHROPIC_MODEL_ENRICH} (batch={ENRICH_BATCH_SIZE}, concurrency={ANTHROPIC_CONCURRENCY})")
    print(f"  checkpoint: {ENRICHED_PARQUET}")

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    t0 = time.monotonic()
    enriched, in_tok, out_tok = await enrich_dataframe(
        df,
        client=client,
        model=ANTHROPIC_MODEL_ENRICH,
        batch_size=ENRICH_BATCH_SIZE,
        concurrency=ANTHROPIC_CONCURRENCY,
        checkpoint_path=ENRICHED_PARQUET,
        checkpoint_every=200,
    )
    elapsed = time.monotonic() - t0

    # Sanity: how many rows ended up with at least one null enum cell?
    n_null = int(enriched[list(ENUM_COLS)].isna().any(axis=1).sum())
    print()
    print(f"Stage 03 done in {elapsed:.1f}s")
    print(f"  rows: {len(enriched):,} total, {len(enriched) - n_null:,} fully classified")
    if n_null:
        print(f"  WARNING: {n_null:,} rows have at least one null enum cell — re-run to retry")

    if in_tok or out_tok:
        cost = estimate_cost(ANTHROPIC_MODEL_ENRICH, in_tok, out_tok)
        print(f"  this run: {in_tok:,} in tokens + {out_tok:,} out tokens = ${cost:.4f}")
    else:
        print("  this run: 0 API calls (all rows were already enriched)")

    size_mb = ENRICHED_PARQUET.stat().st_size / 1_000_000
    print(f"  output: {ENRICHED_PARQUET} ({size_mb:.2f} MB)")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
