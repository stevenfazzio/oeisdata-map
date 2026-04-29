"""Apply the scope filter to data/raw_sequences.parquet → data/selected.parquet.

Three scopes, selected via the `OEIS_SCOPE` env var (see config.py):

- core    → 183 sequences with the OEIS `core` keyword (v0 smoke test)
- curated → ~25k sequences via a quality-weighted score (v1 publishable map)
- all     → all sequences after hard excludes (~390k; stretch)

The hard excludes (`dead`, `dupe`, `uned`, `dumb`, n_terms<8, empty name)
are applied uniformly across all scopes; these are sequences we never want
in any map.

The curated selection seeds with `core ∪ nice` (≈ the editor-curated
"interesting" set, ~7k sequences) and tops up to CURATED_TARGET_SIZE with
the highest-scoring sequences from the rest by:

    score = log1p(edit_count)
          + 1.0 * log1p(len(comments))
          + 0.5 * log1p(len(formulas))
          + 0.5 * log1p(len(examples))
          + 0.5 * len(code_languages)
          + 0.3 * log1p(n_references)

Weights are calibrated against a labeled negative class (programmatically
generated cellular-automaton bulk submissions, ~2,400 sequences) using
each signal's AUC for separating bulk from non-bulk. Signals previously
weighted that we removed:

  - `n_links`: bulk submissions bundle template-generated links (Wolfram
    MathWorld, b-file, etc.), giving the signal AUC 0.143 (anti-correlated).
  - `easy` keyword: AUC 0.319; describes computability rather than quality.
  - binary `has_code`: AUC 0.459; replaced by `len(code_languages)` count
    (AUC 0.811), which captures multi-implementer adoption.

`n_references` was reduced from weight 2.0 to 0.3: AUC of 0.384 (slightly
anti-correlated against the bulk class because their template includes a
single citation per row).

The score only ranks the topup pool (non-seed survivors), so it never sees
a row tagged `core` or `nice`; bonuses for those keywords would be inert
and have been omitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from config import (
    CURATED_TARGET_SIZE,
    DATA_DIR,
    RAW_PARQUET,
    SCOPE,
    SELECTED_PARQUET,
)

# Sequences that should never make it into any scope.
HARD_EXCLUDE_KEYWORDS: tuple[str, ...] = ("dead", "dupe", "uned", "dumb")
MIN_TERMS_VISIBLE = 8


# ── Scoring ──────────────────────────────────────────────────────────────────


def compute_score(df: pd.DataFrame) -> pd.Series:
    """Quality-weighted composite score, higher = more likely to include in curated.

    Only used to rank the topup pool (non-seed survivors), so `core` and `nice`
    bonuses are intentionally absent; those keywords route through the seed set.
    See module docstring for the negative-class calibration the weights came from.
    """
    comment_len = df["comments"].str.len().fillna(0)
    formula_len = df["formulas"].str.len().fillna(0)
    example_len = df["examples"].str.len().fillna(0)
    n_code_langs = df["code_languages"].apply(len).astype(float)

    return (
        np.log1p(df["edit_count"].astype(float))
        + 1.0 * np.log1p(comment_len)
        + 0.5 * np.log1p(formula_len)
        + 0.5 * np.log1p(example_len)
        + 0.5 * n_code_langs
        + 0.3 * np.log1p(df["n_references"].astype(float))
    )


# ── Selection ────────────────────────────────────────────────────────────────


def select(scope: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply hard excludes, compute scores, and filter to the requested scope."""
    print("Hard excludes:")
    for kw in HARD_EXCLUDE_KEYWORDS:
        n = df["keywords"].apply(lambda kws, k=kw: k in kws).sum()
        print(f"  keyword={kw:6s}  {n:>6,}")
    n_short = (df["n_terms_visible"] < MIN_TERMS_VISIBLE).sum()
    n_unnamed = (df["name"].str.len() == 0).sum()
    print(f"  n_terms < {MIN_TERMS_VISIBLE:<3} {n_short:>6,}")
    print(f"  empty name   {n_unnamed:>6,}")

    excluded_mask = (
        df["keywords"].apply(lambda kws: any(k in kws for k in HARD_EXCLUDE_KEYWORDS))
        | (df["n_terms_visible"] < MIN_TERMS_VISIBLE)
        | (df["name"].str.len() == 0)
    )
    n_excluded = int(excluded_mask.sum())
    kept = df.loc[~excluded_mask].copy()
    print(f"  → {n_excluded:,} rows excluded (some hit multiple criteria)")
    print(f"  → {len(kept):,} rows survive hard excludes\n")

    kept["select_score"] = compute_score(kept)

    if scope == "core":
        result = kept[kept["keywords"].apply(lambda kws: "core" in kws)].copy()

    elif scope == "curated":
        is_seed = kept["keywords"].apply(lambda kws: "core" in kws or "nice" in kws)
        seed = kept.loc[is_seed]
        rest = kept.loc[~is_seed]
        print(f"Seed set (core ∪ nice): {len(seed):,}")

        if len(seed) >= CURATED_TARGET_SIZE:
            print(f"  Seed alone exceeds target ({CURATED_TARGET_SIZE:,}); taking top by score")
            result = seed.nlargest(CURATED_TARGET_SIZE, "select_score").copy()
        else:
            n_extra = CURATED_TARGET_SIZE - len(seed)
            print(f"  Filling {n_extra:,} more slots from the remaining {len(rest):,} by score")
            extra = rest.nlargest(n_extra, "select_score")
            result = pd.concat([seed, extra], ignore_index=True)

    elif scope == "all":
        result = kept.copy()

    else:
        raise ValueError(f"Unknown OEIS_SCOPE: {scope!r}")

    result = result.sort_values("select_score", ascending=False).reset_index(drop=True)
    result["scope"] = scope
    return result


# ── Reporting ────────────────────────────────────────────────────────────────


def report(result: pd.DataFrame, scope: str) -> None:
    print(f"\nSelected {len(result):,} sequences for scope={scope}")

    if len(result) == 0:
        return

    score_min = result["select_score"].min()
    score_max = result["select_score"].max()
    score_med = result["select_score"].median()
    print(f"  score range: [{score_min:.2f}, {score_max:.2f}], median {score_med:.2f}")

    print("\n  Top 8 by score:")
    for _, row in result.head(8).iterrows():
        print(f"    {row['id']:8s}  score={row['select_score']:6.2f}  {row['name'][:60]}")

    if len(result) > 16:
        print("\n  Bottom 8 by score (the marginal 'just made it' sequences):")
        for _, row in result.tail(8).iterrows():
            print(f"    {row['id']:8s}  score={row['select_score']:6.2f}  {row['name'][:60]}")

    # Spot check: famous sequences should be in core and curated
    famous = [
        "A000040",  # primes
        "A000045",  # Fibonacci
        "A000108",  # Catalan
        "A000142",  # factorials
        "A000079",  # powers of 2
        "A000010",  # Euler totient
        "A000984",  # central binomial
    ]
    present = result["id"].isin(famous)
    print(f"\n  Famous sequences present: {present.sum()}/{len(famous)}")
    for fid in famous:
        in_set = fid in result["id"].values
        marker = "✓" if in_set else "✗"
        print(f"    {marker} {fid}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    if not RAW_PARQUET.exists():
        raise SystemExit(f"{RAW_PARQUET} not found — run `uv run python pipeline/01_parse.py` first")

    print(f"Reading {RAW_PARQUET.name}…")
    df = pd.read_parquet(RAW_PARQUET)
    print(f"  {len(df):,} rows loaded\n")

    print(f"Applying scope filter: OEIS_SCOPE={SCOPE}")
    result = select(SCOPE, df)

    report(result, SCOPE)

    # Atomic write
    print(f"\nWriting {len(result):,} rows to {SELECTED_PARQUET.name}…")
    tmp = SELECTED_PARQUET.with_suffix(".parquet.tmp")
    result.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(SELECTED_PARQUET)
    size_mb = SELECTED_PARQUET.stat().st_size / 1_000_000
    print(f"Done. {SELECTED_PARQUET} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
