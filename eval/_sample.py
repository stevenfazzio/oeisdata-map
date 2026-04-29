"""Sampling helper for the OEIS map eval harness.

This module duplicates two constants from ``pipeline/02_select.py``
(``HARD_EXCLUDE_KEYWORDS`` and ``MIN_TERMS_VISIBLE``) because that file lives
under a digit-prefix module name (``02_select.py``) and cannot be imported
directly. Keep these in sync if 02_select.py changes its hard-exclude logic.

The eval is intentionally decoupled from the production scope state — it
samples directly from ``raw_sequences.parquet``, never from
``selected.parquet``, so running the eval does not depend on (or disturb)
the user's current ``OEIS_SCOPE``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Duplicated from pipeline/02_select.py:41-42 — keep in sync.
HARD_EXCLUDE_KEYWORDS: tuple[str, ...] = ("dead", "dupe", "uned", "dumb")
MIN_TERMS_VISIBLE = 8

# Columns the classifier needs (everything else is dropped from the sample
# parquet to keep it small and avoid leaking unrelated data). The `author` and
# `last_edited` fields were added in Sprint 6 to give the model a stronger
# signal for the `origin_era` enum (Sprint 3 lesson #5).
EVAL_COLUMNS: tuple[str, ...] = (
    "id",
    "name",
    "comments",
    "formulas",
    "keywords",
    "offset",
    "values_preview_str",
    "author",
    "last_edited",
)


def load_sample(raw_path: Path, n: int, seed: int) -> pd.DataFrame:
    """Load raw_sequences.parquet, apply hard excludes, return n random rows.

    The hard excludes match ``pipeline/02_select.py:71-90`` exactly so the
    eval sample is drawn from the same population that the curated scope
    would consider. The sample is reproducible: the same ``(n, seed)`` pair
    always produces the same rows.
    """
    print(f"Loading {raw_path.name}…")
    df = pd.read_parquet(raw_path)
    print(f"  {len(df):,} rows loaded")

    excluded_mask = (
        df["keywords"].apply(lambda kws: any(k in kws for k in HARD_EXCLUDE_KEYWORDS))
        | (df["n_terms_visible"] < MIN_TERMS_VISIBLE)
        | (df["name"].str.len() == 0)
    )
    n_excluded = int(excluded_mask.sum())
    eligible = df.loc[~excluded_mask]
    print(f"  {n_excluded:,} hard-excluded; {len(eligible):,} eligible for sampling")

    if n > len(eligible):
        raise SystemExit(
            f"requested n={n} but only {len(eligible)} eligible rows; reduce --n or check raw_sequences.parquet"
        )

    sample = eligible.sample(n=n, random_state=seed).reset_index(drop=True)
    print(f"  sampled {n} rows with seed={seed}")

    keep = [c for c in EVAL_COLUMNS if c in sample.columns]
    missing = [c for c in EVAL_COLUMNS if c not in sample.columns]
    if missing:
        print(f"  WARNING: sample missing expected columns: {missing}")
    return sample[keep]
