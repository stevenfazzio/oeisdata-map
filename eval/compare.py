"""Compare Haiku and Sonnet classifications from the OEIS eval harness.

Reads ``data/eval/enriched_haiku.parquet`` and
``data/eval/enriched_sonnet.parquet`` (produced by ``eval/run_models.py``)
and reports:

1. Per-field pairwise agreement percentages
2. Per-field distribution per model (Haiku vs Sonnet, side by side)
3. Per-field confusion matrices (Haiku rows × Sonnet columns)
4. Top 30 most-surprising disagreements (sorted by number of fields differing)

Output goes to stdout AND to:
- ``data/eval/comparison_report.md`` (markdown for clean later review)
- ``data/eval/disagreements.csv`` (flat tabular for spreadsheet review)

Usage:
    python eval/compare.py
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import DATA_DIR  # noqa: E402
from pipeline.enrichment import ENUM_COLS  # noqa: E402

EVAL_DIR = DATA_DIR / "eval"
HAIKU_PARQUET = EVAL_DIR / "enriched_haiku.parquet"
SONNET_PARQUET = EVAL_DIR / "enriched_sonnet.parquet"
REPORT_MD = EVAL_DIR / "comparison_report.md"
DISAGREEMENTS_CSV = EVAL_DIR / "disagreements.csv"


# ── Loading ──────────────────────────────────────────────────────────────────


def load_pair() -> pd.DataFrame:
    """Inner-join the two model parquets on id, returning a wide DataFrame.

    Columns: id, name, values_preview_str, plus haiku_{field} and sonnet_{field}
    for each of the 4 enum fields.
    """
    if not HAIKU_PARQUET.exists():
        raise SystemExit(f"{HAIKU_PARQUET} not found — run eval/run_models.py first")
    if not SONNET_PARQUET.exists():
        raise SystemExit(f"{SONNET_PARQUET} not found — run eval/run_models.py first")

    haiku = pd.read_parquet(HAIKU_PARQUET)
    sonnet = pd.read_parquet(SONNET_PARQUET)
    print(f"Loaded haiku: {len(haiku)} rows, sonnet: {len(sonnet)} rows")

    # Use haiku's metadata columns (id, name, values_preview_str) as the join base.
    base_cols = ["id", "name", "values_preview_str"]
    base_cols = [c for c in base_cols if c in haiku.columns]

    h = haiku[base_cols + list(ENUM_COLS)].rename(columns={c: f"haiku_{c}" for c in ENUM_COLS})
    s = sonnet[["id"] + list(ENUM_COLS)].rename(columns={c: f"sonnet_{c}" for c in ENUM_COLS})

    merged = h.merge(s, on="id", how="inner")
    print(f"Joined on id: {len(merged)} rows")
    return merged


# ── Agreement metrics ────────────────────────────────────────────────────────


def field_agreement(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise agreement % for each enum field."""
    rows = []
    n = len(merged)
    for field in ENUM_COLS:
        h = merged[f"haiku_{field}"]
        s = merged[f"sonnet_{field}"]
        n_match = int((h == s).sum())
        n_haiku_null = int(h.isna().sum())
        n_sonnet_null = int(s.isna().sum())
        rows.append(
            {
                "field": field,
                "matches": n_match,
                "total": n,
                "agreement_%": round(100.0 * n_match / n, 1) if n else 0.0,
                "haiku_nulls": n_haiku_null,
                "sonnet_nulls": n_sonnet_null,
            }
        )
    return pd.DataFrame(rows)


def field_distribution(merged: pd.DataFrame, field: str) -> pd.DataFrame:
    """Side-by-side count + percentage of values for one field."""
    h = merged[f"haiku_{field}"].fillna("(null)").value_counts()
    s = merged[f"sonnet_{field}"].fillna("(null)").value_counts()
    all_vals = sorted(set(h.index) | set(s.index))
    n = len(merged)
    out = pd.DataFrame(
        {
            "value": all_vals,
            "haiku_n": [int(h.get(v, 0)) for v in all_vals],
            "haiku_%": [round(100.0 * h.get(v, 0) / n, 1) for v in all_vals],
            "sonnet_n": [int(s.get(v, 0)) for v in all_vals],
            "sonnet_%": [round(100.0 * s.get(v, 0) / n, 1) for v in all_vals],
        }
    )
    return out.sort_values("haiku_n", ascending=False).reset_index(drop=True)


def confusion_matrix(merged: pd.DataFrame, field: str) -> pd.DataFrame:
    """Build a Haiku-row × Sonnet-column count crosstab for one field."""
    h = merged[f"haiku_{field}"].fillna("(null)")
    s = merged[f"sonnet_{field}"].fillna("(null)")
    return pd.crosstab(h, s, rownames=[f"haiku_{field}"], colnames=[f"sonnet_{field}"])


# ── Disagreements ────────────────────────────────────────────────────────────


def find_disagreements(merged: pd.DataFrame) -> pd.DataFrame:
    """Return rows where at least one of the 4 enum fields differs.

    Adds a `n_disagree` column = number of fields differing (1..4).
    """
    diff_cols = []
    for field in ENUM_COLS:
        col = f"diff_{field}"
        merged[col] = merged[f"haiku_{field}"] != merged[f"sonnet_{field}"]
        diff_cols.append(col)

    merged["n_disagree"] = merged[diff_cols].sum(axis=1)
    disagreements = merged[merged["n_disagree"] > 0].copy()
    return disagreements.sort_values(["n_disagree", "id"], ascending=[False, True]).reset_index(drop=True)


def format_disagreement_row(row: pd.Series) -> str:
    """Format a single disagreement row as a multi-line block."""
    lines = [
        f"  {row['id']} ({int(row['n_disagree'])} disagreed): {str(row.get('name', ''))[:70]}",
        f"    values: {str(row.get('values_preview_str', ''))[:70]}",
    ]
    for field in ENUM_COLS:
        h = row[f"haiku_{field}"]
        s = row[f"sonnet_{field}"]
        marker = "≠" if h != s else "="
        lines.append(f"    {marker} {field:14s}  haiku={h!s:24s}  sonnet={s!s}")
    return "\n".join(lines)


# ── Report generation ───────────────────────────────────────────────────────


def render_report(merged: pd.DataFrame) -> str:
    """Build the full report string (used for both stdout and markdown file)."""
    out = StringIO()

    def w(s: str = "") -> None:
        out.write(s + "\n")

    w("# OEIS taxonomy eval — Haiku vs Sonnet")
    w()
    w(f"Sample size: **{len(merged)}** sequences (post inner-join)")
    w()

    # ── Section 1: agreement table ───────────────────────────────────────────
    w("## 1. Per-field agreement")
    w()
    agreement = field_agreement(merged)
    w("```")
    w(agreement.to_string(index=False))
    w("```")
    w()

    lowest = agreement.sort_values("agreement_%").iloc[0]
    highest = agreement.sort_values("agreement_%", ascending=False).iloc[0]
    w(
        f"**Lowest agreement:** `{lowest['field']}` at "
        f"{lowest['agreement_%']:.1f}% — taxonomy or prompt may need revision."
    )
    w(f"**Highest agreement:** `{highest['field']}` at {highest['agreement_%']:.1f}%.")
    w()

    # ── Section 2: distributions ─────────────────────────────────────────────
    w("## 2. Per-field distributions (Haiku vs Sonnet)")
    w()
    w("Watch for `other` / `unknown` overuse (sign of weak enum coverage) and ")
    w("strong asymmetry between models (sign of one model gravitating to a default).")
    w()
    for field in ENUM_COLS:
        w(f"### {field}")
        w("```")
        w(field_distribution(merged, field).to_string(index=False))
        w("```")
        w()

    # ── Section 3: confusion matrices ────────────────────────────────────────
    w("## 3. Confusion matrices (Haiku rows × Sonnet columns)")
    w()
    w("Off-diagonal cells with high counts reveal which value pairs the two models")
    w("conflate. The diagonal is agreement; off-diagonal is disagreement.")
    w()
    for field in ENUM_COLS:
        w(f"### {field}")
        w("```")
        w(confusion_matrix(merged, field).to_string())
        w("```")
        w()

    # ── Section 4: top disagreements ─────────────────────────────────────────
    disagreements = find_disagreements(merged)
    w(f"## 4. Top disagreements ({len(disagreements)} rows with at least one field differing)")
    w()
    if len(disagreements) == 0:
        w("**Perfect agreement on all 4 fields across all rows.** ")
        w("No taxonomy revision indicated.")
        return out.getvalue()

    w(f"Showing top {min(30, len(disagreements))} sorted by number of fields disagreed:")
    w()
    w("```")
    for _, row in disagreements.head(30).iterrows():
        w(format_disagreement_row(row))
        w()
    w("```")
    w()

    # Histogram of n_disagree
    hist = disagreements["n_disagree"].value_counts().sort_index()
    w("**Distribution of disagreement count:**")
    w()
    w("```")
    for n_diff, count in hist.items():
        bar = "█" * int(count)
        w(f"  {int(n_diff)} field(s) differ: {int(count):3d}  {bar}")
    w("```")
    w()

    return out.getvalue()


def write_disagreements_csv(merged: pd.DataFrame) -> int:
    """Write the full disagreements list to disagreements.csv. Returns row count."""
    disagreements = find_disagreements(merged.copy())
    if len(disagreements) == 0:
        return 0
    keep_cols = ["id", "name", "values_preview_str", "n_disagree"]
    keep_cols += [f"haiku_{f}" for f in ENUM_COLS]
    keep_cols += [f"sonnet_{f}" for f in ENUM_COLS]
    keep_cols = [c for c in keep_cols if c in disagreements.columns]
    DISAGREEMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    disagreements[keep_cols].to_csv(DISAGREEMENTS_CSV, index=False)
    return len(disagreements)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    merged = load_pair()
    if len(merged) == 0:
        raise SystemExit("inner-join produced 0 rows — check that both parquets share ids")

    report = render_report(merged)
    print(report)

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(report)
    print(f"\nWrote report to {REPORT_MD}")

    n_disagreements = write_disagreements_csv(merged)
    if n_disagreements > 0:
        print(f"Wrote {n_disagreements} disagreements to {DISAGREEMENTS_CSV}")
    else:
        print("No disagreements to write to CSV (perfect agreement)")


if __name__ == "__main__":
    main()
