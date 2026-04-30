"""Summarize the Toponymy label outputs of the three α-variants side by side.

Run after eval/compare_alphas.py has produced data/labels_compare_*.parquet.
Reports per-layer cluster counts, sample of coarsest layer, and the
"Unlabelled" fraction per layer (a high fraction signals the clusterer
couldn't find a clean theme, which is a quality signal independent of
keyword silhouette and Hits@10).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
LABELS = {
    "baseline (α=0)": DATA / "labels_compare_baseline.parquet",
    "α=0.2": DATA / "labels_compare_alpha_02.parquet",
    "α=0.8": DATA / "labels_compare_alpha_08.parquet",
}


def summarize(label: str, df: pd.DataFrame) -> None:
    print(f"\n=== {label} ===")
    layer_cols = sorted([c for c in df.columns if c.startswith("label_layer_")])
    print(f"layers: {len(layer_cols)}  rows: {len(df):,}")

    print(f"\n{'layer':>14} {'#unique':>8} {'#unlabelled':>12} {'%unlabelled':>11}")
    for col in layer_cols:
        vc = df[col].value_counts(dropna=False)
        n_unique = df[col].nunique()
        n_unlab = int(vc.get("Unlabelled", 0))
        pct_unlab = 100 * n_unlab / len(df)
        print(f"{col:>14} {n_unique:>8d} {n_unlab:>12,d} {pct_unlab:>10.1f}%")

    # Coarsest layer label sample (sorted by support, top 10)
    coarsest = layer_cols[0]
    print(f"\nCoarsest layer ({coarsest}) — top labels by support:")
    vc = df[coarsest].value_counts(dropna=False)
    for name, cnt in vc.head(10).items():
        print(f"  {cnt:>5,d}  {name}")


def main() -> None:
    missing = [p for p in LABELS.values() if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing labels file(s); run eval/compare_alphas.py first:\n  " + "\n  ".join(str(p) for p in missing)
        )

    for label, path in LABELS.items():
        df = pd.read_parquet(path)
        summarize(label, df)


if __name__ == "__main__":
    main()
