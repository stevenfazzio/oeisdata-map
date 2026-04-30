"""Retrofit Cohere embeddings against OEIS %Y cross-reference edges.

Takes frozen `embeddings.npz` + `y_edges.parquet` and produces
`embeddings_retrofit.npz` — a Laplacian-smoothed variant where sequences
connected by `%Y` edges are pulled toward each other in embedding space.
Drop-in for stage 05 when `OEIS_EMBEDDINGS_FILE=embeddings_retrofit.npz`.

Algorithm (Faruqui et al. NAACL 2015, simplified):

    Q ← Q0
    for _ in range(n_iter):
        Q ← (1 - α) · Q0 + α · (A_norm @ Q)
        Q[isolated] ← Q0[isolated]   # no-op for nodes with no train edges
    Q ← Q / ||Q||                     # renormalize per row

`A_norm` is the row-normalized adjacency of the **train** edges only. We hold
out 10% of edges for evaluation — Hits@10 / Hits@100 / MRR against baseline.

Grid: α ∈ {0.2, 0.4, 0.6, 0.8, 0.9, 1.0} × n_iter ∈ {5, 10}. Winner = max Hits@10.

**Multi-signal eval.** Three metrics report per grid point:
  - Held-out %Y Hits@10 / Hits@100 / MRR — graph self-consistency
    (training and eval use the same edge graph, just a 90/10 split).
  - Content-keyword silhouette gap — intra-class minus inter-class mean
    cosine similarity, averaged over OEIS content keywords (tabl, mult,
    sign, …; see pipeline/keywords.py). This is independent of %Y because
    keywords aren't in the embed text or the smoothing graph.
The winner-selection rule is still max Hits@10 (no automatic multi-objective
pick); the keyword gap is surfaced for inspection.

Inputs:
  - data/enriched.parquet       (id + keywords; defines row order)
  - data/embeddings.npz         (baseline Q0, float32 (N, 512))
  - data/embeddings_index.npy   (must match enriched row order)
  - data/y_edges.parquet        (global edge list from stage 01b)

Outputs:
  - data/embeddings_retrofit.npz   (winner Q under key "embeddings")
  - data/y_edges_split.parquet     (in-scope edges + split column)
  - data/retrofit_eval.json        (baseline + per-grid-point metrics)

Env vars:
  - OEIS_SCOPE   (unused directly; scope is implicit via enriched.parquet's row set)
  - OEIS_RETROFIT_SEED  (optional; default 42)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from config import (
    DATA_DIR,
    EMBEDDINGS_INDEX_NPY,
    EMBEDDINGS_RETROFIT_NPZ,
    ENRICHED_PARQUET,
    RETROFIT_EVAL_JSON,
    Y_EDGES_PARQUET,
    Y_EDGES_SPLIT_PARQUET,
)
from keywords import CONTENT_KEYWORDS

# ── Constants ────────────────────────────────────────────────────────────────

# Baseline embeddings path is hardcoded here so we always read the un-retrofitted
# vectors regardless of whether the OEIS_EMBEDDINGS_FILE env var is pointing at a
# retrofit file for downstream stages.
BASELINE_EMBEDDINGS_NPZ = DATA_DIR / "embeddings.npz"

ALPHA_GRID = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
# n_iter sensitivity is empirically zero on this graph (curated grid showed
# 5/10/20 within 0.0005 Hits@10 at every α). Two points are enough to confirm
# convergence; if a future graph behaves differently, add 20 back.
N_ITER_GRID = [5, 10]

HOLDOUT_FRACTION = 0.10
EVAL_BLOCK = 256  # held-out sources per similarity batch

# Winner selection: max Hits@10 across the grid. Earlier iterations of this
# stage tried to protect MRR with a regression budget, on the theory that
# MRR-preserving retrofits would produce better maps. Empirically the opposite:
# higher α (stronger smoothing) produces visibly better maps because it
# sharpens local density, which stage 06's HDBSCAN clusterer is very sensitive
# to. Hits@100 is actually the most-correlated metric with map quality, but
# it trends the same direction as Hits@10 on this grid, so Hits@10 is a fine
# target. See the curated run in ~/.claude/plans/mighty-chasing-pebble.md.
SEED = int(os.environ.get("OEIS_RETROFIT_SEED", "42"))


# ── Atomic save helpers (copied from 04_embed.py pattern) ────────────────────


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(f, **arrays)
    tmp_path.replace(path)


def _atomic_write_parquet(path: Path, df: pd.DataFrame) -> None:
    tmp_path = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False, compression="zstd")
    tmp_path.replace(path)


def _atomic_write_json(path: Path, obj) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(obj, indent=2))
    tmp_path.replace(path)


# ── Retrofit core ────────────────────────────────────────────────────────────


def build_adjacency(n: int, edges: np.ndarray) -> sp.csr_matrix:
    """Return an N×N row-normalized symmetric adjacency matrix.

    `edges` is a (E, 2) int32 array of row indices. Each edge is added in both
    directions so the result is symmetric. Rows with no neighbors stay all-zero
    (row-normalization produces NaN otherwise; we explicitly handle that).
    """
    if len(edges) == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    rows = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int32)
    cols = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int32)
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    adj.sum_duplicates()  # collapse any accidental double-edges
    # Row-normalize: divide each row by its sum (degree).
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv_deg = np.zeros_like(deg, dtype=np.float32)
    nonzero = deg > 0
    inv_deg[nonzero] = (1.0 / deg[nonzero]).astype(np.float32)
    # Multiply each row by inv_deg[row] — scipy's left-multiply by diag does this.
    adj_norm = sp.diags(inv_deg) @ adj
    return adj_norm.astype(np.float32).tocsr()


def retrofit(
    Q0: np.ndarray,
    adj_norm: sp.csr_matrix,
    isolated_mask: np.ndarray,
    alpha: float,
    n_iter: int,
) -> np.ndarray:
    """Faruqui retrofit. Returns a new (N, D) float32 array."""
    Q = Q0.copy()
    for _ in range(n_iter):
        # Mean of neighbors (row-normalized adjacency @ Q). Isolated rows → zero.
        M = adj_norm @ Q
        Q = (1.0 - alpha) * Q0 + alpha * M
        # Isolated nodes shouldn't drift: their M=0 would shrink them.
        Q[isolated_mask] = Q0[isolated_mask]
    # Renormalize to per-row unit length. Guard against zero rows.
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Q = (Q / norms).astype(np.float32)
    return Q


# ── Evaluation ───────────────────────────────────────────────────────────────


def build_keyword_membership(keywords_series: pd.Series) -> dict[str, np.ndarray]:
    """Map each content keyword to the int32 array of row indices having it.

    Skips keywords with fewer than 2 members (no intra-class pair to score).
    Skips keywords absent from CONTENT_KEYWORDS — only the orthogonal,
    content-bearing keywords are useful as eval signal.
    """
    membership: dict[str, list[int]] = {kw: [] for kw in CONTENT_KEYWORDS}
    for i, kws in enumerate(keywords_series):
        if kws is None:
            continue
        try:
            kw_set = {str(k) for k in kws}
        except TypeError:
            continue
        for kw in kw_set & CONTENT_KEYWORDS:
            membership[kw].append(i)
    out: dict[str, np.ndarray] = {}
    for kw, lst in membership.items():
        if len(lst) >= 2:
            out[kw] = np.asarray(lst, dtype=np.int32)
    return out


def eval_keyword_silhouette(
    Q: np.ndarray,
    keyword_to_indices: dict[str, np.ndarray],
) -> dict:
    """Per-keyword intra-class mean cosine similarity vs inter-class.

    For unit-norm Q, intra/inter mean cosine reduce to closed-form sums of
    the per-class centroid (sum of member rows). For class S with sum vector
    M_S = sum_{i in S} Q[i], the mean intra-class cosine over distinct pairs
    is (||M_S||^2 - |S|) / (|S| * (|S| - 1)) (because diagonal contributes
    |S| from the unit norms). The mean inter-class cosine over all (in, out)
    pairs is M_S · (M_total - M_S) / (|S| * (n - |S|)).

    Returns:
      summary  — n_keywords_evaluated, mean_intra, mean_inter, mean_gap
                 (unweighted), weighted_gap (weighted by |S|).
      per_keyword — list of {keyword, n, intra, inter, gap}, sorted by
                    keyword name.

    Q is assumed to be unit-norm (caller ensures this; baseline + retrofit
    both renormalize at the end). Both means use the closed-form identities
    above, so this is O(n*d + k*d) regardless of class size.
    """
    n = len(Q)
    M_total = Q.sum(axis=0)

    per_keyword: list[dict] = []
    intra_sum = 0.0
    inter_sum = 0.0
    weighted_intra = 0.0
    weighted_inter = 0.0
    total_weight = 0

    for kw in sorted(keyword_to_indices):
        idx = keyword_to_indices[kw]
        nk = int(len(idx))
        if nk < 2 or nk >= n:
            continue
        M_k = Q[idx].sum(axis=0)
        # Intra: closed form for unit-norm Q.
        intra = (float(M_k @ M_k) - nk) / (nk * (nk - 1))
        # Inter: dot of class sum with the rest of the corpus's sum.
        inter = float(M_k @ (M_total - M_k)) / (nk * (n - nk))
        gap = intra - inter
        per_keyword.append(
            {
                "keyword": kw,
                "n": nk,
                "intra": intra,
                "inter": inter,
                "gap": gap,
            }
        )
        intra_sum += intra
        inter_sum += inter
        weighted_intra += nk * intra
        weighted_inter += nk * inter
        total_weight += nk

    n_kw = len(per_keyword)
    summary = {
        "n_keywords_evaluated": n_kw,
        "mean_intra": intra_sum / n_kw if n_kw else 0.0,
        "mean_inter": inter_sum / n_kw if n_kw else 0.0,
        "mean_gap": (intra_sum - inter_sum) / n_kw if n_kw else 0.0,
        "weighted_gap": (weighted_intra - weighted_inter) / total_weight if total_weight else 0.0,
    }
    return {"summary": summary, "per_keyword": per_keyword}


def eval_hits(
    Q: np.ndarray,
    eval_pairs: np.ndarray,
) -> dict:
    """Compute Hits@10, Hits@100, MRR for retrieval of the paired node.

    `eval_pairs` is (P, 2) int32: each row is (source, target). For each row,
    we rank all other nodes by cosine-similarity to `Q[source]`, then check
    where `Q[target]` lands.

    Both directions of an undirected held-out edge are expected to appear in
    `eval_pairs` (we duplicate upstream). No extra symmetry handling here.

    Q must be unit-norm for cosine → dot equivalence.
    """
    p = len(eval_pairs)
    if p == 0:
        return {"n_pairs": 0, "hits_at_10": 0.0, "hits_at_100": 0.0, "mrr": 0.0}

    hits10 = 0
    hits100 = 0
    recip_rank_sum = 0.0

    for start in range(0, p, EVAL_BLOCK):
        end = min(start + EVAL_BLOCK, p)
        src_idx = eval_pairs[start:end, 0]
        tgt_idx = eval_pairs[start:end, 1]

        # (B, N) similarity matrix
        sims = Q[src_idx] @ Q.T
        # Exclude self
        sims[np.arange(end - start), src_idx] = -np.inf

        # Rank of tgt in each row. Straightforward: count nodes with strictly
        # greater similarity than the target (ties broken conservatively by
        # counting them as above, which gives a worst-case rank).
        tgt_sims = sims[np.arange(end - start), tgt_idx]
        # Broadcast comparison: (B, N) > (B, 1)
        ranks = (sims > tgt_sims[:, None]).sum(axis=1) + 1  # 1-indexed

        hits10 += int((ranks <= 10).sum())
        hits100 += int((ranks <= 100).sum())
        recip_rank_sum += float((1.0 / ranks).sum())

    return {
        "n_pairs": int(p),
        "hits_at_10": hits10 / p,
        "hits_at_100": hits100 / p,
        "mrr": recip_rank_sum / p,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    # 1. Load enriched row order + keywords (for the silhouette eval).
    if not ENRICHED_PARQUET.exists():
        raise SystemExit(f"{ENRICHED_PARQUET} not found — run the baseline pipeline through stage 03 first")
    print(f"Reading {ENRICHED_PARQUET.name}…")
    df = pd.read_parquet(ENRICHED_PARQUET, columns=["id", "keywords"])
    ids: list[str] = df["id"].tolist()
    n = len(ids)
    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    print(f"  {n:,} in-scope sequences")

    print("Building keyword membership for silhouette eval…")
    keyword_membership = build_keyword_membership(df["keywords"])
    print(f"  {len(keyword_membership)} content keywords with >=2 members:")
    for kw in sorted(keyword_membership):
        print(f"    {kw:8s} {len(keyword_membership[kw]):>5,} sequences")

    # 2. Load baseline embeddings and verify alignment.
    if not BASELINE_EMBEDDINGS_NPZ.exists():
        raise SystemExit(f"{BASELINE_EMBEDDINGS_NPZ} not found — run stage 04 first")
    if not EMBEDDINGS_INDEX_NPY.exists():
        raise SystemExit(f"{EMBEDDINGS_INDEX_NPY} not found — run stage 04 first")
    print(f"Reading {BASELINE_EMBEDDINGS_NPZ.name}…")
    Q0 = np.load(BASELINE_EMBEDDINGS_NPZ)["embeddings"].astype(np.float32)
    emb_ids = np.load(EMBEDDINGS_INDEX_NPY)
    if len(Q0) != n:
        raise SystemExit(f"Baseline embeddings have {len(Q0)} rows, enriched has {n} — re-run stage 04")
    if [str(x) for x in emb_ids] != ids:
        raise SystemExit("embeddings_index.npy row order does not match enriched.parquet — re-run stage 04")
    print(f"  Q0 shape={Q0.shape} dtype={Q0.dtype}")

    # Baseline norm sanity check (Cohere embed-v4.0 is expected to be unit-norm).
    sample_norms = np.linalg.norm(Q0[: min(100, n)], axis=1)
    print(
        f"  baseline row norms: mean={sample_norms.mean():.4f}, min={sample_norms.min():.4f}, "
        f"max={sample_norms.max():.4f}"
    )
    # Renormalize Q0 defensively so cosine ≡ dot throughout eval.
    base_norms = np.linalg.norm(Q0, axis=1, keepdims=True)
    base_norms[base_norms == 0] = 1.0
    Q0 = (Q0 / base_norms).astype(np.float32)

    # 3. Load global %Y edges and filter to in-scope endpoints.
    if not Y_EDGES_PARQUET.exists():
        raise SystemExit(f"{Y_EDGES_PARQUET} not found — run `uv run python pipeline/01b_parse_y_edges.py` first")
    print(f"Reading {Y_EDGES_PARQUET.name}…")
    edges_df = pd.read_parquet(Y_EDGES_PARQUET)
    print(f"  {len(edges_df):,} total edges globally")

    in_scope_mask = edges_df["from_id"].isin(id_to_idx) & edges_df["to_id"].isin(id_to_idx)
    edges_df = edges_df[in_scope_mask].reset_index(drop=True)
    print(f"  {len(edges_df):,} edges with both endpoints in scope")

    if len(edges_df) == 0:
        raise SystemExit("No %Y edges survive the scope filter — retrofit would be a no-op. Check Y_EDGES_PARQUET.")

    # Convert to int32 index pairs for the sparse builder.
    edges_arr = np.column_stack(
        [
            edges_df["from_id"].map(id_to_idx).to_numpy(dtype=np.int32),
            edges_df["to_id"].map(id_to_idx).to_numpy(dtype=np.int32),
        ]
    )

    # 4. Deterministic train/holdout split.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(edges_arr))
    n_hold = max(1, int(round(len(edges_arr) * HOLDOUT_FRACTION)))
    hold_idx = perm[:n_hold]
    train_idx = perm[n_hold:]
    train_edges = edges_arr[train_idx]
    hold_edges = edges_arr[hold_idx]
    print(f"  Split: {len(train_edges):,} train / {len(hold_edges):,} holdout (seed={SEED})")

    split_df = edges_df.assign(split=np.where(np.isin(np.arange(len(edges_arr)), hold_idx), "holdout", "train"))
    print(f"Writing {Y_EDGES_SPLIT_PARQUET.name}…")
    _atomic_write_parquet(Y_EDGES_SPLIT_PARQUET, split_df)

    # 5. Build train adjacency, identify isolated-in-train nodes.
    print("Building train adjacency…")
    adj_norm = build_adjacency(n, train_edges)
    train_deg = np.asarray((adj_norm > 0).sum(axis=1)).ravel()
    isolated_mask = train_deg == 0
    print(f"  {(~isolated_mask).sum():,}/{n:,} nodes have ≥1 train neighbor")

    # 6. Build the held-out eval pair list.
    # Both directions of an undirected held-out edge contribute, so we also
    # want (dst, src) to measure symmetry. Filter to pairs where BOTH endpoints
    # have training edges — otherwise retrofit is a no-op for that node and we'd
    # just measure baseline numbers, diluting the lift signal.
    both_have_train = (~isolated_mask[hold_edges[:, 0]]) & (~isolated_mask[hold_edges[:, 1]])
    eval_base = hold_edges[both_have_train]
    eval_pairs = np.concatenate([eval_base, eval_base[:, ::-1]], axis=0)
    print(f"  {len(eval_base):,} evaluable held-out edges ({len(eval_pairs):,} directed eval pairs)")

    if len(eval_pairs) == 0:
        raise SystemExit(
            "No evaluable held-out edges — graph is too sparse for a meaningful split. "
            "Consider lowering HOLDOUT_FRACTION or checking y_edges.parquet."
        )

    # 7. Baseline eval (Q0 already unit-normed).
    print("\nEvaluating baseline…")
    t0 = time.time()
    baseline_metrics = eval_hits(Q0, eval_pairs)
    baseline_silhouette = eval_keyword_silhouette(Q0, keyword_membership)
    print(
        f"  baseline Hits@10={baseline_metrics['hits_at_10']:.4f}  "
        f"Hits@100={baseline_metrics['hits_at_100']:.4f}  "
        f"MRR={baseline_metrics['mrr']:.4f}  "
        f"keyword_gap(weighted)={baseline_silhouette['summary']['weighted_gap']:.4f}  "
        f"({time.time() - t0:.1f}s)"
    )
    print("  per-keyword baseline (intra - inter):")
    for entry in baseline_silhouette["per_keyword"]:
        print(
            f"    {entry['keyword']:8s} n={entry['n']:>5,} "
            f"intra={entry['intra']:.4f}  inter={entry['inter']:.4f}  gap={entry['gap']:+.4f}"
        )

    # 8. Grid search. Keep only the current winner's Q in memory so we don't
    # balloon RAM at full scope (394k × 512 × 4B ≈ 800MB per candidate).
    grid_results: list[dict] = []
    best_entry: dict | None = None
    best_Q: np.ndarray | None = None

    for alpha in ALPHA_GRID:
        for n_iter in N_ITER_GRID:
            label = f"α={alpha} iter={n_iter}"
            t0 = time.time()
            Q = retrofit(Q0, adj_norm, isolated_mask, alpha=alpha, n_iter=n_iter)
            t_fit = time.time() - t0
            t0 = time.time()
            metrics = eval_hits(Q, eval_pairs)
            silhouette = eval_keyword_silhouette(Q, keyword_membership)
            t_eval = time.time() - t0
            sil_summary = silhouette["summary"]
            base_sil = baseline_silhouette["summary"]
            entry = {
                "alpha": alpha,
                "n_iter": n_iter,
                "hits_at_10": metrics["hits_at_10"],
                "hits_at_100": metrics["hits_at_100"],
                "mrr": metrics["mrr"],
                "delta_hits_at_10": metrics["hits_at_10"] - baseline_metrics["hits_at_10"],
                "delta_hits_at_100": metrics["hits_at_100"] - baseline_metrics["hits_at_100"],
                "delta_mrr": metrics["mrr"] - baseline_metrics["mrr"],
                "keyword_silhouette": sil_summary,
                "delta_keyword_weighted_gap": sil_summary["weighted_gap"] - base_sil["weighted_gap"],
                "delta_keyword_mean_gap": sil_summary["mean_gap"] - base_sil["mean_gap"],
                "keyword_silhouette_per_keyword": silhouette["per_keyword"],
                "seconds_fit": round(t_fit, 2),
                "seconds_eval": round(t_eval, 2),
            }
            grid_results.append(entry)
            print(
                f"  {label}: Hits@10={metrics['hits_at_10']:.4f} "
                f"(Δ{entry['delta_hits_at_10']:+.4f})  "
                f"Hits@100={metrics['hits_at_100']:.4f} "
                f"(Δ{entry['delta_hits_at_100']:+.4f})  "
                f"MRR={metrics['mrr']:.4f}  "
                f"keyword_gap(w)={sil_summary['weighted_gap']:.4f} "
                f"(Δ{entry['delta_keyword_weighted_gap']:+.4f})  "
                f"[{t_fit:.1f}s fit, {t_eval:.1f}s eval]"
            )

            if best_entry is None or entry["hits_at_10"] > best_entry["hits_at_10"]:
                best_entry = entry
                best_Q = Q.copy()

    assert best_entry is not None and best_Q is not None

    # 9. Sanity check the winner and save.
    assert best_Q.shape == Q0.shape, "winner: shape mismatch"
    assert best_Q.dtype == np.float32, "winner: dtype mismatch"
    assert not np.isnan(best_Q).any(), "winner: contains NaN"
    norms = np.linalg.norm(best_Q, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"winner: non-unit norms {norms.min()}..{norms.max()}"

    print(
        f"\nWinner: α={best_entry['alpha']} n_iter={best_entry['n_iter']}  "
        f"Hits@10={best_entry['hits_at_10']:.4f} (Δ{best_entry['delta_hits_at_10']:+.4f})  "
        f"Hits@100={best_entry['hits_at_100']:.4f} (Δ{best_entry['delta_hits_at_100']:+.4f})  "
        f"MRR={best_entry['mrr']:.4f} (Δ{best_entry['delta_mrr']:+.4f})"
    )

    print(f"Writing {EMBEDDINGS_RETROFIT_NPZ.name}…")
    _atomic_savez(EMBEDDINGS_RETROFIT_NPZ, embeddings=best_Q)

    eval_json = {
        "seed": SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "n_sequences": int(n),
        "n_edges_in_scope": int(len(edges_arr)),
        "n_train_edges": int(len(train_edges)),
        "n_holdout_edges": int(len(hold_edges)),
        "n_eval_pairs": int(len(eval_pairs)),
        "baseline": baseline_metrics,
        "baseline_keyword_silhouette": baseline_silhouette,
        "grid": grid_results,
        "winner": best_entry,
    }
    print(f"Writing {RETROFIT_EVAL_JSON.name}…")
    _atomic_write_json(RETROFIT_EVAL_JSON, eval_json)

    size_mb = EMBEDDINGS_RETROFIT_NPZ.stat().st_size / 1_000_000
    print(f"\nDone. {EMBEDDINGS_RETROFIT_NPZ} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
