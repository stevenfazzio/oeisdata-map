"""Sanity tests for pipeline/04b_retrofit.py math helpers.

These are not tuning tests — they verify directional correctness of the
Faruqui retrofit on a 10-node toy graph:

  - Linked pairs move closer (cosine similarity goes up).
  - Isolated nodes are left at their original vectors.
  - Output is unit-normalized per row.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

_spec = importlib.util.spec_from_file_location("retrofit_mod", PIPELINE_DIR / "04b_retrofit.py")
retrofit_mod = importlib.util.module_from_spec(_spec)
sys.modules["retrofit_mod"] = retrofit_mod
_spec.loader.exec_module(retrofit_mod)

build_adjacency = retrofit_mod.build_adjacency
retrofit = retrofit_mod.retrofit
eval_keyword_silhouette = retrofit_mod.eval_keyword_silhouette


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class TestRetrofitDirectionality:
    """10-node graph: nodes 0-1 linked, nodes 2-3 linked, nodes 4-9 isolated."""

    @classmethod
    def setup_class(cls):
        rng = np.random.default_rng(42)
        cls.n = 10
        cls.d = 16
        Q0 = rng.normal(size=(cls.n, cls.d)).astype(np.float32)
        cls.Q0 = _unit(Q0).astype(np.float32)
        # Two linked pairs, symmetric (0-1, 2-3). Everyone else isolated.
        cls.edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
        cls.adj_norm = build_adjacency(cls.n, cls.edges)
        train_deg = np.asarray((cls.adj_norm > 0).sum(axis=1)).ravel()
        cls.isolated_mask = train_deg == 0

    def test_isolated_mask_shape(self):
        # Nodes 4-9 should be isolated.
        assert self.isolated_mask.tolist() == [False, False, False, False] + [True] * 6

    def test_linked_pair_moves_closer(self):
        Q = retrofit(
            self.Q0,
            self.adj_norm,
            self.isolated_mask,
            alpha=0.8,
            n_iter=10,
        )
        # Before retrofit: cosine(Q0[0], Q0[1]). After: cosine(Q[0], Q[1]).
        before = float(self.Q0[0] @ self.Q0[1])
        after = float(Q[0] @ Q[1])
        assert after > before, f"expected linked pair to move closer; before={before:.4f}, after={after:.4f}"

    def test_both_linked_pairs_tighten(self):
        Q = retrofit(self.Q0, self.adj_norm, self.isolated_mask, alpha=0.8, n_iter=10)
        assert float(Q[0] @ Q[1]) > float(self.Q0[0] @ self.Q0[1])
        assert float(Q[2] @ Q[3]) > float(self.Q0[2] @ self.Q0[3])

    def test_isolated_nodes_unchanged(self):
        Q = retrofit(self.Q0, self.adj_norm, self.isolated_mask, alpha=0.8, n_iter=10)
        # Nodes 4-9 have no neighbors in the train graph → retrofit should leave
        # them exactly equal to Q0 (plus or minus the renormalization, which is
        # a no-op because Q0 is already unit-norm).
        for i in range(4, 10):
            assert np.allclose(Q[i], self.Q0[i], atol=1e-6), f"node {i} drifted"

    def test_output_is_unit_norm(self):
        Q = retrofit(self.Q0, self.adj_norm, self.isolated_mask, alpha=0.6, n_iter=5)
        norms = np.linalg.norm(Q, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_alpha_zero_is_identity(self):
        # α=0 means "don't move at all" — output must equal input.
        Q = retrofit(self.Q0, self.adj_norm, self.isolated_mask, alpha=0.0, n_iter=10)
        assert np.allclose(Q, self.Q0, atol=1e-6)

    def test_unlinked_pair_does_not_spuriously_tighten(self):
        # Nodes 0 and 2 are not linked and neither their neighbors are
        # transitively linked. Their similarity shouldn't meaningfully rise.
        Q = retrofit(self.Q0, self.adj_norm, self.isolated_mask, alpha=0.8, n_iter=10)
        before = float(self.Q0[0] @ self.Q0[2])
        after = float(Q[0] @ Q[2])
        # Allow tiny numerical drift in either direction, but nothing close to
        # the lift we expect for linked pairs.
        assert abs(after - before) < 0.1, f"unlinked pair drifted by {after - before:.4f}"


class TestBuildAdjacency:
    def test_empty(self):
        adj = build_adjacency(5, np.zeros((0, 2), dtype=np.int32))
        assert adj.shape == (5, 5)
        assert adj.nnz == 0

    def test_symmetric(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
        adj = build_adjacency(3, edges)
        dense = adj.toarray()
        # Row-normalized symmetric adjacency — every non-zero row sums to 1.
        row_sums = dense.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
        # Node 0 has one neighbor (1) → row 0 is [0, 1, 0].
        assert np.allclose(dense[0], [0.0, 1.0, 0.0])
        # Node 1 has two neighbors (0, 2) → row 1 is [0.5, 0, 0.5].
        assert np.allclose(dense[1], [0.5, 0.0, 0.5])
        # Node 2 has one neighbor (1) → row 2 is [0, 1, 0].
        assert np.allclose(dense[2], [0.0, 1.0, 0.0])

    def test_duplicate_edges_collapsed(self):
        # Two copies of the same edge should still row-normalize correctly.
        edges = np.array([[0, 1], [0, 1]], dtype=np.int32)
        adj = build_adjacency(2, edges)
        dense = adj.toarray()
        assert np.allclose(dense, [[0.0, 1.0], [1.0, 0.0]])


class TestKeywordSilhouette:
    """Closed-form silhouette computation matches a brute-force pair-sum."""

    @staticmethod
    def _brute_force(Q, idx, all_idx):
        """Mean intra-class and mean inter-class cosine, summed pairwise.

        Q is unit-norm so cosine = dot. Used to verify the closed-form
        identities in eval_keyword_silhouette.
        """
        in_set = set(idx.tolist())
        intra_pairs = [(i, j) for i in idx for j in idx if i < j]
        inter_pairs = [(i, j) for i in idx for j in all_idx if j not in in_set]
        # Mean over UNORDERED intra pairs (each pair counted once); the
        # closed-form uses ORDERED pairs (each counted twice) divided by
        # nk*(nk-1), which is mathematically equal to the unordered mean.
        intra_mean = np.mean([float(Q[i] @ Q[j]) for i, j in intra_pairs]) if intra_pairs else 0.0
        inter_mean = np.mean([float(Q[i] @ Q[j]) for i, j in inter_pairs]) if inter_pairs else 0.0
        return intra_mean, inter_mean

    def test_closed_form_matches_brute_force(self):
        rng = np.random.default_rng(7)
        n, d = 20, 8
        Q = rng.normal(size=(n, d)).astype(np.float32)
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        # Two overlapping classes
        kw_a = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        kw_b = np.array([3, 4, 5, 6, 7, 8], dtype=np.int32)
        result = eval_keyword_silhouette(Q, {"a": kw_a, "b": kw_b})

        all_idx = np.arange(n)
        for entry in result["per_keyword"]:
            kw = entry["keyword"]
            idx = kw_a if kw == "a" else kw_b
            expected_intra, expected_inter = self._brute_force(Q, idx, all_idx)
            assert abs(entry["intra"] - expected_intra) < 1e-5, f"{kw} intra mismatch"
            assert abs(entry["inter"] - expected_inter) < 1e-5, f"{kw} inter mismatch"

    def test_skips_singletons(self):
        rng = np.random.default_rng(11)
        Q = rng.normal(size=(5, 4)).astype(np.float32)
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        # Class with one member has no intra-pair to score; should be skipped.
        result = eval_keyword_silhouette(
            Q, {"singleton": np.array([0], dtype=np.int32), "pair": np.array([1, 2], dtype=np.int32)}
        )
        keywords = [e["keyword"] for e in result["per_keyword"]]
        assert "singleton" not in keywords
        assert "pair" in keywords
        assert result["summary"]["n_keywords_evaluated"] == 1

    def test_homogeneous_class_has_max_intra(self):
        # All members of class A point in the same direction → intra cosine = 1.
        Q = np.zeros((4, 3), dtype=np.float32)
        Q[0:2] = np.array([1, 0, 0])  # class A, identical
        Q[2] = np.array([0, 1, 0])  # not in class
        Q[3] = np.array([0, 0, 1])  # not in class
        result = eval_keyword_silhouette(Q, {"a": np.array([0, 1], dtype=np.int32)})
        entry = result["per_keyword"][0]
        assert abs(entry["intra"] - 1.0) < 1e-5
        # Inter is mean over (in × out): both cross-products are zero.
        assert abs(entry["inter"] - 0.0) < 1e-5
        assert entry["gap"] > 0.99

    def test_weighted_gap_is_size_weighted_mean(self):
        # Two classes of different sizes with known gaps; verify weighted mean.
        rng = np.random.default_rng(13)
        n, d = 30, 6
        Q = rng.normal(size=(n, d)).astype(np.float32)
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        kw_small = np.array([0, 1], dtype=np.int32)
        kw_large = np.array(list(range(2, 12)), dtype=np.int32)
        result = eval_keyword_silhouette(Q, {"small": kw_small, "large": kw_large})
        per_kw = {e["keyword"]: e for e in result["per_keyword"]}
        expected_weighted = (2 * per_kw["small"]["gap"] + 10 * per_kw["large"]["gap"]) / 12
        assert abs(result["summary"]["weighted_gap"] - expected_weighted) < 1e-6
        # And mean_gap is the unweighted average.
        expected_mean = (per_kw["small"]["gap"] + per_kw["large"]["gap"]) / 2
        assert abs(result["summary"]["mean_gap"] - expected_mean) < 1e-6
