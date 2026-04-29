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
