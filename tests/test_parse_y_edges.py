"""Tests for pipeline/01b_parse_y_edges.py against fixture .seq files.

Covers:
- Extraction of A-numbers from real `%Y` lines (A000001, A000045, A395003).
- Self-reference filtering.
- Sequences with no `%Y` fields produce empty lists (none in fixtures, so we
  fall back to synthetic text).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Add pipeline/ to sys.path so the loaded script's `from config import ...` resolves
REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

# Load pipeline/01b_parse_y_edges.py via importlib (digit-prefix module names aren't importable)
_spec = importlib.util.spec_from_file_location("parse_y_edges", PIPELINE_DIR / "01b_parse_y_edges.py")
parse_y_edges = importlib.util.module_from_spec(_spec)
sys.modules["parse_y_edges"] = parse_y_edges
_spec.loader.exec_module(parse_y_edges)

extract_y_edges = parse_y_edges.extract_y_edges

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestExtractA000001:
    """A000001 has 5 %Y lines citing group-theory sequences."""

    @classmethod
    def setup_class(cls):
        cls.pairs = extract_y_edges(FIXTURES / "A000001.seq")

    def test_nonempty(self):
        assert len(self.pairs) > 0

    def test_source_is_always_self(self):
        # Every pair in this file's %Y block should have A000001 as the source.
        for src, _ in self.pairs:
            assert src == "A000001"

    def test_self_references_dropped(self):
        # The first %Y line literally contains "A000001 (this one)" — the
        # self-reference must be filtered out even though it's in the text.
        for _, dst in self.pairs:
            assert dst != "A000001"

    def test_known_neighbors_present(self):
        # Spot-check a few well-known group-theory sequences mentioned in the
        # %Y lines of A000001's real file.
        dsts = {dst for _, dst in self.pairs}
        assert "A000679" in dsts
        assert "A001228" in dsts
        assert "A046057" in dsts  # from the 4th %Y line

    def test_no_malformed_ids(self):
        # Every captured A-number should match the canonical A\d{6} form.
        for src, dst in self.pairs:
            assert len(src) == 7 and src[0] == "A" and src[1:].isdigit()
            assert len(dst) == 7 and dst[0] == "A" and dst[1:].isdigit()


class TestExtractA000045:
    """Fibonacci has 10 %Y lines with many Cf. references."""

    @classmethod
    def setup_class(cls):
        cls.pairs = extract_y_edges(FIXTURES / "A000045.seq")

    def test_many_neighbors(self):
        # Fibonacci is well-connected; expect a healthy count across 10 %Y lines.
        assert len(self.pairs) >= 20

    def test_self_references_dropped(self):
        for _, dst in self.pairs:
            assert dst != "A000045"

    def test_known_neighbors_present(self):
        dsts = {dst for _, dst in self.pairs}
        # From the real %Y lines in the fixture:
        assert "A001175" in dsts  # Pisano periods (Cf. line)
        assert "A001177" in dsts  # entry points
        assert "A000738" in dsts  # Boustrophedon transform
        assert "A000010" in dsts  # totient comment

    def test_inline_formula_refs_captured(self):
        # The "%Y A000045 Number of digits of F(n): A020909 (base 2), ..." line
        # is inline-formula-ish rather than pure "Cf." — v1 should still catch it.
        dsts = {dst for _, dst in self.pairs}
        assert "A020909" in dsts
        assert "A060384" in dsts


class TestExtractA395003:
    """Minimal modern sequence with 2 %Y lines (one inline formula, one Cf.)."""

    @classmethod
    def setup_class(cls):
        cls.pairs = extract_y_edges(FIXTURES / "A395003.seq")

    def test_expected_neighbors(self):
        dsts = sorted({dst for _, dst in self.pairs})
        # %Y A395003 A395004 is the larger factor.
        # %Y A395003 Cf. A053644, A394989, A394990, A394991.
        assert dsts == sorted(["A395004", "A053644", "A394989", "A394990", "A394991"])

    def test_source_always_self(self):
        for src, _ in self.pairs:
            assert src == "A395003"


# No fixture has zero %Y lines, so the empty-field tests below build tiny
# synthetic .seq files in pytest's tmp_path.


def test_extract_no_y_lines(tmp_path: Path):
    """A sequence file with no %Y lines must produce an empty edge list."""
    fake = tmp_path / "A999999.seq"
    fake.write_text(
        "%I A999999 #1 Apr 11 2026 00:00:00\n"
        "%N A999999 Synthetic test sequence.\n"
        "%S A999999 1,2,3,4,5\n"
        "%K A999999 nonn\n"
    )
    assert extract_y_edges(fake) == []


def test_extract_y_line_with_no_content(tmp_path: Path):
    """A bare `%Y A999999` line (no tail) must produce no pairs."""
    fake = tmp_path / "A999999.seq"
    fake.write_text("%I A999999 #1 Apr 11 2026 00:00:00\n%N A999999 Bare Y line.\n%Y A999999\n%K A999999 nonn\n")
    assert extract_y_edges(fake) == []


def test_extract_multiple_ids_per_line(tmp_path: Path):
    """All A-numbers on a single %Y line are extracted in order."""
    fake = tmp_path / "A999998.seq"
    fake.write_text(
        "%I A999998 #1 Apr 11 2026 00:00:00\n"
        "%N A999998 Multi-ref test.\n"
        "%Y A999998 Cf. A000001, A000002, A000003.\n"
        "%K A999998 nonn\n"
    )
    pairs = extract_y_edges(fake)
    assert pairs == [
        ("A999998", "A000001"),
        ("A999998", "A000002"),
        ("A999998", "A000003"),
    ]
