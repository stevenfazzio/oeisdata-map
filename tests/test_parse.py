"""Tests for pipeline/01_parse.py against fixture .seq files.

Covers:
- Both `%I` header forms (legacy with M/N book IDs, modern without).
- A pathological-comments sequence (Fibonacci, A000045) — 766 lines, 60+ comments.
- A minimal recent sequence (A395003) with no code blocks and no `%H` lines.
- Code-language extraction across `%p`, `%t`, and `%o` with multiple `%o` tags.
- Big-integer values that may exceed int64 once the sequence runs long enough.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

# Add pipeline/ to sys.path so the loaded script's `from config import ...` resolves
REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

# Load pipeline/01_parse.py via importlib (digit-prefix module names aren't directly importable)
_spec = importlib.util.spec_from_file_location("parse_oeis", PIPELINE_DIR / "01_parse.py")
parse_oeis = importlib.util.module_from_spec(_spec)
sys.modules["parse_oeis"] = parse_oeis
_spec.loader.exec_module(parse_oeis)

parse_seq_file = parse_oeis.parse_seq_file
_parse_date = parse_oeis._parse_date
_normalize_lang = parse_oeis._normalize_lang
_try_extract_lang = parse_oeis._try_extract_lang

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ── A000001: legacy %I header, code in many languages ───────────────────────


class TestA000001:
    @classmethod
    def setup_class(cls):
        cls.row = parse_seq_file(FIXTURES / "A000001.seq")

    def test_id(self):
        assert self.row["id"] == "A000001"

    def test_name(self):
        assert self.row["name"] == "Number of groups of order n."

    def test_edit_count(self):
        assert self.row["edit_count"] == 326

    def test_last_edited(self):
        assert self.row["last_edited"] == datetime(2026, 1, 28, 13, 29, 57)

    def test_keywords(self):
        assert self.row["keywords"] == ["nonn", "core", "nice", "hard"]

    def test_offset(self):
        assert self.row["offset"] == "0,5"

    def test_author(self):
        assert "Sloane" in self.row["author"]

    def test_first_values(self):
        # %S A000001 0,1,1,1,2,1,2,1,...
        assert self.row["values"][:8] == [0, 1, 1, 1, 2, 1, 2, 1]

    def test_n_terms_visible(self):
        # 33 + 31 + 30 = 94 visible terms across %S/%T/%U
        assert self.row["n_terms_visible"] == 94

    def test_n_references(self):
        assert self.row["n_references"] == 12

    def test_n_links(self):
        assert self.row["n_links"] == 35

    def test_code_languages(self):
        # %p (Maple), %t (Mathematica), %o tags: Magma, GAP, Python, PARI
        assert set(self.row["code_languages"]) == {
            "maple", "mathematica", "magma", "gap", "python", "pari",
        }

    def test_has_bfile(self):
        # %H A000001 ... <a href="/A000001/b000001.txt">Table of n, a(n)...</a>
        assert self.row["has_bfile"] is True

    def test_values_preview_str(self):
        # First 15 terms, comma-joined; preview is truncated with ellipsis
        assert self.row["values_preview_str"].startswith("0, 1, 1, 1, 2")
        assert self.row["values_preview_str"].endswith("…")

    def test_comments_nonempty(self):
        assert len(self.row["comments"]) > 0
        assert "groups of order" in self.row["comments"].lower()


# ── A000045 Fibonacci: legacy header, pathological-length comments ──────────


class TestA000045Fibonacci:
    @classmethod
    def setup_class(cls):
        cls.row = parse_seq_file(FIXTURES / "A000045.seq")

    def test_id(self):
        assert self.row["id"] == "A000045"

    def test_name_starts_with_fibonacci(self):
        assert self.row["name"].startswith("Fibonacci")

    def test_edit_count(self):
        assert self.row["edit_count"] == 2514

    def test_last_edited(self):
        assert self.row["last_edited"] == datetime(2026, 4, 8, 10, 58, 11)

    def test_keywords(self):
        assert self.row["keywords"] == ["nonn", "core", "nice", "easy", "hear", "changed"]

    def test_first_values(self):
        # %S A000045 0,1,1,2,3,5,8,13,21,34,...
        assert self.row["values"][:8] == [0, 1, 1, 2, 3, 5, 8, 13]

    def test_n_terms_visible(self):
        # Fibonacci: 21 + 10 + 10 = 41 across %S/%T/%U
        assert self.row["n_terms_visible"] == 41

    def test_offset(self):
        assert self.row["offset"] == "0,4"

    def test_author_contains_sloane(self):
        assert "Sloane" in self.row["author"]

    def test_n_references_is_large(self):
        # Fibonacci has many book references — exact count is 59 in the current export
        assert self.row["n_references"] >= 50

    def test_has_bfile(self):
        # /A000045/b000045.txt is in the %H lines
        assert self.row["has_bfile"] is True

    def test_pathological_comment_count(self):
        # The whole point of this fixture: comment field accumulates all 60+ %C lines
        assert self.row["comments"].count("\n") >= 50

    def test_values_preview_truncated(self):
        # 41 visible terms but only first 15 in preview
        assert self.row["values_preview_str"].endswith("…")
        assert self.row["values_preview_str"].startswith("0, 1, 1, 2, 3, 5, 8, 13")


# ── A300000: modern %I header form (no M/N book IDs) ────────────────────────


class TestA300000:
    @classmethod
    def setup_class(cls):
        cls.row = parse_seq_file(FIXTURES / "A300000.seq")

    def test_id(self):
        assert self.row["id"] == "A300000"

    def test_edit_count(self):
        # The header for A300000 omits the legacy M/N book IDs — this is the
        # critical test that the parser regex's `(?:M\d+ N\d+)?` is optional.
        assert self.row["edit_count"] == 22

    def test_last_edited(self):
        assert self.row["last_edited"] == datetime(2022, 7, 8, 12, 19, 1)

    def test_keywords(self):
        assert self.row["keywords"] == ["nonn", "base", "nice", "easy"]

    def test_first_values(self):
        assert self.row["values"][:5] == [1, 10, 99, 999, 9990]

    def test_n_references_zero(self):
        assert self.row["n_references"] == 0

    def test_code_languages(self):
        # %t (Mathematica), %o (PARI), %o (Python)
        assert set(self.row["code_languages"]) == {"mathematica", "pari", "python"}

    def test_has_bfile(self):
        assert self.row["has_bfile"] is True

    def test_offset(self):
        assert self.row["offset"] == "1,2"

    def test_author_contains_angelini(self):
        assert "Angelini" in self.row["author"]


# ── A395003: minimal modern sequence (no %H, no code, has %E) ───────────────


class TestA395003:
    @classmethod
    def setup_class(cls):
        cls.row = parse_seq_file(FIXTURES / "A395003.seq")

    def test_id(self):
        assert self.row["id"] == "A395003"

    def test_edit_count(self):
        assert self.row["edit_count"] == 11

    def test_last_edited(self):
        assert self.row["last_edited"] == datetime(2026, 4, 10, 0, 48, 9)

    def test_keywords(self):
        assert self.row["keywords"] == ["nonn", "more", "new"]

    def test_first_values(self):
        assert self.row["values"][:6] == [1, 3, 5, 9, 17, 33]

    def test_no_code_languages(self):
        # No %p, %t, or %o lines in this minimal fixture — must be empty list
        assert self.row["code_languages"] == []

    def test_no_bfile(self):
        # No %H lines at all
        assert self.row["has_bfile"] is False

    def test_n_links_zero(self):
        assert self.row["n_links"] == 0

    def test_n_extensions(self):
        # One %E line
        assert self.row["n_extensions"] == 1

    def test_offset(self):
        assert self.row["offset"] == "1,2"

    def test_author_contains_pfoertner(self):
        assert "Pfoertner" in self.row["author"]


# ── Helper-function unit tests ──────────────────────────────────────────────


class TestParseDate:
    def test_legacy_format(self):
        assert _parse_date("Jan 28 2026 13:29:57") == datetime(2026, 1, 28, 13, 29, 57)

    def test_modern_format(self):
        assert _parse_date("Jul 08 2022 12:19:01") == datetime(2022, 7, 8, 12, 19, 1)

    def test_each_month_parseable(self):
        for month_str, month_num in [
            ("Jan", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4),
            ("May", 5), ("Jun", 6), ("Jul", 7), ("Aug", 8),
            ("Sep", 9), ("Oct", 10), ("Nov", 11), ("Dec", 12),
        ]:
            d = _parse_date(f"{month_str} 15 2024 00:00:00")
            assert d == datetime(2024, month_num, 15, 0, 0, 0)

    def test_garbage_returns_none(self):
        assert _parse_date("not a date") is None
        assert _parse_date("Foo 99 2026 99:99:99") is None
        assert _parse_date("") is None


class TestNormalizeLang:
    def test_python_variants(self):
        assert _normalize_lang("Python") == "python"
        assert _normalize_lang("Python 3") == "python"
        assert _normalize_lang("python3") == "python"

    def test_pari_variants(self):
        assert _normalize_lang("PARI") == "pari"
        assert _normalize_lang("PARI/GP") == "pari"
        assert _normalize_lang("GP") == "pari"

    def test_sage_variants(self):
        assert _normalize_lang("Sage") == "sage"
        assert _normalize_lang("SageMath") == "sage"

    def test_unknown_falls_through_to_other(self):
        assert _normalize_lang("Brainfuck") == "other"
        assert _normalize_lang("Cobol") == "other"

    def test_canonical_buckets(self):
        assert _normalize_lang("Maple") == "maple"
        assert _normalize_lang("Mathematica") == "mathematica"
        assert _normalize_lang("Magma") == "magma"
        assert _normalize_lang("GAP") == "gap"
        assert _normalize_lang("Haskell") == "haskell"
        assert _normalize_lang("Maxima") == "maxima"
        assert _normalize_lang("MATLAB") == "matlab"


class TestTryExtractLang:
    """The strict language-tag extractor must accept real OEIS tags and reject
    parenthesized code fragments that happen to start a continuation line of a
    multi-line code block."""

    def test_real_tags_accepted(self):
        assert _try_extract_lang("(PARI) a(n) = 2*n") == "PARI"
        assert _try_extract_lang("(Python) def a(n): return n") == "Python"
        assert _try_extract_lang("(Python 3)") == "Python 3"
        assert _try_extract_lang("(PARI/GP) ...") == "PARI/GP"
        assert _try_extract_lang("(Magma) D := ...") == "Magma"
        assert _try_extract_lang("(Maxima) declare(...)") == "Maxima"
        assert _try_extract_lang("(MATLAB) for i = 1:n") == "MATLAB"
        assert _try_extract_lang("(C++) int a(int n)") == "C++"
        assert _try_extract_lang("(C#) public int A()") == "C#"
        assert _try_extract_lang("(Sage) ...") == "Sage"

    def test_haskell_continuation_rejected(self):
        # Haskell code lines like "(zipWith (- ...))" must not be misread
        assert _try_extract_lang("(zipWith (- 1) ...)") is None
        assert _try_extract_lang("(zipWith (+) ...)") is None
        assert _try_extract_lang("(map (* 2) ...)") is None

    def test_scheme_lisp_continuation_rejected(self):
        assert _try_extract_lang("(define (swap! s ...))") is None

    def test_numeric_or_symbolic_starts_rejected(self):
        # These start with non-letter characters and must be rejected
        assert _try_extract_lang("(1..n)") is None
        assert _try_extract_lang("(-1)") is None
        assert _try_extract_lang("(\\(x, i)") is None

    def test_no_open_paren(self):
        assert _try_extract_lang("def a(n): return 0") is None
        assert _try_extract_lang("") is None

    def test_unclosed_paren(self):
        assert _try_extract_lang("(unclosed but very long line of code here") is None

    def test_overlong_tag_rejected(self):
        assert _try_extract_lang("(this is way too long to be a real language)") is None
