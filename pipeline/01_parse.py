"""Parse all OEIS .seq files into a single parquet table.

Walks `seq/**/*.seq`, parses each file's `%X AXXXXXX content` lines into
structured fields, and writes the result to `data/raw_sequences.parquet`.

Incremental: a re-run only re-parses files whose (mtime, size) differ from
the cached values in the existing parquet. The full corpus (~394k files)
takes a few minutes cold and seconds warm.
"""

from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from config import (
    DATA_DIR,
    MAX_VALUES_SHOWN,
    PARSE_WORKERS,
    RAW_PARQUET,
    SEQ_DIR,
)
from tqdm import tqdm

# ── Regexes ──────────────────────────────────────────────────────────────────

# Single field line: "%X AXXXXXX content"  (content may be empty)
LINE_RE = re.compile(r"^%(\w)\s+A\d+(?:\s(.*))?$")

# Header line — handles both forms:
#   old: %I A000001 M0098 N0035 #326 Jan 28 2026 13:29:57
#   new: %I A300000 #22 Jul 08 2022 12:19:01
HEADER_RE = re.compile(
    r"^%I\s+(A\d+)"  # sequence id
    r"(?:\s+M\d+\s+N\d+)?"  # optional legacy book ids
    r"\s+#(\d+)"  # edit count
    r"\s+(.+?)\s*$"  # date string
)

# Allowed characters inside an OEIS %o language tag — real tags are short
# identifiers like "PARI", "Python 3", "PARI/GP", "C++", "C#", "MATLAB". This
# whitelist keeps the parser from misinterpreting parenthesized code fragments
# (e.g. Haskell `(zipWith (-`, Scheme `(define (swap! s`) as language tags.
_LANG_TAG_ALLOWED_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/+#-. ")


def _try_extract_lang(content: str) -> str | None:
    """Return the language tag at the start of a %o line, or None if absent.

    Real OEIS language tags are short (≤30 char), wrapped in parens at the
    start of the content, begin with a letter, and consist only of letters,
    digits, and `/+#-. `. Anything else is almost certainly a parenthesized
    code expression continuing a multi-line code block.
    """
    if not content.startswith("("):
        return None
    end = content.find(")")
    if end == -1 or end > 30:
        return None
    tag = content[1:end]
    if not tag or not tag[0].isalpha():
        return None
    if not all(c in _LANG_TAG_ALLOWED_CHARS for c in tag):
        return None
    return tag


# ── Constants ────────────────────────────────────────────────────────────────

INT64_MAX = int(np.iinfo(np.int64).max)
INT64_MIN = int(np.iinfo(np.int64).min)

MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

# Maps the inside of an %o language tag to a canonical bucket. Anything not
# listed falls through to "other". Lowercased and whitespace-collapsed before
# lookup.
LANGUAGE_NORMALIZATION = {
    "python": "python",
    "python2": "python",
    "python3": "python",
    "pari": "pari",
    "pari/gp": "pari",
    "parigp": "pari",
    "gp": "pari",
    "sage": "sage",
    "sagemath": "sage",
    "magma": "magma",
    "gap": "gap",
    "maple": "maple",
    "mathematica": "mathematica",
    "wolfram": "mathematica",
    "haskell": "haskell",
    "julia": "julia",
    "scala": "scala",
    "perl": "perl",
    "ruby": "ruby",
    "axiom": "axiom",
    "smalltalk": "smalltalk",
    "java": "java",
    "javascript": "javascript",
    "js": "javascript",
    "fortran": "fortran",
    "c": "c",
    "c++": "c",
    "cpp": "c",
    "c#": "csharp",
    "csharp": "csharp",
    "r": "r",
    "lisp": "lisp",
    "common lisp": "lisp",
    "scheme": "lisp",
    "racket": "lisp",
    "maxima": "maxima",
    "macsyma": "maxima",
    "matlab": "matlab",
    "octave": "matlab",
    "aribas": "aribas",
    "mupad": "mupad",
    "ubasic": "ubasic",
    "pfgw": "pfgw",
    "prime95": "pfgw",
    "apl": "apl",
    "j": "apl",
    "k": "apl",
    "nauty": "nauty",
    "sh": "shell",
    "bash": "shell",
    "zsh": "shell",
}

# OEIS field codes for sequence values: S/T/U are unsigned, V/W/X are signed.
VALUE_FIELDS = {"S", "T", "U", "V", "W", "X"}

# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_date(s: str) -> datetime | None:
    """Parse 'Jan 28 2026 13:29:57' robustly without locale dependency."""
    parts = s.strip().split()
    if len(parts) != 4:
        return None
    month_str, day_str, year_str, time_str = parts
    if month_str not in MONTH_MAP:
        return None
    try:
        h, m, sec = (int(x) for x in time_str.split(":"))
        return datetime(int(year_str), MONTH_MAP[month_str], int(day_str), h, m, sec)
    except (ValueError, KeyError):
        return None


def _normalize_lang(tag: str) -> str:
    """Normalize an %o code language tag to a canonical bucket name."""
    cleaned = re.sub(r"\s+", " ", tag.strip().lower())
    # Strip trailing version numbers like "python 3" -> "python"
    cleaned = re.sub(r"\s+\d+(\.\d+)?$", "", cleaned)
    return LANGUAGE_NORMALIZATION.get(cleaned, "other")


# ── Per-file parser ──────────────────────────────────────────────────────────


def parse_seq_file(path: Path) -> dict:
    """Parse a single .seq file into a dict of structured fields.

    The returned dict has the columns expected in raw_sequences.parquet
    (except mtime/size/path_rel, which are added by the worker that calls
    this function so it can use a single os.stat per file).
    """
    text = path.read_text(encoding="utf-8")

    seq_id: str | None = None
    edit_count = 0
    last_edited: datetime | None = None
    name = ""
    comments_lines: list[str] = []
    formulas_lines: list[str] = []
    examples_lines: list[str] = []
    value_chunks: list[str] = []
    keywords: list[str] = []
    offset = ""
    author = ""
    n_references = 0
    n_links = 0
    n_extensions = 0
    has_bfile = False
    code_languages: set[str] = set()

    for line in text.splitlines():
        if not line.startswith("%"):
            continue

        m = LINE_RE.match(line)
        if not m:
            continue
        field = m.group(1)
        content = m.group(2) or ""

        if field == "I":
            hm = HEADER_RE.match(line)
            if hm:
                seq_id = hm.group(1)
                try:
                    edit_count = int(hm.group(2))
                except ValueError:
                    edit_count = 0
                last_edited = _parse_date(hm.group(3))
        elif field == "N":
            if not name:
                name = content.strip()
        elif field in VALUE_FIELDS:
            value_chunks.append(content.strip())
        elif field == "C":
            comments_lines.append(content)
        elif field == "F":
            formulas_lines.append(content)
        elif field == "e":
            examples_lines.append(content)
        elif field == "K":
            if not keywords:
                keywords = [k.strip() for k in content.split(",") if k.strip()]
        elif field == "O":
            if not offset:
                offset = content.strip()
        elif field == "A":
            if not author:
                author = content.strip()
        elif field == "D":
            n_references += 1
        elif field == "H":
            n_links += 1
            # Detect a /AXXXXXX/bXXXXXX.txt link (the b-file with full sequence values)
            if seq_id and f'href="/{seq_id}/b{seq_id[1:]}.txt"' in content:
                has_bfile = True
        elif field == "p":
            code_languages.add("maple")
        elif field == "t":
            code_languages.add("mathematica")
        elif field == "o":
            tag = _try_extract_lang(content.strip())
            if tag is not None:
                code_languages.add(_normalize_lang(tag))
        elif field == "E":
            n_extensions += 1
        # %Y (cross-references) explicitly ignored for v1

    # Concatenate all sequence value chunks ("0,1,1,2,3," + "5,8,...") and split
    all_values_str = ",".join(chunk.rstrip(",") for chunk in value_chunks if chunk)
    value_strs = [v.strip() for v in all_values_str.split(",") if v.strip()]
    n_terms_visible = len(value_strs)

    # Cast leading values to int64 only where they fit; stop at first overflow
    values: list[int] = []
    for s in value_strs[:MAX_VALUES_SHOWN]:
        try:
            v = int(s)
        except ValueError:
            break
        if INT64_MIN <= v <= INT64_MAX:
            values.append(v)
        else:
            break

    # Build display preview from the original strings (not the cast ints) so
    # huge Fibonacci-style values still render in tooltips
    if value_strs:
        preview_parts = value_strs[:MAX_VALUES_SHOWN]
        preview = ", ".join(preview_parts)
        if len(value_strs) > MAX_VALUES_SHOWN:
            preview += ", …"
    else:
        preview = ""

    return {
        "id": seq_id or path.stem.upper(),
        "name": name,
        "comments": "\n".join(comments_lines),
        "formulas": "\n".join(formulas_lines),
        "examples": "\n".join(examples_lines),
        "keywords": keywords,
        "offset": offset,
        "values": values,
        "values_preview_str": preview,
        "n_terms_visible": n_terms_visible,
        "author": author,
        "edit_count": edit_count,
        "last_edited": last_edited,
        "n_references": n_references,
        "n_links": n_links,
        "n_extensions": n_extensions,
        "code_languages": sorted(code_languages),
        "has_bfile": has_bfile,
    }


# ── Multiprocessing worker ───────────────────────────────────────────────────


def _parse_chunk(args: list[tuple[str, float, int]]) -> list[dict]:
    """Worker entry point: parse a list of (path_str, mtime, size) tuples.

    Receives string paths (not Path objects) so the chunk pickles smaller. The
    mtime and size are pre-computed in the main process during file discovery
    so workers don't need a second os.stat call.
    """
    results: list[dict] = []
    for path_str, mtime, size in args:
        path = Path(path_str)
        try:
            row = parse_seq_file(path)
        except Exception as e:  # pragma: no cover — defensive
            print(f"  ERROR parsing {path}: {e}", file=sys.stderr)
            continue
        row["mtime"] = mtime
        row["size"] = size
        row["path_rel"] = str(path.relative_to(SEQ_DIR.parent))
        results.append(row)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    # 1. Discover every .seq file with its (mtime, size).
    print(f"Discovering files in {SEQ_DIR}…")
    t0 = time.time()
    discovered: dict[str, tuple[str, float, int]] = {}
    for path in SEQ_DIR.rglob("*.seq"):
        st = path.stat()
        rel = str(path.relative_to(SEQ_DIR.parent))
        discovered[rel] = (str(path), st.st_mtime, st.st_size)
    print(f"  Found {len(discovered):,} files in {time.time() - t0:.1f}s")

    if not discovered:
        print(f"No .seq files found under {SEQ_DIR}. Aborting.")
        return

    # 2. Decide which files need (re-)parsing by diffing against the existing
    # parquet's mtime+size cache.
    existing_df: pd.DataFrame | None = None
    needs_reparse: list[tuple[str, float, int]] = []
    if RAW_PARQUET.exists():
        print(f"Loading existing {RAW_PARQUET.name} for incremental rebuild…")
        existing_df = pd.read_parquet(RAW_PARQUET)
        cache: dict[str, tuple[float, int]] = {
            row.path_rel: (row.mtime, row.size) for row in existing_df.itertuples(index=False)
        }
        for rel, (path_str, mtime, size) in discovered.items():
            cached = cache.get(rel)
            if cached is None or cached[0] != mtime or cached[1] != size:
                needs_reparse.append((path_str, mtime, size))
        # Drop rows for files that no longer exist on disk
        still_present_mask = existing_df["path_rel"].isin(discovered.keys())
        existing_df = existing_df[still_present_mask].reset_index(drop=True)
        print(f"  {len(needs_reparse):,} files need (re-)parsing; {len(existing_df):,} unchanged")
    else:
        needs_reparse = list(discovered.values())
        print(f"  No existing parquet; parsing all {len(needs_reparse):,} files")

    if not needs_reparse:
        print("Nothing to do.")
        return

    # 3. Parse the changed files in parallel.
    print(f"Parsing {len(needs_reparse):,} files using {PARSE_WORKERS} workers…")
    t0 = time.time()
    chunk_size = max(100, (len(needs_reparse) + PARSE_WORKERS * 4 - 1) // (PARSE_WORKERS * 4))
    chunks = [needs_reparse[i : i + chunk_size] for i in range(0, len(needs_reparse), chunk_size)]

    new_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=PARSE_WORKERS) as executor:
        for chunk_results in tqdm(
            executor.map(_parse_chunk, chunks),
            total=len(chunks),
            desc="parsing",
            unit="chunk",
        ):
            new_rows.extend(chunk_results)
    print(f"  Parsed {len(new_rows):,} files in {time.time() - t0:.1f}s")

    new_df = pd.DataFrame(new_rows)

    # 4. Merge: drop reparsed rows from the existing frame, then concat.
    if existing_df is not None and len(existing_df) > 0:
        existing_df = existing_df[~existing_df["path_rel"].isin(new_df["path_rel"])].reset_index(drop=True)
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        merged_df = new_df

    merged_df = merged_df.sort_values("id").reset_index(drop=True)

    # 5. Write atomically: temp file then rename.
    print(f"Writing {len(merged_df):,} rows to {RAW_PARQUET.name}…")
    tmp_path = RAW_PARQUET.with_suffix(".parquet.tmp")
    merged_df.to_parquet(tmp_path, index=False, compression="zstd")
    tmp_path.replace(RAW_PARQUET)

    # 6. Sanity report
    size_mb = RAW_PARQUET.stat().st_size / 1_000_000
    print(f"\nDone. {len(merged_df):,} sequences in {RAW_PARQUET} ({size_mb:.1f} MB)")
    if len(merged_df) > 0:
        first = merged_df.iloc[0]
        print(f"  Sample row: {first['id']} — {first['name'][:70]}")
        print(f"  edit_count: median={merged_df['edit_count'].median():.0f}, max={merged_df['edit_count'].max()}")
        with_code = (merged_df["code_languages"].apply(len) > 0).sum()
        print(f"  Sequences with code: {with_code:,}")
        with_keywords = (merged_df["keywords"].apply(len) > 0).sum()
        print(f"  Sequences with keywords: {with_keywords:,}")


if __name__ == "__main__":
    main()
