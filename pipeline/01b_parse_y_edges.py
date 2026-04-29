"""Extract OEIS %Y cross-reference edges into a sidecar parquet.

Stage 01 deliberately drops `%Y` lines. This stage walks `seq/**/*.seq`, pulls
every A-number mentioned in a `%Y` line, canonicalizes each pair as an
undirected sorted tuple, and writes `data/y_edges.parquet`.

The output is **global** — all 394k sequences regardless of `OEIS_SCOPE`.
Stage 04b is responsible for filtering to the in-scope id set. This keeps
01b's cache valid across scope switches.

Incremental: like stage 01, we cache `(mtime, size)` per `path_rel` and
re-parse only changed files. Per-file edge lists are stored as an intermediate
parquet so we can rebuild the flat edge table without re-reading `.seq` files.

Edge extraction (v1, simple):

    For each `%Y A_src <content>` line, extract every `A\\d{6}` token in
    `<content>`, drop the self-reference, emit `(A_src, A_other)` for each.
    At the end, canonicalize to `(min, max)` and dedupe.

This is permissive — it catches both "Cf." lists and inline formula references
like `a(n) = A000040(n) + A000045(n)`. If retrofit signal is noisy we can
tighten to "Cf." only as a v2 refinement.
"""

from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from config import (
    DATA_DIR,
    PARSE_WORKERS,
    RAW_PARQUET,
    SEQ_DIR,
    Y_EDGES_PARQUET,
)
from tqdm import tqdm

# %Y lines look like: "%Y A000045 Cf. A000032, A000071, ..."
# Capture the source id and everything after it; we regex-find A-numbers in the tail.
Y_LINE_RE = re.compile(r"^%Y\s+(A\d+)(?:\s+(.*))?$")
A_NUMBER_RE = re.compile(r"A\d{6}")

# Cached per-file edges live here. Not strictly required by the pipeline
# contract, but lets us rebuild y_edges.parquet without re-reading .seq files
# when only a handful of sources changed. Keyed by path_rel.
_PER_FILE_CACHE = DATA_DIR / "y_edges_per_file.parquet"


# ── Per-file parser ──────────────────────────────────────────────────────────


def extract_y_edges(path: Path) -> list[tuple[str, str]]:
    """Return a list of (src_id, dst_id) pairs from one `.seq` file's `%Y` lines.

    The source id comes from the `%Y A_src` header; the destination ids come
    from every `A\\d{6}` token found in the tail content. Self-references are
    dropped. Pairs are left in natural (src, dst) order here — the caller is
    responsible for canonicalizing to sorted undirected tuples.
    """
    text = path.read_text(encoding="utf-8")
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("%Y"):
            continue
        m = Y_LINE_RE.match(line)
        if not m:
            continue
        src = m.group(1)
        content = m.group(2) or ""
        if not content:
            continue
        for dst in A_NUMBER_RE.findall(content):
            if dst != src:
                pairs.append((src, dst))
    return pairs


# ── Multiprocessing worker ───────────────────────────────────────────────────


def _parse_chunk(args: list[tuple[str, str, float, int]]) -> list[dict]:
    """Worker entry point: parse a list of (path_str, path_rel, mtime, size) tuples.

    Returns one row per input file with columns
    `(path_rel, mtime, size, pairs)` where `pairs` is a list of (src, dst)
    tuples. Empty lists are kept so the per-file cache covers every file that
    was ever scanned.
    """
    results: list[dict] = []
    for path_str, path_rel, mtime, size in args:
        path = Path(path_str)
        try:
            pairs = extract_y_edges(path)
        except Exception as e:  # pragma: no cover — defensive
            print(f"  ERROR parsing {path}: {e}", file=sys.stderr)
            pairs = []
        results.append(
            {
                "path_rel": path_rel,
                "mtime": mtime,
                "size": size,
                "pairs": pairs,
            }
        )
    return results


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    if not RAW_PARQUET.exists():
        raise SystemExit(
            f"{RAW_PARQUET} not found — run `uv run python pipeline/01_parse.py` first. "
            "Stage 01b uses its file list to skip discovery."
        )

    # 1. Get the file list from stage 01's output and stat each file fresh.
    print(f"Reading file list from {RAW_PARQUET.name}…")
    raw_df = pd.read_parquet(RAW_PARQUET, columns=["path_rel"])
    print(f"  {len(raw_df):,} files listed")

    print(f"Stat'ing {SEQ_DIR}…")
    t0 = time.time()
    discovered: list[tuple[str, str, float, int]] = []
    missing = 0
    for rel in raw_df["path_rel"]:
        path = SEQ_DIR.parent / rel
        try:
            st = path.stat()
        except FileNotFoundError:
            missing += 1
            continue
        discovered.append((str(path), rel, st.st_mtime, st.st_size))
    print(f"  Stat'd {len(discovered):,} files in {time.time() - t0:.1f}s ({missing} missing on disk)")

    if not discovered:
        print("Nothing to parse.")
        return

    # 2. Diff against per-file cache to decide which files need (re-)parsing.
    needs_reparse: list[tuple[str, str, float, int]] = []
    cached_df: pd.DataFrame | None = None
    if _PER_FILE_CACHE.exists():
        print(f"Loading existing {_PER_FILE_CACHE.name} for incremental rebuild…")
        cached_df = pd.read_parquet(_PER_FILE_CACHE)
        cache: dict[str, tuple[float, int]] = {
            row.path_rel: (row.mtime, row.size) for row in cached_df.itertuples(index=False)
        }
        live_paths = {rel for _, rel, _, _ in discovered}
        for path_str, rel, mtime, size in discovered:
            cur = cache.get(rel)
            if cur is None or cur[0] != mtime or cur[1] != size:
                needs_reparse.append((path_str, rel, mtime, size))
        # Drop cache rows for files that are no longer on disk
        still_present_mask = cached_df["path_rel"].isin(live_paths)
        cached_df = cached_df[still_present_mask].reset_index(drop=True)
        print(f"  {len(needs_reparse):,} files need (re-)parsing; {len(cached_df):,} unchanged")
    else:
        needs_reparse = list(discovered)
        print(f"  No cache; parsing all {len(needs_reparse):,} files")

    # 3. Parse changed files in parallel.
    if needs_reparse:
        print(f"Extracting %Y edges from {len(needs_reparse):,} files using {PARSE_WORKERS} workers…")
        t0 = time.time()
        chunk_size = max(100, (len(needs_reparse) + PARSE_WORKERS * 4 - 1) // (PARSE_WORKERS * 4))
        chunks = [needs_reparse[i : i + chunk_size] for i in range(0, len(needs_reparse), chunk_size)]
        new_rows: list[dict] = []
        with ProcessPoolExecutor(max_workers=PARSE_WORKERS) as executor:
            for chunk_results in tqdm(
                executor.map(_parse_chunk, chunks),
                total=len(chunks),
                desc="parsing %Y",
                unit="chunk",
            ):
                new_rows.extend(chunk_results)
        print(f"  Parsed {len(new_rows):,} files in {time.time() - t0:.1f}s")
        new_df = pd.DataFrame(new_rows)
    else:
        new_df = pd.DataFrame(columns=["path_rel", "mtime", "size", "pairs"])
        print("No files need reparsing.")

    # 4. Merge per-file cache: drop reparsed rows, concat fresh rows.
    if cached_df is not None and len(cached_df) > 0 and len(new_df) > 0:
        cached_df = cached_df[~cached_df["path_rel"].isin(new_df["path_rel"])].reset_index(drop=True)
        merged_df = pd.concat([cached_df, new_df], ignore_index=True)
    elif cached_df is not None and len(cached_df) > 0:
        merged_df = cached_df
    else:
        merged_df = new_df

    # 5. Write per-file cache atomically.
    print(f"Writing {_PER_FILE_CACHE.name} ({len(merged_df):,} rows)…")
    tmp_cache = _PER_FILE_CACHE.with_suffix(".parquet.tmp")
    merged_df.to_parquet(tmp_cache, index=False, compression="zstd")
    tmp_cache.replace(_PER_FILE_CACHE)

    # 6. Flatten per-file pairs into a global edge list, canonicalize, dedupe.
    print("Flattening edges and canonicalizing…")
    t0 = time.time()
    # Explode `pairs` (list of tuples) into one edge per row.
    exploded = merged_df[["pairs"]].explode("pairs", ignore_index=True)
    exploded = exploded[exploded["pairs"].notna()].reset_index(drop=True)
    if len(exploded) == 0:
        print("  No %Y edges found in any file. Writing empty edge table.")
        edges_df = pd.DataFrame({"from_id": pd.Series(dtype="string"), "to_id": pd.Series(dtype="string")})
    else:
        # Each row is a (src, dst) tuple; split into two columns.
        src_ids = exploded["pairs"].map(lambda t: t[0])
        dst_ids = exploded["pairs"].map(lambda t: t[1])
        # Canonicalize undirected: from = min, to = max
        a = src_ids.where(src_ids <= dst_ids, dst_ids)
        b = dst_ids.where(src_ids <= dst_ids, src_ids)
        edges_df = pd.DataFrame({"from_id": a.astype("string"), "to_id": b.astype("string")})
        # Dedupe and sort for stable output.
        edges_df = edges_df.drop_duplicates().sort_values(["from_id", "to_id"]).reset_index(drop=True)
    print(f"  {len(edges_df):,} unique undirected edges in {time.time() - t0:.1f}s")

    # 7. Write the flat edge table atomically.
    print(f"Writing {Y_EDGES_PARQUET.name}…")
    tmp_edges = Y_EDGES_PARQUET.with_suffix(".parquet.tmp")
    edges_df.to_parquet(tmp_edges, index=False, compression="zstd")
    tmp_edges.replace(Y_EDGES_PARQUET)

    # 8. Sanity report.
    size_mb = Y_EDGES_PARQUET.stat().st_size / 1_000_000
    print(f"\nDone. {len(edges_df):,} edges in {Y_EDGES_PARQUET} ({size_mb:.1f} MB)")
    if len(edges_df) > 0:
        # Distinct sequence count touched by edges.
        touched = pd.unique(pd.concat([edges_df["from_id"], edges_df["to_id"]], ignore_index=True))
        print(f"  Touched {len(touched):,} distinct sequences")
        # Per-source fan-out spot-check.
        fanout = edges_df.groupby("from_id").size()
        print(f"  Fan-out: median={fanout.median():.0f}, p95={fanout.quantile(0.95):.0f}, max={fanout.max()}")


if __name__ == "__main__":
    main()
