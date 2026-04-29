"""Shared LLM classification module for OEIS sequences (stage 03 + eval).

Both ``pipeline/03_enrich.py`` and ``eval/run_models.py`` consume
:func:`enrich_dataframe` from this module so the eval validates exactly the
same code path that production runs.

The classification is a single Anthropic tool-use call that returns four
enum fields per sequence:

- ``math_domain``    — 12 values
- ``sequence_type``  — 9 values
- ``growth_class``   — 9 values
- ``origin_era``     — 5 values

Sequences are batched (default 25 per call), batches are run concurrently
under an :class:`asyncio.Semaphore` (default 30), and progress is atomically
checkpointed every ~200 rows so an interrupted run can resume.

This is library code with no side effects at import time. Both scripts that
consume it are expected to load ``pipeline.config`` (which handles dotenv)
before instantiating an :class:`anthropic.AsyncAnthropic` client.

References:
- Tool-use schema + index-based row mapping:
  ``~/repos/claude-code-changelog-analysis/scripts/enrich.py:108-220``
- Async semaphore + retry-with-backoff + atomic checkpoint:
  ``~/repos/semantic-github-map/pipeline/03_summarize_readmes.py:78-243``
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
from tqdm import tqdm

# ── Taxonomy enums ───────────────────────────────────────────────────────────
# Locked at the master plan stage; see ``~/.claude/plans/lexical-enchanting-sunrise.md``.

MATH_DOMAIN = [
    "number_theory",
    "combinatorics",
    "algebra",
    "analysis",
    "geometry",
    "graph_theory",
    "discrete_dynamics",
    "recreational",
    "physics_chemistry",
    "computer_science",
    "probability_stochastic",
    "other",
]

SEQUENCE_TYPE = [
    "enumeration",
    "arithmetic_function",
    "recurrence",
    "closed_form",
    "constant_digits",
    "table_flattened",
    "characteristic",
    "ranked_list",
    "other",
]

GROWTH_CLASS = [
    "finite",
    "bounded",
    "linear",
    "polynomial",
    "exponential",
    "factorial_or_faster",
    "logarithmic_or_subpoly",
    "oscillating",
    "unknown",
]

ORIGIN_ERA = [
    "classical_pre1900",
    "early_20c_1900_1950",
    "mid_20c_1950_2000",
    "modern_post2000",
    "unknown",
]

ENUM_COLS: tuple[str, ...] = ("math_domain", "sequence_type", "growth_class", "origin_era")

# Per-field allowed-value lists, used to validate API responses defensively.
# Anthropic's tool-use enum validation is not always strict — during the 200-row
# Sonnet eval run, one row came back as math_domain="chemistry_physics" (a typo
# of the actual enum "physics_chemistry"). Maps to fallback if seen.
ENUM_VALUES: dict[str, list[str]] = {
    "math_domain": MATH_DOMAIN,
    "sequence_type": SEQUENCE_TYPE,
    "growth_class": GROWTH_CLASS,
    "origin_era": ORIGIN_ERA,
}

ENUM_FALLBACK: dict[str, str] = {
    "math_domain": "other",
    "sequence_type": "other",
    "growth_class": "unknown",
    "origin_era": "unknown",
}

# ── Tool-use schema ──────────────────────────────────────────────────────────

CLASSIFY_TOOL: dict[str, Any] = {
    "name": "classify_sequences",
    "description": "Submit classifications for a batch of OEIS integer sequences.",
    "input_schema": {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "0-based position of the sequence in the input batch",
                        },
                        "math_domain": {"type": "string", "enum": MATH_DOMAIN},
                        "sequence_type": {"type": "string", "enum": SEQUENCE_TYPE},
                        "growth_class": {"type": "string", "enum": GROWTH_CLASS},
                        "origin_era": {"type": "string", "enum": ORIGIN_ERA},
                    },
                    "required": ["index", "math_domain", "sequence_type", "growth_class", "origin_era"],
                },
            },
        },
        "required": ["classifications"],
    },
}

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
<task>
You classify Online Encyclopedia of Integer Sequences (OEIS) entries into a fixed
4-field taxonomy. For each sequence in the input, return one classification by
calling the classify_sequences tool. Use the values preview, name, formula,
comments, keywords, author, and last-edited year to inform your decision. The
Author field often carries explicit historical attribution (e.g., a name + year);
the LastEdited field is a weak hint about OEIS-entry recency, NOT original
authorship of the mathematical concept. Always pick a single best-fitting enum
value.
</task>

<math-domain>
Pick the SINGLE most-fitting mathematical area for the sequence:
- "number_theory" — primes, divisors, totient, modular arithmetic, Diophantine equations,
  polygonal/figurate numbers (triangular, square, pentagonal, k-gonal, lattice spirals)
- "combinatorics" — counting, partitions, permutations, lattice paths, set enumeration
- "algebra" — group/ring/field structure, polynomial sequences, algebraic invariants
- "analysis" — Taylor coefficients, integral transforms, special functions, decimal expansions
  of constants, modular forms / theta-eta quotients / q-series / Ramanujan-style power-series
  coefficients (these are coefficients of meromorphic functions, not "combinatorics")
- "geometry" — distances, areas, volumes, lattice points, polytopes, packings, actual
  geometric objects in space (NOT polygonal numbers — those are number_theory)
- "graph_theory" — graphs, trees, networks, colorings, matchings, automorphisms
- "discrete_dynamics" — iterated maps, cellular automata (including row/column/diagonal
  binary or decimal representations of CA growth), Collatz-like, recurrences over finite state
- "recreational" — puzzles, magic squares, palindromes, base-dependent curiosities, word play
- "physics_chemistry" — physical constants, chemical isomer counts, lattice/spin models
- "computer_science" — algorithm complexity, codes, automata theory, programming-language objects
  (NOT cellular automata — those are discrete_dynamics)
- "probability_stochastic" — random walks, branching processes, expected values, occupancy
- "other" — doesn't fit any of the above
</math-domain>

<sequence-type>
What does each term a(n) of the sequence represent?
- "enumeration" — counts of combinatorial structures parameterized by n (e.g., "number of partitions of n")
- "arithmetic_function" — value of a number-theoretic function at n (e.g., d(n), φ(n), σ(n))
- "recurrence" — defined by a recursive formula in earlier terms (e.g., a(n) = a(n-1) + a(n-2))
- "closed_form" — defined by an explicit closed-form expression in n
- "constant_digits" — the n-th digit (or term) of a real-valued constant's expansion
- "table_flattened" — a 2D triangle/table read by antidiagonals or rows (Pascal's triangle, Stirling numbers)
- "characteristic" — 1 if n has property P, else 0 (characteristic function of a set of integers)
- "ranked_list" — the n-th element of an enumeration of integers with some property (e.g., the n-th prime)
- "other" — none of the above

Boundary rules:
- Prefer "enumeration" over "arithmetic_function" if the sequence counts structures, even if it can be written as f(n).
- Prefer "ranked_list" over "characteristic" for sequences like "the prime numbers" (2, 3, 5, 7, …)
  where the indexing is over qualifying elements rather than over all integers.
- Prefer "table_flattened" over "enumeration" for triangular tables like Pascal's,
  even though each row counts something.
- Pure polynomial closed forms with NO combinatorial interpretation (e.g., "a(n) = 12*n^2 + 1",
  "a(n) = n*(6*n+4)", "a(n) = (9*n^2 - 3*n + 2)/2") are "closed_form", NOT "arithmetic_function"
  ("arithmetic_function" is reserved for well-known number-theoretic functions like d(n), φ(n),
  σ(n), ω(n), Ω(n)).
</sequence-type>

<growth-class>
How fast does a(n) grow with n? (Look at the values preview AND the formula.)
- "finite" — sequence has finitely many terms (look for "fini" or "full" in keywords)
- "bounded" — bounded above; doesn't grow (constant sequence, single-digit terms like π's expansion)
- "linear" — grows like cn or cn + d
- "polynomial" — grows like n^k for some k > 1
- "exponential" — grows like c · r^n for r > 1 (Fibonacci, 2^n, Catalan ~ 4^n)
- "factorial_or_faster" — grows like n! or faster (factorials, powers of factorials, Bell numbers)
- "logarithmic_or_subpoly" — grows slower than any polynomial (the n-th prime ~ n log n is here, NOT polynomial)
- "oscillating" — does not have a clear monotonic growth (alternating signs, periodic, ±1 patterns)
- "unknown" — growth rate not determinable from the data given

Boundary rule: for ranked-list sequences (n-th prime, n-th squarefree), classify by how the n-th
ELEMENT grows, not by how the count of qualifying integers grows. The n-th prime grows ~ n log n,
which is "logarithmic_or_subpoly" (not "polynomial").
</growth-class>

<origin-era>
When was the underlying mathematical concept first studied? (NOT when the OEIS entry was created.)
Use the name, comments, formulas, **Author**, and **LastEdited** year to infer.
- "classical_pre1900" — known before 1900 (Fibonacci, primes, Catalan, factorials, π digits, Pascal,
  Euler totient, Bernoulli numbers, polygonal numbers, ancient combinatorial puzzles)
- "early_20c_1900_1950" — first studied 1900–1950 (Ramanujan partitions, Hardy-Littlewood era,
  Polya enumeration, Bell numbers, early algebraic combinatorics)
- "mid_20c_1950_2000" — first studied 1950–2000 (most computer-era sequences, OEIS founding era,
  Conway's recreational sequences, early CA work, modern algorithmic combinatorics)
- "modern_post2000" — first studied after 2000 (21st-century OEIS contributions, modern combinatorics
  papers, recent CA enumeration, contemporary number-theoretic experiments)
- "unknown" — cannot determine

How to use the Author and LastEdited fields:
- If **Author** contains an explicit year (e.g., "_Wolfdieter Lang_, May 04 2018"), that year is
  when the OEIS entry was created — a strong upper bound but NOT necessarily the original era.
  Combine it with the name/comments: if the comments establish pre-1900 pedigree (e.g., "studied
  by Euler", "Leonardo of Pisa, 1202"), use the historical era; otherwise the explicit Author year
  is the best signal we have.
- If Author is "_N. J. A. Sloane_" with NO year, the entry comes from the OEIS founder's seed
  population and could be any era — fall back to name/comments analysis.
- LastEdited is the most-recent-edit year, NOT the original authorship date. Use it ONLY as a
  weak negative signal: a sequence with LastEdited 2020+, no historical pedigree in name/comments,
  and a modern Author name leans toward modern_post2000 or mid_20c_1950_2000.

Default: if the sequence is named after a pre-1900 mathematician or refers to an ancient object,
pick "classical_pre1900" regardless of Author/LastEdited dates. For obscure sequences with comments
referencing modern computer experiments and no clear historical pedigree, prefer "modern_post2000"
when the Author year is post-2000, else "mid_20c_1950_2000". When truly uncertain, use "unknown" —
but try to commit to an era when there is even weak evidence.
</origin-era>

<examples>
A000045 Fibonacci numbers: F(n) = F(n-1) + F(n-2), values 0, 1, 1, 2, 3, 5, 8, 13, 21, …
→ math_domain=combinatorics, sequence_type=recurrence, growth_class=exponential, origin_era=classical_pre1900
Reasoning: counts pairs in many bijections; classic two-term recurrence; golden ratio growth; Leonardo of Pisa, 1202.

A000040 The prime numbers: 2, 3, 5, 7, 11, 13, 17, 19, 23, …
→ math_domain=number_theory, sequence_type=ranked_list,
  growth_class=logarithmic_or_subpoly, origin_era=classical_pre1900
Reasoning: number theory's foundational object; the n-th prime grows ~ n log n (subpolynomial); studied since Euclid.

A000108 Catalan numbers: 1, 1, 2, 5, 14, 42, 132, 429, …
→ math_domain=combinatorics, sequence_type=enumeration, growth_class=exponential, origin_era=classical_pre1900
Reasoning: counts many structures (Dyck paths, binary trees, triangulations); 4^n / n^(3/2) growth; Catalan 1838.

A000796 Decimal expansion of Pi: 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, …
→ math_domain=analysis, sequence_type=constant_digits, growth_class=bounded, origin_era=classical_pre1900
Reasoning: digits of a transcendental constant; bounded between 0 and 9; π is studied since antiquity.

A007318 Pascal's triangle read by rows: 1, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 4, 6, 4, 1, …
→ math_domain=combinatorics, sequence_type=table_flattened, growth_class=exponential, origin_era=classical_pre1900
Reasoning: a flattened triangular table of binomial coefficients; central column ~ 2^n / sqrt(n); Pascal 1654.
</examples>"""

# ── Pricing (USD per million tokens, as of 2026-04-10) ───────────────────────
# Update if Anthropic pricing has drifted.

MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
}

# ── Defaults ─────────────────────────────────────────────────────────────────

MAX_RETRIES = 5
MAX_PROMPT_COMMENT_CHARS = 800
MAX_PROMPT_FORMULA_CHARS = 400
MAX_PROMPT_AUTHOR_CHARS = 100


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a given (model, input_tokens, output_tokens)."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0
    return (input_tokens / 1e6) * pricing["input"] + (output_tokens / 1e6) * pricing["output"]


# ── User prompt building ─────────────────────────────────────────────────────


def build_user_prompt(rows: list[dict]) -> str:
    """Format a batch of OEIS sequences into a numbered classification prompt.

    Each row should have at least: id, name, offset, values_preview_str,
    keywords (list-like), comments, formulas. The optional ``author`` and
    ``last_edited`` fields, when present, are passed to the model as weak
    hints for the ``origin_era`` enum (see Sprint 6 prompt refinement).
    Missing fields are tolerated.
    """
    parts: list[str] = [
        f"Classify the following {len(rows)} OEIS sequences. "
        f"Return one classification per sequence using its 0-based index."
    ]

    for i, row in enumerate(rows):
        # `keywords` arrives as a numpy array from pandas.to_dict("records"); avoid
        # `or []` because numpy raises on ambiguous-truthiness for multi-element arrays.
        kws_raw = row.get("keywords")
        if kws_raw is None:
            kws = ""
        else:
            try:
                kws = ", ".join(str(k) for k in kws_raw)
            except TypeError:
                kws = str(kws_raw)

        name = (row.get("name") if row.get("name") is not None else "").strip()
        offset = row.get("offset") if row.get("offset") is not None else ""
        values = row.get("values_preview_str") if row.get("values_preview_str") is not None else ""
        comments_raw = row.get("comments") if row.get("comments") is not None else ""
        formula_raw = row.get("formulas") if row.get("formulas") is not None else ""
        comments = comments_raw[:MAX_PROMPT_COMMENT_CHARS]
        formula = formula_raw[:MAX_PROMPT_FORMULA_CHARS]

        # Author often includes an explicit year (e.g., "_Wolfdieter Lang_, May 04 2018").
        # Truncate aggressively because some sequences have long contributor lists.
        author_raw = row.get("author")
        if author_raw is None or (isinstance(author_raw, float) and pd.isna(author_raw)):
            author = "(unknown)"
        else:
            author = str(author_raw)[:MAX_PROMPT_AUTHOR_CHARS]

        # last_edited is a pandas Timestamp; render as year only to keep tokens minimal.
        # Note: this is the most-recent-edit year, NOT the original authorship date.
        last_edited_raw = row.get("last_edited")
        try:
            if last_edited_raw is None or pd.isna(last_edited_raw):
                last_edited_year = "(unknown)"
            else:
                last_edited_year = str(last_edited_raw.year)
        except (AttributeError, TypeError):
            last_edited_year = "(unknown)"

        parts.append(
            f"\n--- Sequence {i} ---\n"
            f"ID: {row.get('id', '?')}\n"
            f"Name: {name}\n"
            f"Offset: {offset}\n"
            f"Values: {values}\n"
            f"Keywords: {kws}\n"
            f"Author: {author}\n"
            f"LastEdited: {last_edited_year}\n"
            f"Comments: {comments}\n"
            f"Formula: {formula}"
        )

    return "\n".join(parts)


# ── Single-batch classify ────────────────────────────────────────────────────


def _null_result() -> dict[str, None]:
    return {col: None for col in ENUM_COLS}


def _validate_enum_value(field: str, value: object) -> str | None:
    """Defensive client-side enum validation; map invalid values to per-field fallback.

    Returns the original value if it's in the allowed enum, else the per-field
    fallback ("other" or "unknown") with a stderr warning. Returns None if value
    is None (which signals "row failed and should be retried on next run").
    """
    if value is None:
        return None
    sval = str(value)
    if sval in ENUM_VALUES[field]:
        return sval
    fallback = ENUM_FALLBACK[field]
    print(f"\n  WARNING: invalid {field}={sval!r} from API; mapping to {fallback!r}")
    return fallback


async def classify_batch(
    client: anthropic.AsyncAnthropic,
    rows: list[dict],
    model: str,
    semaphore: asyncio.Semaphore,
) -> tuple[list[dict], int, int]:
    """Classify one batch of OEIS sequences via tool-use.

    Returns ``(classifications, input_tokens, output_tokens)`` where
    ``classifications`` is a list aligned with ``rows`` (one dict per row,
    each containing the 4 enum fields or None values on unrecoverable failure).
    """
    if not rows:
        return [], 0, 0

    prompt = build_user_prompt(rows)
    response = None

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=[CLASSIFY_TOOL],
                    tool_choice={"type": "tool", "name": "classify_sequences"},
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except anthropic.RateLimitError as e:
                wait = min(2**attempt * 5, 60)
                print(f"\n  Rate limit ({e}); retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"\n  API error after {MAX_RETRIES} retries ({e}); marking batch null")
                    return [_null_result() for _ in rows], 0, 0
                wait = min(2**attempt * 5, 60)
                print(f"\n  API error ({e.__class__.__name__}: {e}); retrying in {wait}s")
                await asyncio.sleep(wait)
        else:
            # Exhausted retries on RateLimitError
            print(f"\n  Rate limit exhausted after {MAX_RETRIES} retries; marking batch null")
            return [_null_result() for _ in rows], 0, 0

    if response is None:
        return [_null_result() for _ in rows], 0, 0

    # Extract the tool_use block
    tool_use = next((b for b in response.content if getattr(b, "type", None) == "tool_use"), None)
    if tool_use is None:
        return [_null_result() for _ in rows], 0, 0

    raw = tool_use.input.get("classifications", []) if isinstance(tool_use.input, dict) else []
    by_idx: dict[int, dict] = {}
    for item in raw:
        if isinstance(item, dict) and "index" in item:
            by_idx[int(item["index"])] = item

    out: list[dict] = []
    for i in range(len(rows)):
        item = by_idx.get(i)
        if item is None:
            out.append(_null_result())
        else:
            out.append({col: _validate_enum_value(col, item.get(col)) for col in ENUM_COLS})

    in_tok = getattr(response.usage, "input_tokens", 0) or 0
    out_tok = getattr(response.usage, "output_tokens", 0) or 0
    return out, in_tok, out_tok


# ── Atomic checkpoint helper ─────────────────────────────────────────────────


def safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomically write a parquet file via temp + verify + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=str(path.parent), suffix=".parquet.tmp")
    os.close(tmp_fd)
    tmp_path = Path(tmp_path_str)
    try:
        df.to_parquet(tmp_path, index=False, compression="zstd")
        verify = pd.read_parquet(tmp_path)
        if len(verify) != len(df):
            raise RuntimeError(f"checkpoint verify mismatch: wrote {len(df)} rows, read back {len(verify)}")
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# ── Main entry: enrich a DataFrame ───────────────────────────────────────────


def _init_result(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the 4 enum cols added as nullable string dtype."""
    result = df.copy().reset_index(drop=True)
    for col in ENUM_COLS:
        if col not in result.columns:
            result[col] = pd.Series([pd.NA] * len(result), dtype="string")
        else:
            result[col] = result[col].astype("string")
    return result


async def enrich_dataframe(
    df: pd.DataFrame,
    *,
    client: anthropic.AsyncAnthropic,
    model: str,
    batch_size: int = 25,
    concurrency: int = 30,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 200,
) -> tuple[pd.DataFrame, int, int]:
    """Classify OEIS sequences in df via batched async tool-use.

    Returns ``(enriched_df, total_input_tokens, total_output_tokens)``. The
    DataFrame contains all input rows plus the 4 enum columns. Token totals
    cover only API calls made in this invocation (not rows skipped from a
    pre-existing checkpoint), so the caller can use them for a per-run cost
    estimate via :func:`estimate_cost`.

    Incremental: if ``checkpoint_path`` exists and contains rows for some of
    the input ids with all 4 enum cols populated, those rows are skipped on
    re-runs. Rows with any null enum col are re-classified.

    The result is checkpointed atomically every ``checkpoint_every`` rows.
    """
    if df.empty:
        print("  enrich_dataframe: input df is empty")
        return _init_result(df), 0, 0

    # ── Initialize result, possibly from existing checkpoint ─────────────────
    result = _init_result(df)
    id_to_idx: dict[str, int] = {row_id: idx for idx, row_id in result["id"].items()}

    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            existing = pd.read_parquet(checkpoint_path)
        except Exception as e:
            print(f"  Could not read checkpoint {checkpoint_path}: {e}; starting fresh")
            existing = None

        if existing is not None and "id" in existing.columns and all(c in existing.columns for c in ENUM_COLS):
            for _, row in existing.iterrows():
                row_id = row["id"]
                if row_id not in id_to_idx:
                    continue
                # Skip rows from a stale checkpoint that don't have all 4 cols populated
                if any(pd.isna(row[c]) for c in ENUM_COLS):
                    continue
                idx = id_to_idx[row_id]
                for col in ENUM_COLS:
                    result.at[idx, col] = row[col]

    # Determine which rows still need work (any enum col null)
    needs = result[list(ENUM_COLS)].isna().any(axis=1)
    rows_to_classify = result.loc[needs].to_dict("records")
    n_done = len(result) - len(rows_to_classify)

    if not rows_to_classify:
        print(f"  All {len(result):,} rows already enriched; nothing to do")
        return result, 0, 0

    print(f"  {model}: {n_done:,} already done, {len(rows_to_classify):,} to enrich")

    # ── Build batches ────────────────────────────────────────────────────────
    batches = [rows_to_classify[i : i + batch_size] for i in range(0, len(rows_to_classify), batch_size)]
    chunk_size_in_batches = max(1, checkpoint_every // batch_size)
    print(
        f"  {len(batches):,} batches × {batch_size} rows, "
        f"concurrency={concurrency}, checkpoint every ~{chunk_size_in_batches * batch_size} rows"
    )

    semaphore = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(rows_to_classify), desc=f"  {model}", unit="seq")

    total_in_tok = 0
    total_out_tok = 0

    async def _process_batch(batch_rows: list[dict]) -> tuple[list[dict], list[dict], int, int]:
        results, in_tok, out_tok = await classify_batch(client, batch_rows, model, semaphore)
        pbar.update(len(batch_rows))
        return batch_rows, results, in_tok, out_tok

    # ── Process in checkpoint-sized chunks ───────────────────────────────────
    for chunk_start in range(0, len(batches), chunk_size_in_batches):
        chunk = batches[chunk_start : chunk_start + chunk_size_in_batches]
        chunk_results = await asyncio.gather(*[_process_batch(b) for b in chunk])

        for batch_rows, batch_results, in_tok, out_tok in chunk_results:
            total_in_tok += in_tok
            total_out_tok += out_tok
            for row, classification in zip(batch_rows, batch_results):
                idx = id_to_idx[row["id"]]
                for col in ENUM_COLS:
                    val = classification.get(col)
                    result.at[idx, col] = val if val is not None else pd.NA

        if checkpoint_path is not None:
            safe_write_parquet(result, checkpoint_path)

    pbar.close()

    # ── Cost report ──────────────────────────────────────────────────────────
    cost = estimate_cost(model, total_in_tok, total_out_tok)
    print(
        f"  {model} done: {total_in_tok:,} in tokens + {total_out_tok:,} out tokens "
        f"→ ${cost:.4f} (pricing as of 2026-04-10)"
    )

    return result, total_in_tok, total_out_tok
