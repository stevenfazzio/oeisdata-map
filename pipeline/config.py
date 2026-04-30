"""Shared paths, constants, and env var loading for the OEIS map pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load centralized credentials first, then project-local .env (which overrides)
load_dotenv(Path.home() / ".config" / "data-apis" / ".env")
load_dotenv(override=True)

# ── Directories ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = REPO_ROOT / "seq"  # input — READ-ONLY from the pipeline's perspective
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DOCS_DIR = REPO_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# ── File paths (one parquet/npz per pipeline stage) ──────────────────────────
RAW_PARQUET = DATA_DIR / "raw_sequences.parquet"  # stage 01 output (full 394k)
Y_EDGES_PARQUET = DATA_DIR / "y_edges.parquet"  # stage 01b output (global, all 394k)
SELECTED_PARQUET = DATA_DIR / "selected.parquet"  # stage 02 output (scope-filtered)
ENRICHED_PARQUET = DATA_DIR / "enriched.parquet"  # stage 03 output
# Stage 05/06 read embeddings from whichever file OEIS_EMBEDDINGS_FILE points at,
# so a single pipeline run can target either the baseline Cohere vectors
# (`embeddings.npz`) or the retrofit output (`embeddings_retrofit.npz`) without
# editing code. The default is the baseline. The index file is identical in
# both cases — it's just the `enriched.parquet` row order — so it has no override.
_EMBEDDINGS_FILENAME = os.environ.get("OEIS_EMBEDDINGS_FILE", "embeddings.npz")
EMBEDDINGS_NPZ = DATA_DIR / _EMBEDDINGS_FILENAME  # stage 04 output (or retrofit variant)
EMBEDDINGS_INDEX_NPY = DATA_DIR / "embeddings_index.npy"  # row IDs covered by EMBEDDINGS_NPZ
EMBEDDINGS_RETROFIT_NPZ = DATA_DIR / "embeddings_retrofit.npz"  # stage 04b output (always this name)
Y_EDGES_SPLIT_PARQUET = DATA_DIR / "y_edges_split.parquet"  # stage 04b: in-scope edges + train/holdout flag
RETROFIT_EVAL_JSON = DATA_DIR / "retrofit_eval.json"  # stage 04b: Hits@k metrics vs baseline
UMAP_COORDS_NPZ = DATA_DIR / "umap_coords.npz"  # stage 05
LABELS_PARQUET = DATA_DIR / "labels.parquet"  # stage 06
TOPONYMY_MODEL_JOBLIB = DATA_DIR / "toponymy_model.joblib"  # stage 06
OEIS_MAP_HTML = DATA_DIR / "oeis_map.html"  # stage 07 (local)
DOCS_INDEX_HTML = DOCS_DIR / "index.html"  # stage 07 (GitHub Pages, curated scope)
DOCS_FULL_HTML = DOCS_DIR / "full.html"  # stage 07 (GitHub Pages, all scope)

# ── Scope control ────────────────────────────────────────────────────────────
# `core`    → 183 sequences with the OEIS "core" keyword (v0 smoke test)
# `curated` → ~25k sequences via the quality filter in 02_select.py (v1)
# `all`     → all 394k sequences (stretch; stage 03 LLM enrichment is skipped)
SCOPE = os.environ.get("OEIS_SCOPE", "core").lower()
VALID_SCOPES = ("core", "curated", "all")
if SCOPE not in VALID_SCOPES:
    raise ValueError(f"OEIS_SCOPE must be one of {VALID_SCOPES}, got {SCOPE!r}")

CURATED_TARGET_SIZE = 25_000

# Optional override: skip stage 03 (LLM enrichment) at any scope. Stage 03
# automatically skips at SCOPE=all due to the $650 cost-cap; setting
# OEIS_SKIP_ENRICH=1 extends that behavior to any scope (useful during
# Sprint 5 when iterating on the taxonomy before committing to a full run).
SKIP_ENRICH = os.environ.get("OEIS_SKIP_ENRICH", "").strip().lower() in ("1", "true", "yes")

# ── API keys ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CO_API_KEY = os.environ.get("CO_API_KEY", "")

# ── LLM constants ────────────────────────────────────────────────────────────
# Stage 03 was Haiku in Sprints 3-5; switched to Sonnet in Sprint 6 after the
# 200-row eval showed Haiku trailing Opus by 8-20pp on every enum (worst on
# origin_era at 58% vs Sonnet's 77.5%). Sonnet at 25k curated costs ~$54 vs
# Haiku's ~$25, still well under the ~$100/run cap. See Sprint 6 plan in
# ~/.claude/plans/swift-snacking-kite.md for the full eval data.
ANTHROPIC_MODEL_ENRICH = "claude-sonnet-4-5"  # stage 03: classification (Sprint 6 onward)
# Stage 06 namer is overridable via env so experiments (e.g., α-grid comparison
# at lower cost) can swap Sonnet → Haiku without editing code. Production
# renders default to Sonnet.
ANTHROPIC_MODEL_NAMER = os.environ.get("ANTHROPIC_MODEL_NAMER", "claude-sonnet-4-5")
ANTHROPIC_CONCURRENCY = 30  # async semaphore size for stage 03
ENRICH_BATCH_SIZE = 25  # sequences per tool-use call

# ── Cohere ───────────────────────────────────────────────────────────────────
COHERE_BATCH_SIZE = 96
COHERE_EMBED_DIMENSION = 512
COHERE_EMBED_MODEL = "embed-v4.0"

# ── Parser ───────────────────────────────────────────────────────────────────
PARSE_WORKERS = max(4, (os.cpu_count() or 4) - 1)

# ── Text budgets (Fibonacci-pathological-comments mitigation) ────────────────
MAX_NAME_CHARS = 300
MAX_COMMENT_CHARS = 1500
MAX_FORMULA_CHARS = 500
MAX_EXAMPLE_CHARS = 300
MAX_VALUES_SHOWN = 15  # how many leading terms to keep for display + embedding hints
