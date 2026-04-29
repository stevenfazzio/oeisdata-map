# CLAUDE.md

## Project Overview

Semantic map of the Online Encyclopedia of Integer Sequences (OEIS). The
pipeline reads from `seq/` (a sparse checkout of the upstream `oeis/oeisdata`
repository, populated by `scripts/sync_seq.sh`) and writes to `data/` and
`docs/`.

## Hard rules

- `seq/` is a build input, not source. It's gitignored and managed by
  `scripts/sync_seq.sh` (sparse-checkout of `oeis/oeisdata`). **Never** create,
  modify, or delete anything under `seq/` from this repo's code or commits.
  To refresh, re-run the sync script.
- `LICENSE` carries the upstream OEIS data attribution; do not modify.
- Treat `data/` as expensive but reproducible. Pipeline stages write outputs
  via temp+rename for atomicity. Never overwrite a parquet file in place.
- LLM-enriched and embedded outputs are incremental: re-running a stage
  should only process rows that aren't already present in the output.

## Pipeline

Stages are idempotent and resumable. Each reads outputs of the previous:

```
python pipeline/01_parse.py          # seq/*.seq → data/raw_sequences.parquet
python pipeline/01b_parse_y_edges.py # seq/*.seq → data/y_edges.parquet (global %Y cross-refs; retrofit only)
python pipeline/02_select.py         # raw → data/selected.parquet (scope filter)
python pipeline/03_enrich.py         # selected → data/enriched.parquet (LLM classify; SKIPPED at OEIS_SCOPE=all)
python pipeline/04_embed.py          # enriched → data/embeddings.npz (Cohere)
python pipeline/04b_retrofit.py      # embeddings + y_edges → embeddings_retrofit.npz (Faruqui smoothing; optional)
python pipeline/05_reduce.py         # embeddings → data/umap_coords.npz (UMAP)
python pipeline/06_label.py          # Toponymy topic labels → data/labels.parquet
python pipeline/07_visualize.py      # DataMapPlot → data/oeis_map.html + docs/index.html (or docs/full.html at scope=all)
```

Stages 01b + 04b are the retrofit pipeline: they re-pull Cohere vectors toward
their OEIS `%Y` cross-reference neighbors via closed-form Laplacian smoothing.
Downstream stages opt into retrofit by setting `OEIS_EMBEDDINGS_FILE=embeddings_retrofit.npz`.
The published curated map (`docs/index.html`) is built with retrofit at α=0.8,
producing visibly sharper clusters than pure text embeddings.

## Scope control

The `OEIS_SCOPE` env var selects which sequences to process:

- `core` (default for development) — 183 sequences with the OEIS `core`
  keyword. Used for fast end-to-end validation runs (~5 min, <$5).
- `curated` — ~25k sequences via a quality filter (see `02_select.py`). The
  publishable v1 target (~1 hour, ~$65 with Sonnet stage 03).
- `all` — all 394,561 sequences. **Stage 03 (LLM enrichment) is automatically
  skipped at this scope** because a full Sonnet enrichment would cost ~$1900.
  **Stage 06 (Toponymy cluster naming) is ALSO cost-prohibitive at full scope**
  — its disambiguation pass is O(n²) in cluster count per layer, and empirical
  runs have shown a single layer (of 7) consuming ~$100 on Sonnet before
  credits exhausted. A complete full-scope stage 06 would likely cost $300–700+.
  **The curated 25k is the publishable deliverable; full scope is not pursued.**
  If revisited, swap `ANTHROPIC_MODEL_NAMER` to Haiku 4.5 (3× cheaper) or skip
  stage 06 entirely and ship with only keyword-based colormaps.

**Stage 03 model:** Sonnet 4.5 since Sprint 6. Sprint 3-5 used Haiku 4.5;
the Sprint 6 eval showed Haiku trailing Opus by 8-20pp on every enum, so
production switched to Sonnet for the curated map at ~3× the per-row cost.
See `~/.claude/plans/swift-snacking-kite.md` for the full eval data.

## Required Environment Variables

- `ANTHROPIC_API_KEY` — Claude (stages 03, 06)
- `CO_API_KEY` — Cohere (stages 04, 06)

Copy `.env.example` to `.env` and fill in your keys.

## OEIS file format notes

Sequence files are plain text with one `%X AXXXXXX <content>` line per field.
Fields parsed by stage 01:

- `%I` header — sequence ID, edit count, last-edited date
- `%N` — name (single line)
- `%C` — comments (multi-line)
- `%F` — formulas (multi-line)
- `%e` — examples (multi-line)
- `%S`/`%T`/`%U` — sequence values (multi-line, comma-separated)
- `%H` — hyperlinks (multi-line)
- `%D` — book/paper references (multi-line)
- `%K` — keywords (single line, comma-separated)
- `%O` — offset
- `%A` — author
- `%p`, `%t`, `%o` — code in Maple, Mathematica, and other languages
- `%E` — extension/edit history

**Two `%I` header forms exist** (parser must handle both):

```
%I A000001 M0098 N0035 #326 Jan 28 2026 13:29:57   (legacy: includes Sloane/Plouffe book IDs)
%I A300000 #22 Jul 08 2022 12:19:01                (modern: no M/N, post-~2007)
```

Cross-references (`%Y`) are parsed by stage 01b (sidecar) into a global edge
table and consumed by stage 04b to retrofit Cohere embeddings via Faruqui
smoothing. Stage 01 itself still deliberately ignores `%Y`.

## Development

```
make install                     # uv sync --extra dev
make lint                        # ruff check + format check
make test                        # pytest
make v0                          # OEIS_SCOPE=core, full pipeline end-to-end
make pipeline-curated            # OEIS_SCOPE=curated, full pipeline (~$65)
make pipeline-curated-retrofit   # stages 01b + 04b + 05–07 with retrofit (free after baseline)
make pipeline-full-retrofit      # NOT RECOMMENDED: see stage 06 cost warning above
make clean-data                  # rm -rf data/* (interactive confirm)
```

## Reference projects

The architecture is ported from two prior map projects in sibling repos:

- `~/repos/semantic-github-map/` — primary template (top-10k GitHub repos
  → Cohere embed → UMAP → Toponymy → DataMapPlot)
- `~/repos/claude-code-changelog-analysis/` — secondary template, especially
  for the structured tool-use LLM classification pattern in stage 03 and the
  multi-model (Haiku/Sonnet/Opus) eval harness in `eval/`
