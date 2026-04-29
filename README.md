# OEIS Semantic Map

An interactive 2D map of the [Online Encyclopedia of Integer Sequences](https://oeis.org).
A pipeline embeds sequences, reduces them with UMAP, clusters them, and renders
the result with DataMapPlot.

**Live map:** https://stevenfazzio.github.io/oeisdata-map/

## Sequence data

The pipeline reads from `seq/`, a sparse checkout of the upstream
[`oeis/oeisdata`](https://github.com/oeis/oeisdata) repository. `seq/` is
gitignored — it's a build input, not source. To populate or refresh it:

```
bash scripts/sync_seq.sh
```

The first run clones upstream with `--filter=blob:none` and a sparse-checkout
restricted to `seq/` (skipping the supplementary `files/` assets entirely).
Subsequent runs are fast `git pull` updates.

## Pipeline

Each stage reads the previous stage's output and writes via temp+rename. Stages
are idempotent and resumable — re-running only processes rows that aren't
already in the output.

| Stage | Script | Input → Output |
| --- | --- | --- |
| 01 | `01_parse.py`     | `seq/*.seq` → `data/raw_sequences.parquet` |
| 01b | `01b_parse_y_edges.py` | `seq/*.seq` → `data/y_edges.parquet` (global `%Y` cross-reference edges; used only by the retrofit pipeline) |
| 02 | `02_select.py`    | raw → `data/selected.parquet` (applies `OEIS_SCOPE` filter) |
| 03 | `03_enrich.py`    | selected → `data/enriched.parquet` (Claude taxonomy classification) |
| 04 | `04_embed.py`     | enriched → `data/embeddings.npz` (Cohere embeddings) |
| 04b | `04b_retrofit.py` | embeddings + `y_edges.parquet` → `data/embeddings_retrofit.npz` + `retrofit_eval.json` (optional graph-aware post-process) |
| 05 | `05_reduce.py`    | embeddings → `data/umap_coords.npz` (UMAP to 2D) |
| 06 | `06_label.py`     | coords + embeddings → `data/labels.parquet` (Toponymy topic labels) |
| 07 | `07_visualize.py` | all of the above → `data/oeis_map.html` + `docs/index.html` |

## Retrofit pipeline (optional)

Stages 01b and 04b produce a `%Y`-aware variant of the Cohere embeddings. The
idea: `%Y` cross-references are human-curated "related to" edges, and we can
use them to pull linked sequences closer together in embedding space via a
closed-form Laplacian smoother (Faruqui et al. NAACL 2015). No model
retraining, no GPU, deterministic.

Run it (curated scope):

```
make pipeline-curated-retrofit
```

This assumes the baseline pipeline (`make pipeline-curated`) has already been
run, so `data/embeddings.npz` exists. It then runs `01b_parse_y_edges.py`,
`04b_retrofit.py`, and regenerates stages 05–07 with
`OEIS_EMBEDDINGS_FILE=embeddings_retrofit.npz` pointing downstream stages at
the retrofit output.

The retrofit stage holds out 10% of `%Y` edges, grid-searches `α ∈ {0.2, 0.4,
0.6, 0.8, 0.9, 1.0}` × `n_iter ∈ {5, 10}`, evaluates Hits@10 / Hits@100 / MRR on
the held-out pairs against the baseline, and saves:

- `data/retrofit_eval.json` — baseline vs. every grid point, with winner
- `data/y_edges_split.parquet` — in-scope edges with a `split` column
- `data/embeddings_retrofit.npz` — winning retrofit embeddings

To point any individual downstream stage at the retrofit output, set
`OEIS_EMBEDDINGS_FILE=embeddings_retrofit.npz` in the environment. Unset it
(or leave it empty) to fall back to `embeddings.npz`.

**Full scope (`OEIS_SCOPE=all`) is not recommended** — the Makefile target
exists (`make pipeline-full-retrofit`) but stage 06's Toponymy cluster-name
disambiguation is O(n²) in clusters per layer. Empirical runs at full scope
have shown a single layer consuming ~$100 on Sonnet before credits exhausted;
a complete run would likely cost $300–700+. The curated 25k is the intended
deliverable. If you need to attempt full scope anyway, swap
`ANTHROPIC_MODEL_NAMER` to Haiku 4.5 (3× cheaper) first.

## Quick start

```
cp .env.example .env          # then fill in ANTHROPIC_API_KEY and CO_API_KEY
make install                  # uv sync --extra dev
bash scripts/sync_seq.sh      # sparse-checkout seq/ from oeis/oeisdata
make v0                       # full pipeline at core scope (~183 sequences)
open docs/index.html
```

Scope is selected via the `OEIS_SCOPE` environment variable:

- `core` — 183 sequences tagged with the OEIS `core` keyword. Fast end-to-end
  smoke test. This is what `make v0` runs.
- `curated` — ~25k sequences via a quality filter. The publishable v1 target.
  Run with `make pipeline-curated`.
- `all` — all ~394k sequences. Stage 03 (LLM enrichment) is automatically
  skipped at this scope; stage 06 (Toponymy cluster naming) is also
  cost-prohibitive due to O(n²) disambiguation. Not pursued as a published
  deliverable — see the retrofit section above for details.

## Credits

- Sequence data: [OEIS Foundation](https://oeis.org) and its contributors.
- Embeddings: [Cohere](https://cohere.com) `embed-english-v3.0`.
- Taxonomy classification: [Anthropic Claude](https://www.anthropic.com).
- Layout: [UMAP](https://umap-learn.readthedocs.io).
- Topic labels: [Toponymy](https://github.com/TutteInstitute/toponymy).
- Rendering: [DataMapPlot](https://github.com/TutteInstitute/datamapplot).

Architecture adapted from two prior sibling map projects:
[semantic-github-map](https://github.com/stevenfazzio/semantic-github-map) and
an internal claude-code-changelog-analysis project.
