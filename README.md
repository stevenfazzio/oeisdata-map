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
| 02 | `02_select.py`    | raw → `data/selected.parquet` (applies `OEIS_SCOPE` filter) |
| 03 | `03_enrich.py`    | selected → `data/enriched.parquet` (Claude taxonomy classification) |
| 04 | `04_embed.py`     | enriched → `data/embeddings.npz` (Cohere embeddings) |
| 05 | `05_reduce.py`    | embeddings → `data/umap_coords.npz` (UMAP to 2D) |
| 06 | `06_label.py`     | coords + embeddings → `data/labels.parquet` (Toponymy topic labels) |
| 07 | `07_visualize.py` | all of the above → `data/oeis_map.html` + `docs/index.html` |

Pipeline stages `01b_parse_y_edges.py` and `04b_retrofit.py` exist but aren't
part of the canonical pipeline — see the experimental retrofit section below.

## Experimental: %Y-edge retrofit

Stages 01b and 04b implement Faruqui-style Laplacian smoothing of the Cohere
vectors against the OEIS `%Y` cross-reference graph (Faruqui et al. NAACL
2015). The idea: `%Y` edges are human-curated "related to" links and might
let us pull linked sequences closer in embedding space without retraining.

Once the embed text was upgraded to include worked examples and drop content
keywords (so they could serve as an orthogonal eval signal), the smoothing
turned out to trade content structure for graph reconstruction without a
clear win on visual cluster quality. The published map is built without it.

The code is preserved as an experiment; the eval framework reports both
held-out `%Y` Hits@10 (graph self-consistency) and per-content-keyword
silhouette (orthogonal to the training signal) per (α, n_iter) grid point:

```
make pipeline-curated-retrofit
```

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
  cost-prohibitive due to O(n²) cluster-name disambiguation. Not pursued
  as a published deliverable.

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
