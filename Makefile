.PHONY: install lint test v0 pipeline-curated pipeline-curated-retrofit clean-data

install:
	uv sync --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

# Full pipeline, core scope (~183 sequences, end-to-end smoke test).
v0:
	OEIS_SCOPE=core uv run python pipeline/01_parse.py
	OEIS_SCOPE=core uv run python pipeline/02_select.py
	OEIS_SCOPE=core uv run python pipeline/03_enrich.py
	OEIS_SCOPE=core uv run python pipeline/04_embed.py
	OEIS_SCOPE=core uv run python pipeline/05_reduce.py
	OEIS_SCOPE=core uv run python pipeline/06_label.py
	OEIS_SCOPE=core uv run python pipeline/07_visualize.py

# Full pipeline, curated scope (~25k sequences, publishable v1 map).
pipeline-curated:
	OEIS_SCOPE=curated uv run python pipeline/01_parse.py
	OEIS_SCOPE=curated uv run python pipeline/02_select.py
	OEIS_SCOPE=curated uv run python pipeline/03_enrich.py
	OEIS_SCOPE=curated uv run python pipeline/04_embed.py
	OEIS_SCOPE=curated uv run python pipeline/05_reduce.py
	OEIS_SCOPE=curated uv run python pipeline/06_label.py
	OEIS_SCOPE=curated uv run python pipeline/07_visualize.py

# EXPERIMENTAL: %Y-edge retrofit pipeline. NOT used by the published map.
# We tried Faruqui-style Laplacian smoothing of the Cohere vectors against
# the OEIS %Y graph; once the embed text included examples and dropped
# content keywords (the v2 baseline), the smoothing destroyed content
# structure faster than it improved graph reconstruction. Code preserved
# for experimentation; eval framework in pipeline/04b_retrofit.py reports
# Hits@10 + per-keyword silhouette per (alpha, n_iter) grid point.
pipeline-curated-retrofit:
	OEIS_SCOPE=curated uv run python pipeline/01b_parse_y_edges.py
	OEIS_SCOPE=curated uv run python pipeline/04b_retrofit.py
	OEIS_SCOPE=curated OEIS_EMBEDDINGS_FILE=embeddings_retrofit.npz uv run python pipeline/05_reduce.py
	OEIS_SCOPE=curated OEIS_EMBEDDINGS_FILE=embeddings_retrofit.npz uv run python pipeline/06_label.py
	OEIS_SCOPE=curated OEIS_EMBEDDINGS_FILE=embeddings_retrofit.npz uv run python pipeline/07_visualize.py

clean-data:
	@echo "This will delete everything under data/. Ctrl-C to abort."
	@read -p "Continue? [y/N] " ans && [ "$$ans" = "y" ] || exit 1
	rm -rf data/*
