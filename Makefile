.PHONY: install lint test v0 pipeline-curated pipeline-curated-retrofit pipeline-full-retrofit clean-data

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

# Retrofit pipeline, curated scope. Assumes baseline stages 01-04 have already
# been run at curated scope (embeddings.npz exists). Parses %Y globally,
# retrofits against the curated in-scope subset, then regenerates stages 05-07
# reading from embeddings_retrofit.npz.
pipeline-curated-retrofit:
	OEIS_SCOPE=curated uv run python pipeline/01b_parse_y_edges.py
	OEIS_SCOPE=curated uv run python pipeline/04b_retrofit.py
	OEIS_SCOPE=curated OEIS_EMBEDDINGS_FILE=embeddings_retrofit.npz uv run python pipeline/05_reduce.py
	OEIS_SCOPE=curated OEIS_EMBEDDINGS_FILE=embeddings_retrofit.npz uv run python pipeline/06_label.py
	OEIS_SCOPE=curated OEIS_EMBEDDINGS_FILE=embeddings_retrofit.npz uv run python pipeline/07_visualize.py

# Retrofit pipeline, full scope (394k). Uses the alpha chosen during the curated
# run because full scope has no independent cross-check signal. Assumes baseline
# stages 01-04 have been run at OEIS_SCOPE=all first (this is the expensive
# prerequisite — stage 04 at full scope is ~394k Cohere embed calls).
pipeline-full-retrofit:
	OEIS_SCOPE=all uv run python pipeline/01b_parse_y_edges.py
	OEIS_SCOPE=all uv run python pipeline/04b_retrofit.py
	OEIS_SCOPE=all OEIS_EMBEDDINGS_FILE=embeddings_retrofit.npz uv run python pipeline/05_reduce.py
	OEIS_SCOPE=all OEIS_EMBEDDINGS_FILE=embeddings_retrofit.npz uv run python pipeline/06_label.py
	OEIS_SCOPE=all OEIS_EMBEDDINGS_FILE=embeddings_retrofit.npz uv run python pipeline/07_visualize.py

clean-data:
	@echo "This will delete everything under data/. Ctrl-C to abort."
	@read -p "Continue? [y/N] " ans && [ "$$ans" = "y" ] || exit 1
	rm -rf data/*
