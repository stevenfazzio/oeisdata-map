#!/usr/bin/env bash
# Sync the upstream OEIS seq/ directory into ./seq via a sparse, blob-filtered
# clone of oeis/oeisdata. Skips the supplementary files/ assets entirely.
# First run clones; subsequent runs fast-forward.

set -euo pipefail

UPSTREAM_URL="${OEIS_UPSTREAM_URL:-https://github.com/oeis/oeisdata.git}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLONE_DIR="$REPO_ROOT/.cache/oeisdata-upstream"
SEQ_LINK="$REPO_ROOT/seq"

if [ ! -d "$CLONE_DIR/.git" ]; then
    echo "Cloning $UPSTREAM_URL (sparse, blob-filtered) into .cache/oeisdata-upstream/…"
    mkdir -p "$(dirname "$CLONE_DIR")"
    git clone \
        --filter=blob:none \
        --no-checkout \
        --sparse \
        "$UPSTREAM_URL" \
        "$CLONE_DIR"
    git -C "$CLONE_DIR" sparse-checkout set --cone seq
    git -C "$CLONE_DIR" checkout main
else
    echo "Updating sparse clone in .cache/oeisdata-upstream/…"
    git -C "$CLONE_DIR" pull --ff-only
fi

if [ ! -e "$SEQ_LINK" ]; then
    ln -s "$CLONE_DIR/seq" "$SEQ_LINK"
    echo "Linked $SEQ_LINK -> $CLONE_DIR/seq"
elif [ -L "$SEQ_LINK" ]; then
    echo "Symlink $SEQ_LINK already in place."
else
    echo "WARNING: $SEQ_LINK exists but is not a symlink. Leaving it alone." >&2
fi

count=$(find -L "$SEQ_LINK" -name '*.seq' -type f 2>/dev/null | wc -l | tr -d ' ')
echo "Sync complete. $count .seq files available under seq/."
