"""Render the OEIS interactive map with DataMapPlot.

Minimal UI (colormap dropdown + hover + click-through, no filter panel).
The full filter UI is deferred to Sprint 6 alongside a redesigned stage 03
enrichment prompt.

Reads:
- ``data/enriched.parquet``     — id, name, comments, values_preview_str,
                                  edit_count, n_references, keywords list,
                                  and optionally math_domain / sequence_type /
                                  growth_class / origin_era (from stage 03).
- ``data/umap_coords.npz``      — float32 (N, 2) under key ``coords``
- ``data/labels.parquet``       — id + label_layer_*

Writes:
- ``data/oeis_map.html``        — local preview
- ``docs/index.html``           — GitHub Pages entry point (identical copy)

**LLM-column handling:** if ``enriched.parquet`` is missing any of the 4
stage 03 LLM columns (math_domain / sequence_type / growth_class /
origin_era), the map still renders — just with fewer colormaps and a
simpler hover template. This lets the ``OEIS_SKIP_ENRICH=1`` pathway
work at any scope without crashing.

Design decisions:
- Marker size: ``sqrt(edit_count)`` linearly scaled to [3, 15] px
  → Fibonacci (max edit_count) is the largest dot
- Colormaps always include: primary_keyword (derived from OEIS keywords
  list), log10(edit_count), log10(n_references + 1), plus any Toponymy
  hierarchy layers (coarsest first).
- Colormaps conditional on stage 03: math_domain, sequence_type,
  growth_class, origin_era.
- Click handler: opens https://oeis.org/{id} in a new tab.
"""

from __future__ import annotations

import re
import shutil
from html import escape

import datamapplot
import glasbey
import numpy as np
import pandas as pd
from config import (
    DOCS_FULL_HTML,
    DOCS_INDEX_HTML,
    ENRICHED_PARQUET,
    LABELS_PARQUET,
    OEIS_MAP_HTML,
    SCOPE,
    UMAP_COORDS_NPZ,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

# OEIS has many keywords on a single sequence (e.g., core,nice,easy,hard).
# Collapse each row to ONE primary category via fixed precedence — the
# earlier entries "win" because they're the most interesting labels.
_KEYWORD_PRECEDENCE = ("core", "nice", "easy", "hard", "tabl", "cons", "base")

# The 4 stage-03 LLM columns. If any are missing, stage 07 falls back to
# primary_keyword + continuous colormaps only.
_LLM_COLS = ("math_domain", "sequence_type", "growth_class", "origin_era")

# Author colormap: top N most-prolific contributors in the selected set get
# their own bucket; the long tail collapses into "Other". 10 + Other gives
# enough granularity to surface major contributors without diluting the
# palette beyond legibility.
_AUTHOR_TOP_N = 10
_AUTHOR_OTHER_LABEL = "Other"

# Match the FIRST `_..._` underscore-wrapped block at the start of a `%A` line
# — that's the canonical primary author. Rest (joint authors, dates, email
# notes) is dropped.
_AUTHOR_PRIMARY_RE = re.compile(r"^_([^_]+)_")


def _esc(values) -> list[str]:
    """HTML-escape every value (defends against `%C` raw HTML fragments)."""
    return [escape(str(v) if v is not None else "") for v in values]


def _glasbey_mapping(values) -> dict[str, str]:
    """Build a {category → hex color} dict from a Glasbey palette."""
    unique = sorted({(v if v else "unknown") for v in values})
    palette = glasbey.create_palette(palette_size=len(unique))
    return dict(zip(unique, palette))


def _primary_keyword(kws) -> str:
    """Map a row's keywords list to a single primary label via fixed precedence.

    Returns the highest-precedence keyword present, or "other" if none match.
    Accepts numpy arrays (from parquet list cols), Python lists, or None.
    """
    if kws is None:
        return "other"
    try:
        kw_set = {str(k) for k in kws}
    except TypeError:
        return "other"
    for p in _KEYWORD_PRECEDENCE:
        if p in kw_set:
            return p
    return "other"


def _darken_for_text(hex_color: str, threshold: float = 0.72, factor: float = 0.55) -> str:
    """Darken pale hex colors so they remain legible as text on a cream card.

    sRGB luminance threshold; values above it get multiplied by `factor`.
    Map dots use the original palette — only tooltip text values are darkened.
    """
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return hex_color
    try:
        r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return hex_color
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum <= threshold:
        return hex_color
    r, g, b = (int(round(c * factor * 255)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _first_formula(f: str | None, max_chars: int = 140) -> str:
    """First non-empty line of %F, newline-flattened, ellipsized to max_chars.

    Returns "" when no formula is present so CSS `:empty { display: none }`
    can collapse the row in the hover card.
    """
    if not f:
        return ""
    for line in f.splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= max_chars else line[: max_chars - 1].rstrip() + "…"
    return ""


def _normalize_marker_sizes(values: np.ndarray, lo: float = 3.0, hi: float = 15.0) -> np.ndarray:
    """Linearly stretch sqrt(values) to [lo, hi] px."""
    sqrt_v = np.sqrt(np.clip(values.astype(float), 1.0, None))
    vmin, vmax = sqrt_v.min(), sqrt_v.max()
    if vmax - vmin < 1e-9:
        return np.full_like(sqrt_v, (lo + hi) / 2)
    return lo + (hi - lo) * (sqrt_v - vmin) / (vmax - vmin)


def _clean_author(a: str | None) -> str:
    """Extract a canonical primary-author name from a raw `%A` line.

    Most modern entries wrap the name in markdown underscores and may append
    ", Mon DD YYYY" or further contributor notes:

        "_N. J. A. Sloane_, Jul 14 2001"               -> "N. J. A. Sloane"
        "_Sloane_ and _Plouffe_"                       -> "Sloane"
        "_N. J. A. Sloane_, based on a message from …" -> "N. J. A. Sloane"

    Older entries lack the underscore wrap and may have an email in parens:

        "Jonathan Wellons (wellons(AT)gmail.com), Jan 22 2008" -> "Jonathan Wellons"

    Returns "" for missing/empty values; those are bucketed into "Other".
    """
    if not a:
        return ""
    m = _AUTHOR_PRIMARY_RE.match(a)
    if m:
        return m.group(1).strip()
    # No underscore wrap: take the head before the first comma, paren, or " and ".
    return re.split(r"\s+and\s+|,|\(", a, maxsplit=1)[0].strip()


def _bucketize_authors(values: np.ndarray, top_n: int = _AUTHOR_TOP_N) -> np.ndarray:
    """Replace any author outside the top-N most frequent with `_AUTHOR_OTHER_LABEL`."""
    counts = pd.Series(values).value_counts()
    # An empty cleaned name (anonymous / unparseable) always lands in Other.
    counts = counts.drop(labels=[""], errors="ignore")
    keep = set(counts.head(top_n).index)
    return np.array([(v if v in keep else _AUTHOR_OTHER_LABEL) for v in values])


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    for label, path in (
        ("enriched.parquet", ENRICHED_PARQUET),
        ("umap_coords.npz", UMAP_COORDS_NPZ),
        ("labels.parquet", LABELS_PARQUET),
    ):
        if not path.exists():
            raise SystemExit(f"{label} not found at {path} — run earlier pipeline stages first")

    print(f"Reading {ENRICHED_PARQUET.name}…")
    df = pd.read_parquet(ENRICHED_PARQUET)
    print(f"  {len(df):,} rows")

    print(f"Reading {UMAP_COORDS_NPZ.name}…")
    coords = np.load(UMAP_COORDS_NPZ)["coords"]
    print(f"  {coords.shape} {coords.dtype}")

    print(f"Reading {LABELS_PARQUET.name}…")
    labels_df = pd.read_parquet(LABELS_PARQUET)
    # Align labels to enriched.parquet's row order via id
    labels_df = labels_df.set_index("id").reindex(df["id"]).reset_index()
    layer_cols = sorted([c for c in labels_df.columns if c.startswith("label_layer_")])
    if not layer_cols:
        raise SystemExit("labels.parquet has no label_layer_* columns — re-run pipeline/06_label.py")
    print(f"  {len(layer_cols)} layer(s): {layer_cols}")

    if len(coords) != len(df):
        raise SystemExit(f"Length mismatch: coords={len(coords)} vs enriched={len(df)}")

    # ── Detect whether stage 03 LLM columns are present ──────────────────────
    has_llm = all(col in df.columns for col in _LLM_COLS)
    if has_llm:
        print("  LLM enum columns present — including full colormap set")
    else:
        missing = [c for c in _LLM_COLS if c not in df.columns]
        print(f"  LLM enum columns missing ({', '.join(missing)}); falling back to primary_keyword + continuous only")

    # ── primary_keyword derivation (always available) ────────────────────────
    primary_keyword_vals = np.array([_primary_keyword(kws) for kws in df["keywords"]])
    pk_counts = pd.Series(primary_keyword_vals).value_counts()
    print(f"  primary_keyword distribution: {dict(pk_counts)}")

    # ── Hover text + template ────────────────────────────────────────────────
    # DataMapPlot uses `hover_text` as the search index (search_field defaults
    # to "hover_text"), so we pack every field we want searchable into this
    # string. The visible tooltip uses `hover_text_html_template` below, not
    # this string, so formatting here is purely for search matching.
    #
    # Included:
    #   - id            → "A000045"
    #   - name          → "Fibonacci numbers: F(n) = F(n-1) + F(n-2) with..."
    #   - author        → "N. J. A. Sloane" (markdown wrap and trailing date
    #                     stripped via _clean_author)
    #   - comments[:400] → first 400 chars of %C, newlines flattened to
    #                     spaces. Captures "Ramanujan", "Mersenne", named
    #                     theorems, and historical context.
    _COMMENT_SEARCH_CHARS = 400

    author_clean_vals = np.array([_clean_author(a) for a in df["author"]])

    def _clean_comments(c: str | None) -> str:
        if not c:
            return ""
        return c.replace("\n", " ")[:_COMMENT_SEARCH_CHARS]

    hover_text = [
        f"{sid} {name} {author} {_clean_comments(comments)}"
        for sid, name, author, comments in zip(df["id"], df["name"], author_clean_vals, df["comments"], strict=True)
    ]

    # Bucketize authors for the colormap: top N + Other.
    author_bucketed = _bucketize_authors(author_clean_vals)
    author_top = pd.Series(author_bucketed).value_counts()
    print(
        f"  author distribution: {len(author_top)} buckets, top entry "
        f"{author_top.index[0]!r} ({author_top.iloc[0]} seqs)"
    )

    # Three-section editorial card: mono header (id + edits/refs) → serif
    # body (name + optional formula + mono value run) → mono tag footer.
    # `.oeis-formula:empty` and tag `span:empty` let missing fields collapse
    # without leaving hollow rows.
    _hover_head = (
        '<div class="oeis-card">'
        '  <div class="oeis-head">'
        '    <span class="oeis-id">{id}</span>'
        '    <span class="oeis-meta">'
        '      <span class="oeis-meta-k">edits</span> {edit_count}'
        '      <span class="oeis-dot">·</span>'
        '      <span class="oeis-meta-k">refs</span> {n_references}'
        "    </span>"
        "  </div>"
        '  <div class="oeis-body">'
        '    <div class="oeis-name">{name}</div>'
        '    <div class="oeis-formula">{first_formula}</div>'
        '    <div class="oeis-values">{values_preview_str}</div>'
        "  </div>"
    )
    # Footer is a 2-col grid of <label, value> rows. Each value is colored by
    # its own colormap's palette (computed once, reused for colormap_metadata
    # below). `.oeis-dim:has(.oeis-dim-v:empty)` collapses rows with missing
    # LLM values so we never show a dangling label.
    _author_dim = (
        '    <div class="oeis-dim">'
        '      <span class="oeis-dim-k">Author</span>'
        '      <span class="oeis-dim-v" style="color:{author_color}">{author}</span>'
        "    </div>"
    )
    if has_llm:
        hover_html = (
            _hover_head + '  <div class="oeis-dims">' + _author_dim + '    <div class="oeis-dim">'
            '      <span class="oeis-dim-k">Domain</span>'
            '      <span class="oeis-dim-v" style="color:{math_domain_color}">{math_domain}</span>'
            "    </div>"
            '    <div class="oeis-dim">'
            '      <span class="oeis-dim-k">Type</span>'
            '      <span class="oeis-dim-v" style="color:{sequence_type_color}">{sequence_type}</span>'
            "    </div>"
            '    <div class="oeis-dim">'
            '      <span class="oeis-dim-k">Growth</span>'
            '      <span class="oeis-dim-v" style="color:{growth_class_color}">{growth_class}</span>'
            "    </div>"
            "  </div>"
            "</div>"
        )
    else:
        hover_html = (
            _hover_head + '  <div class="oeis-dims">' + _author_dim + '    <div class="oeis-dim">'
            '      <span class="oeis-dim-k">Keyword</span>'
            '      <span class="oeis-dim-v" style="color:{primary_keyword_color}">{primary_keyword}</span>'
            "    </div>"
            "  </div>"
            "</div>"
        )

    # Palette dicts: one per categorical colormap. Reused below when building
    # colormap_metadata so the tooltip colors are guaranteed to match the map.
    pk_colors = _glasbey_mapping(primary_keyword_vals)
    author_colors = _glasbey_mapping(author_bucketed)
    if has_llm:
        md_vals = df["math_domain"].fillna("unknown").to_numpy()
        st_vals = df["sequence_type"].fillna("unknown").to_numpy()
        gc_vals = df["growth_class"].fillna("unknown").to_numpy()
        oe_vals = df["origin_era"].fillna("unknown").to_numpy()
        md_colors = _glasbey_mapping(md_vals)
        st_colors = _glasbey_mapping(st_vals)
        gc_colors = _glasbey_mapping(gc_vals)
        oe_colors = _glasbey_mapping(oe_vals)

    def _row_colors(values, palette) -> list[str]:
        """Look up each row's category → (darkened, if pale) hex for tooltip text."""
        return [_darken_for_text(palette.get(v or "unknown", "#5a5a5a")) for v in values]

    extra_data_dict = {
        "id": _esc(df["id"]),
        "name": _esc(df["name"]),
        "first_formula": _esc([_first_formula(f) for f in df["formulas"]]),
        "values_preview_str": _esc(df["values_preview_str"]),
        "edit_count": df["edit_count"].fillna(0).astype(int).astype(str).tolist(),
        "n_references": df["n_references"].fillna(0).astype(int).astype(str).tolist(),
        "primary_keyword": _esc(primary_keyword_vals),
        "primary_keyword_color": _row_colors(primary_keyword_vals, pk_colors),
        "author": _esc(author_bucketed),
        "author_color": _row_colors(author_bucketed, author_colors),
    }
    if has_llm:
        extra_data_dict["math_domain"] = _esc(md_vals)
        extra_data_dict["sequence_type"] = _esc(st_vals)
        extra_data_dict["growth_class"] = _esc(gc_vals)
        extra_data_dict["math_domain_color"] = _row_colors(md_vals, md_colors)
        extra_data_dict["sequence_type_color"] = _row_colors(st_vals, st_colors)
        extra_data_dict["growth_class_color"] = _row_colors(gc_vals, gc_colors)
    extra_data = pd.DataFrame(extra_data_dict)

    # ── Marker sizes (sqrt of edit_count, normalized to [3, 15]) ─────────────
    marker_sizes = _normalize_marker_sizes(df["edit_count"].fillna(1).to_numpy())
    print(f"\nMarker sizes: min={marker_sizes.min():.1f}, max={marker_sizes.max():.1f}")

    # ── Colormaps ────────────────────────────────────────────────────────────
    log_edits = np.log10(df["edit_count"].fillna(1).clip(lower=1).astype(float)).to_numpy()
    log_refs = np.log10(df["n_references"].fillna(0).astype(float) + 1).to_numpy()

    # primary_keyword is always first (and thus DataMapPlot's default).
    # Palette dicts were computed above (shared with tooltip colors).
    colormap_rawdata: list = [primary_keyword_vals, author_bucketed]
    colormap_metadata: list = [
        {
            "field": "primary_keyword",
            "description": "Primary keyword",
            "kind": "categorical",
            "color_mapping": pk_colors,
        },
        {
            "field": "author",
            "description": "Author",
            "kind": "categorical",
            "color_mapping": author_colors,
        },
    ]

    if has_llm:
        colormap_rawdata.extend([md_vals, st_vals, gc_vals, oe_vals])
        colormap_metadata.extend(
            [
                {
                    "field": "math_domain",
                    "description": "Math domain",
                    "kind": "categorical",
                    "color_mapping": md_colors,
                },
                {
                    "field": "sequence_type",
                    "description": "Sequence type",
                    "kind": "categorical",
                    "color_mapping": st_colors,
                },
                {
                    "field": "growth_class",
                    "description": "Growth class",
                    "kind": "categorical",
                    "color_mapping": gc_colors,
                },
                {
                    "field": "origin_era",
                    "description": "Origin era",
                    "kind": "categorical",
                    "color_mapping": oe_colors,
                },
            ]
        )

    # Continuous colormaps always included (work at any scope).
    colormap_rawdata.extend([log_edits, log_refs])
    colormap_metadata.extend(
        [
            {
                "field": "log_edits",
                "description": "Edit count (log10)",
                "kind": "continuous",
                "cmap": "YlOrRd",
            },
            {
                "field": "log_refs",
                "description": "References (log10 + 1)",
                "kind": "continuous",
                "cmap": "BuPu",
            },
        ]
    )

    print(f"  {len(colormap_rawdata)} colormaps registered")

    # ── Topic name vectors (Toponymy labels, coarsest first) ─────────────────
    # labels.parquet stores them as label_layer_0 = coarsest already.
    topic_name_vectors = [labels_df[col].to_numpy() for col in layer_cols]
    print(f"  {len(topic_name_vectors)} Toponymy layer(s) passed to DataMapPlot")

    # ── Hover card styling (editorial: serif name, mono id/values/tags) ──────
    tooltip_css = """
        color: #2c2c2c !important;
        background: rgba(250, 249, 246, 0.94) !important;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 10px;
        backdrop-filter: blur(16px) saturate(1.2);
        -webkit-backdrop-filter: blur(16px) saturate(1.2);
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.10),
            0 2px 8px rgba(0, 0, 0, 0.04),
            inset 0 0 0 1px rgba(255, 255, 255, 0.5);
        max-width: 420px;
        padding: 0 !important;
        overflow: hidden;
    """

    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');

    .oeis-card { display: flex; flex-direction: column; }

    .oeis-head {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 12px;
        padding: 9px 14px 7px;
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
        font-family: 'IBM Plex Mono', ui-monospace, monospace;
    }
    .oeis-id {
        font-size: 12px;
        font-weight: 600;
        color: #1a1a1a;
        letter-spacing: -0.01em;
    }
    .oeis-meta {
        font-size: 10.5px;
        color: #6a6a6a;
        white-space: nowrap;
    }
    .oeis-meta-k {
        color: #a8a8a8;
        text-transform: lowercase;
        letter-spacing: 0.02em;
    }
    .oeis-dot { color: #c8c8c8; margin: 0 4px; }

    .oeis-body { padding: 11px 14px 10px; }

    .oeis-name {
        font-family: 'Newsreader', Georgia, 'Times New Roman', serif;
        font-size: 14.5px;
        line-height: 1.45;
        font-weight: 400;
        color: #2a2a2a;
    }
    .oeis-formula {
        font-family: 'IBM Plex Mono', ui-monospace, monospace;
        font-size: 11px;
        line-height: 1.5;
        color: #7a7a7a;
        margin-top: 8px;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .oeis-formula:empty { display: none; }
    .oeis-values {
        font-family: 'IBM Plex Mono', ui-monospace, monospace;
        font-size: 11.5px;
        line-height: 1.5;
        color: #3a3a3a;
        margin-top: 9px;
        padding: 7px 9px;
        background: rgba(0, 0, 0, 0.028);
        border-radius: 5px;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .oeis-dims {
        display: grid;
        grid-template-columns: auto 1fr;
        column-gap: 14px;
        row-gap: 4px;
        padding: 9px 14px 10px;
        border-top: 1px solid rgba(0, 0, 0, 0.06);
        background: rgba(0, 0, 0, 0.018);
        font-family: 'IBM Plex Mono', ui-monospace, monospace;
    }
    .oeis-dim { display: contents; }
    .oeis-dim:has(.oeis-dim-v:empty) { display: none; }
    .oeis-dim-k {
        font-size: 9px;
        font-weight: 500;
        color: #a8a8a8;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        align-self: center;
        white-space: nowrap;
    }
    .oeis-dim-v {
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.01em;
        align-self: center;
    }
    .oeis-dim-v:empty { display: none; }
    """

    # ── Render ───────────────────────────────────────────────────────────────
    print("\nBuilding DataMapPlot…")
    fig = datamapplot.create_interactive_plot(
        coords,
        *topic_name_vectors,
        hover_text=hover_text,
        hover_text_html_template=hover_html,
        marker_size_array=marker_sizes,
        extra_point_data=extra_data,
        on_click="window.open(`https://oeis.org/{id}`,'_blank')",
        colormap_rawdata=colormap_rawdata,
        colormap_metadata=colormap_metadata,
        title="A Semantic Map of the OEIS",
        sub_title=f"{len(df)} core integer sequences, mapped by description similarity",
        enable_search=True,
        font_family="IBM Plex Sans",
        custom_css=custom_css,
        tooltip_css=tooltip_css,
        darkmode=False,
    )

    print(f"\nWriting {OEIS_MAP_HTML.name}…")
    fig.save(str(OEIS_MAP_HTML))
    size_kb = OEIS_MAP_HTML.stat().st_size / 1_000
    print(f"  → {OEIS_MAP_HTML} ({size_kb:.0f} KB)")

    # Publish under scope-specific name: curated is the headline map at
    # docs/index.html; full scope ships alongside at docs/full.html. core
    # scope is a local smoke-test and also publishes to index.html so
    # `make v0` still produces a viewable artifact for development.
    docs_out = DOCS_FULL_HTML if SCOPE == "all" else DOCS_INDEX_HTML
    print(f"Copying to {docs_out.relative_to(docs_out.parent.parent)}…")
    shutil.copy2(OEIS_MAP_HTML, docs_out)
    print(f"  → {docs_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
