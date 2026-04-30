"""Render the OEIS interactive map with DataMapPlot.

Colormap dropdown + hover + click-through + Advanced Filters panel
(categorical multi-selects + numeric range sliders, with URL state).

Reads:
- ``data/enriched.parquet``     — id, name, comments, values_preview_str,
                                  edit_count, n_references, n_links,
                                  last_edited, keywords list, and optionally
                                  math_domain / sequence_type / growth_class /
                                  origin_era (from stage 03).
- ``data/umap_coords.npz``      — float32 (N, 2) under key ``coords``
- ``data/labels.parquet``       — id + label_layer_*

Writes:
- ``data/oeis_map.html``        — local preview
- ``docs/index.html``           — GitHub Pages entry point (identical copy)

**LLM-column handling:** if ``enriched.parquet`` is missing any of the 4
stage 03 LLM columns (math_domain / sequence_type / growth_class /
origin_era), the map still renders — just with fewer colormaps and the
filter panel skips those four category sections. This lets the
``OEIS_SKIP_ENRICH=1`` pathway work at any scope without crashing.

Design decisions:
- Marker size: ``sqrt(edit_count)`` linearly scaled to [3, 15] px
  → Fibonacci (max edit_count) is the largest dot
- Colormap order: Toponymy clusters (auto-default) → math_domain →
  sequence_type → growth_class → origin_era → author → log10(edit_count)
  → log10(n_references + 1).
- The four LLM enums require stage 03; without enrichment the menu
  collapses to author + the two continuous colormaps.
- Click handler: opens https://oeis.org/{id} in a new tab.
- Filter panel: legend-clicks on categorical colormaps (Domain / Type /
  Growth / Era / Author) sync with the corresponding filter checkboxes.
  Range sliders cap at p99 to keep the meaningful range visible despite
  long tails (e.g., max edit_count is 2,514 but p99 is ~282).
"""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

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

FILTER_PANEL_HTML = Path(__file__).resolve().parent / "filter_panel.html"

# ── Helpers ──────────────────────────────────────────────────────────────────

# The 4 stage-03 LLM columns. If any are missing, stage 07 falls back to
# author + continuous colormaps only.
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


def _format_keywords(kws) -> str:
    """Comma-separated keyword list for the hover card.

    Returns "" when the list is missing/empty so the row can collapse via
    the `:has(.oeis-dim-v:empty)` rule.
    """
    if kws is None:
        return ""
    try:
        items = [str(k) for k in kws if k]
    except TypeError:
        return ""
    return ", ".join(items)


def _pill_bg(hex_color: str, alpha: float = 0.18) -> str:
    """Soft-tinted background for the Domain pill: input hex → `rgba(r,g,b,α)`."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(90, 90, 90, {alpha})"
    try:
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return f"rgba(90, 90, 90, {alpha})"
    return f"rgba({r}, {g}, {b}, {alpha})"


def _darken_for_pill(hex_color: str, factor: float = 0.4) -> str:
    """Darken any category color to legible pill text on a same-hue tinted bg.

    `_darken_for_text` only darkens above a luminance threshold, which leaves
    mid-luminance hues (e.g. saturated orange) untouched and unreadable on
    the pill's faded same-hue background. Pills always darken aggressively.
    """
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "#3a3a3a"
    try:
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return "#3a3a3a"
    r, g, b = (int(round(c * factor)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Site-nav injection (Visualization ↔ About) ──────────────────────────────
# Injected only into the GitHub Pages copies (docs/index.html, docs/full.html),
# not into data/oeis_map.html — about.html lives alongside the docs/ pages and
# the local artifact is single-page anyway. Keeping the nav out of the
# DataMapPlot template means future DataMapPlot upgrades don't have to be
# audited for nav-template breakage.

# Plain string (not str.format) — CSS braces don't survive format placeholders.
_SITE_NAV_HTML = """\
<style>
.site-nav{position:fixed;top:0;left:0;right:0;z-index:200;
  background:rgba(255,255,255,0.85);backdrop-filter:blur(8px);
  -webkit-backdrop-filter:blur(8px);border-bottom:1px solid #e0e0e0;
  padding:0 24px;height:44px;display:flex;align-items:center;gap:24px;
  font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:14px;font-weight:500;pointer-events:auto;}
.site-nav a{color:#333;text-decoration:none;transition:color 0.15s;}
.site-nav a:hover{color:#0d9488;}
.site-nav a.active{color:#0d9488;border-bottom:2px solid #0d9488;line-height:42px;}
/* Push the DataMapPlot top corners below the nav bar so the title/search
   (top-left) and the colormap legend (top-right) don't get clipped by the
   translucent strip. Bottom corners are unaffected. */
.stack.top-left,.stack.top-right{margin-top:44px;}
</style>
<nav class="site-nav">
  <a href="index.html"__VIS_ACTIVE__>Visualization</a>
  <a href="about.html"__ABOUT_ACTIVE__>About</a>
</nav>
"""


def _inject_site_nav(html: str, active: str) -> str:
    """Insert the Visualization/About nav block immediately after <body>.

    `active` is "vis" or "about" — chooses which link gets the .active class.
    No-op (with a warning) if <body> is not found in the rendered HTML, since
    DataMapPlot output is expected to contain it.
    """
    nav = _SITE_NAV_HTML.replace("__VIS_ACTIVE__", ' class="active"' if active == "vis" else "").replace(
        "__ABOUT_ACTIVE__", ' class="active"' if active == "about" else ""
    )
    new_html, n_subs = re.subn(r"<body>", "<body>" + nav, html, count=1)
    if n_subs == 0:
        print("  WARNING: <body> tag not found; site-nav not injected")
        return html
    return new_html


def _publish_with_nav(src_html_path, dest_html_path) -> None:
    """Copy the rendered map to the docs/ output and inject the site-nav."""
    html = src_html_path.read_text(encoding="utf-8")
    html = _inject_site_nav(html, active="vis")
    tmp = dest_html_path.with_suffix(dest_html_path.suffix + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(dest_html_path)


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
        print(f"  LLM enum columns missing ({', '.join(missing)}); falling back to author + continuous only")

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
    # body (name + optional formula + mono value run) → mono dim-row footer.
    # The dim section orders the four LLM enums first (Domain rendered as a
    # colored pill, the rest as plain text), then Author and Keywords.
    # `.oeis-dim:has(.oeis-dim-v:empty)` collapses rows with missing values.
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
    _author_dim = (
        '    <div class="oeis-dim">'
        '      <span class="oeis-dim-k">Author</span>'
        '      <span class="oeis-dim-v">{author}</span>'
        "    </div>"
    )
    _keywords_dim = (
        '    <div class="oeis-dim">'
        '      <span class="oeis-dim-k">Keywords</span>'
        '      <span class="oeis-dim-v oeis-dim-v-kw">{keywords}</span>'
        "    </div>"
    )
    if has_llm:
        hover_html = (
            _hover_head + '  <div class="oeis-dims">'
            '    <div class="oeis-dim">'
            '      <span class="oeis-dim-k">Domain</span>'
            '      <span class="oeis-dim-v"><span class="oeis-pill"'
            '            style="background:{math_domain_bg};color:{math_domain_color}">'
            "{math_domain}</span></span>"
            "    </div>"
            '    <div class="oeis-dim">'
            '      <span class="oeis-dim-k">Type</span>'
            '      <span class="oeis-dim-v">{sequence_type}</span>'
            "    </div>"
            '    <div class="oeis-dim">'
            '      <span class="oeis-dim-k">Growth</span>'
            '      <span class="oeis-dim-v">{growth_class}</span>'
            "    </div>"
            '    <div class="oeis-dim">'
            '      <span class="oeis-dim-k">Era</span>'
            '      <span class="oeis-dim-v">{origin_era}</span>'
            "    </div>" + _author_dim + _keywords_dim + "  </div>"
            "</div>"
        )
    else:
        hover_html = _hover_head + '  <div class="oeis-dims">' + _author_dim + _keywords_dim + "  </div></div>"

    # Palette dicts: one per categorical colormap. Reused below when building
    # colormap_metadata so the legend colors match the map dots.
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

    keywords_str = [_format_keywords(kws) for kws in df["keywords"]]

    # Filter-panel arrays. The `*_filter` columns hold plain (un-escaped) values
    # that the JS reads from `hoverData[field]` to drive checkbox/range filters
    # and to keep legend-clicks in sync with checkbox state. Hover-card columns
    # remain HTML-escaped above; the filter columns parallel them.
    edits_int = df["edit_count"].fillna(0).astype(int).to_numpy()
    refs_int = df["n_references"].fillna(0).astype(int).to_numpy()
    links_int = df["n_links"].fillna(0).astype(int).to_numpy()
    # Null last_edited (~4% of curated rows) → year 0, which falls below any
    # slider range. Default state is "no filter applied" so nulls are visible
    # initially; touching the year slider excludes them, which matches the
    # semantic of "I want sequences edited in window X" (we don't know for nulls).
    year_int = pd.to_datetime(df["last_edited"], errors="coerce").dt.year.fillna(0).astype(int).to_numpy()
    keyword_universe = sorted({str(k) for kws in df["keywords"] if kws is not None for k in kws if k})
    keywords_pipe = ["|".join(str(k) for k in (kws if kws is not None else []) if k) for kws in df["keywords"]]

    extra_data_dict = {
        "id": _esc(df["id"]),
        "name": _esc(df["name"]),
        "first_formula": _esc([_first_formula(f) for f in df["formulas"]]),
        "values_preview_str": _esc(df["values_preview_str"]),
        "edit_count": df["edit_count"].fillna(0).astype(int).astype(str).tolist(),
        "n_references": df["n_references"].fillna(0).astype(int).astype(str).tolist(),
        "author": _esc(author_clean_vals),
        "keywords": _esc(keywords_str),
        # Plain-text filter columns (JS Number()-parses range columns).
        "author_filter": author_bucketed.tolist(),
        "keywords_filter": keywords_pipe,
        "edits_filter": edits_int.astype(str).tolist(),
        "refs_filter": refs_int.astype(str).tolist(),
        "links_filter": links_int.astype(str).tolist(),
        "year_filter": year_int.astype(str).tolist(),
    }
    if has_llm:
        extra_data_dict["math_domain"] = _esc(md_vals)
        extra_data_dict["sequence_type"] = _esc(st_vals)
        extra_data_dict["growth_class"] = _esc(gc_vals)
        extra_data_dict["origin_era"] = _esc(oe_vals)
        extra_data_dict["math_domain_color"] = [
            _darken_for_pill(md_colors.get(v or "unknown", "#5a5a5a")) for v in md_vals
        ]
        extra_data_dict["math_domain_bg"] = [_pill_bg(md_colors.get(v or "unknown", "#5a5a5a")) for v in md_vals]
        # Plain-text LLM-enum filter columns (parallel the colormap rawdata).
        extra_data_dict["domain_filter"] = md_vals.tolist()
        extra_data_dict["type_filter"] = st_vals.tolist()
        extra_data_dict["growth_filter"] = gc_vals.tolist()
        extra_data_dict["era_filter"] = oe_vals.tolist()
    extra_data = pd.DataFrame(extra_data_dict)

    # ── Marker sizes (sqrt of edit_count, normalized to [3, 15]) ─────────────
    marker_sizes = _normalize_marker_sizes(df["edit_count"].fillna(1).to_numpy())
    print(f"\nMarker sizes: min={marker_sizes.min():.1f}, max={marker_sizes.max():.1f}")

    # ── Colormaps ────────────────────────────────────────────────────────────
    log_edits = np.log10(df["edit_count"].fillna(1).clip(lower=1).astype(float)).to_numpy()
    log_refs = np.log10(df["n_references"].fillna(0).astype(float) + 1).to_numpy()

    # Order: LLM enums (Domain, Type, Growth, Era) → Author → continuous
    # (Edits, Refs). Toponymy clusters auto-show as the dropdown's default
    # via the topic_name_vectors below.
    colormap_rawdata: list = []
    colormap_metadata: list = []

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

    colormap_rawdata.append(author_bucketed)
    colormap_metadata.append(
        {
            "field": "author",
            "description": "Author",
            "kind": "categorical",
            "color_mapping": author_colors,
        }
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
        color: #3a3a3a;
        letter-spacing: 0.01em;
        align-self: center;
    }
    .oeis-dim-v:empty { display: none; }
    .oeis-pill {
        display: inline-block;
        padding: 1px 9px 2px;
        margin-left: -9px;
        border-radius: 10px;
        font-size: 10.5px;
        font-weight: 500;
        letter-spacing: 0.01em;
        line-height: 1.45;
    }
    .oeis-dim-v-kw {
        color: #6a6a6a;
        font-size: 10.5px;
        font-weight: 400;
        letter-spacing: 0;
        word-spacing: 0.02em;
    }
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
        sub_title=f"{len(df):,} integer sequences ({SCOPE} scope), mapped by description similarity",
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

    print("Injecting Advanced Filters panel…")
    filter_config = _build_filter_config(
        n_rows=len(df),
        has_llm=has_llm,
        md_vals=md_vals if has_llm else None,
        st_vals=st_vals if has_llm else None,
        gc_vals=gc_vals if has_llm else None,
        oe_vals=oe_vals if has_llm else None,
        author_bucketed=author_bucketed,
        keyword_universe=keyword_universe,
        edits_int=edits_int,
        refs_int=refs_int,
        links_int=links_int,
        year_int=year_int,
    )
    _inject_filters(OEIS_MAP_HTML, filter_config)
    size_kb = OEIS_MAP_HTML.stat().st_size / 1_000
    print(f"  → {OEIS_MAP_HTML} ({size_kb:.0f} KB after injection)")

    # Publish under scope-specific name: curated is the headline map at
    # docs/index.html; full scope ships alongside at docs/full.html. core
    # scope is a local smoke-test and also publishes to index.html so
    # `make v0` still produces a viewable artifact for development.
    docs_out = DOCS_FULL_HTML if SCOPE == "all" else DOCS_INDEX_HTML
    print(f"Publishing to {docs_out.relative_to(docs_out.parent.parent)} with site-nav…")
    _publish_with_nav(OEIS_MAP_HTML, docs_out)
    print(f"  → {docs_out}")

    print("\nDone.")


# ── Advanced Filters injection ──────────────────────────────────────────────

# Tail labels are pinned to the end of the checkbox list so "unknown" / "Other"
# don't visually dominate the alphabetic head.
_TAIL_LABELS = ("unknown", "Unknown", "other", "Other")


def _sorted_with_tail(values) -> list[str]:
    """Unique values, alphabetic, with tail labels (unknown/Other) pinned last."""
    uniques = sorted({str(v) for v in values})
    head = [v for v in uniques if v not in _TAIL_LABELS]
    tail = [v for v in uniques if v in _TAIL_LABELS]
    tail.sort(key=lambda v: _TAIL_LABELS.index(v))
    return head + tail


def _p99_cap(arr: np.ndarray) -> int:
    """99th-percentile cap for slider max so the long tail doesn't compress
    the meaningful range. Values above the cap fold into the '+' indicator."""
    return int(np.percentile(arr, 99))


def _build_filter_config(
    *,
    n_rows: int,
    has_llm: bool,
    md_vals,
    st_vals,
    gc_vals,
    oe_vals,
    author_bucketed: np.ndarray,
    keyword_universe: list[str],
    edits_int: np.ndarray,
    refs_int: np.ndarray,
    links_int: np.ndarray,
    year_int: np.ndarray,
) -> dict:
    """Build the JSON config consumed by filter_panel.html JS."""
    # Year minimum: ignore the 0 sentinel for missing dates, otherwise the
    # slider would start at 0 and squash the meaningful 2000-onwards range.
    nonzero_years = year_int[year_int > 0]
    year_min = int(nonzero_years.min()) if nonzero_years.size else 2000
    year_max = int(year_int.max()) if year_int.size else 2026

    cfg: dict = {
        "totalCount": int(n_rows),
        "authors": _sorted_with_tail(author_bucketed),
        "keywords": list(keyword_universe),
        "domains": [],
        "types": [],
        "growths": [],
        "eras": [],
        "ranges": {
            "edits": {
                "min": int(edits_int.min()),
                "max": int(edits_int.max()),
                "sliderMax": _p99_cap(edits_int),
            },
            "refs": {
                "min": int(refs_int.min()),
                "max": int(refs_int.max()),
                "sliderMax": _p99_cap(refs_int),
            },
            "links": {
                "min": int(links_int.min()),
                "max": int(links_int.max()),
                "sliderMax": _p99_cap(links_int),
            },
            "year": {
                "min": year_min,
                "max": year_max,
                "sliderMax": year_max,
            },
        },
        "colormapFieldToFilterId": {
            "author": "filter-author",
        },
        "filterIdToColormapField": {
            "filter-author": "author",
        },
    }

    if has_llm:
        cfg["domains"] = _sorted_with_tail(md_vals)
        cfg["types"] = _sorted_with_tail(st_vals)
        cfg["growths"] = _sorted_with_tail(gc_vals)
        cfg["eras"] = _sorted_with_tail(oe_vals)
        cfg["colormapFieldToFilterId"].update(
            {
                "math_domain": "filter-domain",
                "sequence_type": "filter-type",
                "growth_class": "filter-growth",
                "origin_era": "filter-era",
            }
        )
        cfg["filterIdToColormapField"].update(
            {
                "filter-domain": "math_domain",
                "filter-type": "sequence_type",
                "filter-growth": "growth_class",
                "filter-era": "origin_era",
            }
        )

    return cfg


def _inject_filters(html_path: Path, filter_config: dict) -> None:
    """Splice the Advanced Filters CSS / HTML / JS into a DataMapPlot output.

    Steps mirror the sister ``huggingface-dataset-map`` injection so future
    DataMapPlot upgrades only need to be tested in one place:
      1. Dispatch a ``datamapReady`` event after metadata loads, so the JS
         can attach to the live ``datamap`` and ``hoverData`` references.
      2. Splice the panel's CSS before ``</head>``.
      3. Splice the panel's HTML next to the search box (top-left stack).
      4. Splice the panel's JS before ``</html>``, with the live filter
         config injected as JSON.
    """
    html = html_path.read_text(encoding="utf-8")

    # 1. Dispatch datamapReady once metadata is loaded, before checkAllDataLoaded.
    html = re.sub(
        r"(updateProgressBar\('meta-data-progress', 100\);\s*)(checkAllDataLoaded\(\);)",
        r"\1window.dispatchEvent(new CustomEvent('datamapReady', "
        r"{ detail: { datamap, hoverData } }));\n          \2",
        html,
        count=1,
    )

    # 2. Read the panel template and split by section markers.
    template = FILTER_PANEL_HTML.read_text(encoding="utf-8")
    sections = re.split(r"<!-- SECTION: (\w+) -->", template)
    section_map = {}
    for i in range(1, len(sections), 2):
        section_map[sections[i]] = sections[i + 1].strip()

    # 3. Inject CSS before </head>.
    html = html.replace("</head>", section_map["css"] + "\n</head>", 1)

    # 4. Splice HTML in after the search-container div (matches HF map layout).
    search_pattern = re.compile(
        r'(<div id="search-container" class="container-box[^"]*">\s*'
        r"<input[^/]*/>\s*</div>)"
    )
    match = search_pattern.search(html)
    if match:
        insert_pos = match.end()
        html = html[:insert_pos] + "\n      " + section_map["html"] + "\n" + html[insert_pos:]
    else:
        # Fallback: inside .stack.top-left, before any other content.
        html = html.replace(
            '<div class="stack top-left">',
            '<div class="stack top-left">\n      ' + section_map["html"],
            1,
        )

    # 5. Inject JS (with config) before </html>.
    js_section = section_map["js"].replace("__FILTER_CONFIG_JSON__", json.dumps(filter_config))
    html = html.replace("</html>", js_section + "\n</html>", 1)

    html_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
