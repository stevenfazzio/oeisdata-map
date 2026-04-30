"""Classification of OEIS keywords by role.

OEIS keywords serve two distinct purposes:

- **Content keywords** describe what the sequence IS — its mathematical
  character (e.g., ``tabl`` = table by rows, ``mult`` = multiplicative
  function, ``cons`` = constant digits). These are useful as an *independent*
  signal for evaluating the embedding geometry: if two sequences share a
  content keyword, are they close in embedding space? Because of that role,
  content keywords MUST NOT appear in the embedding-input text — otherwise
  the eval becomes circular (we'd be measuring whether the encoder copied
  the literal token rather than whether it captured the mathematical
  property from prose / formulas / examples).

- **Editorial keywords** describe how editors feel about the sequence
  (e.g., ``nice``, ``easy``, ``hard``). They're meta-judgments rather than
  content, and they're partially co-determined with the ``%Y`` graph
  (the same editors curate both). They're not useful as an orthogonal
  eval signal, and including them in the embed text doesn't pollute the
  content-keyword eval.

The split is based on the OEIS keyword glossary plus the actual frequency
distribution in the curated 25k:

- ``nonn`` (92.7% of curated rows) is technically content (nonnegative) but
  so universal it carries near-zero discriminative signal. Dropped from the
  embed text and excluded from eval (would saturate intra-class similarity).

- ``nice`` (26.8%) and ``core`` (0.73%) are editorial AND part of the curated
  seed, so they co-determine corpus membership. Excluded from eval. Kept in
  the embed text since they reflect editorial attention.

- ``easy`` / ``hard`` (47.9% / 2.9%) are editorial. Kept in embed text.

- Boundary content keywords ``fini``, ``full``, ``hear``, ``word`` are very
  low-frequency (<1%) — fine to use as eval but the silhouette numbers will
  be noisy due to small sample sizes. We include them anyway and let the
  reader weight by support.
"""

from __future__ import annotations

# Content keywords: mathematical character. Drop from embed text. Use as eval.
# Frequencies in curated 25k (April 2026):
#   tabl=15.1% tabf=5.9% base=7.7% sign=7.3% cons=3.2% frac=2.7% look=2.5%
#   mult=2.1% walk=0.6% cofr=0.3% eigen=0.3% fini=0.8% full=0.6% hear=0.3%
#   word=0.1%
CONTENT_KEYWORDS: frozenset[str] = frozenset(
    {
        "tabl",  # table read by rows (Pascal-like)
        "tabf",  # table read flat (irregular triangle)
        "base",  # base-dependent / digit-related
        "sign",  # signed entries (alternating, negatives)
        "cons",  # constant digits (decimal expansion)
        "frac",  # fraction expansion
        "cofr",  # continued fraction expansion
        "look",  # graphical / visual pattern when plotted
        "hear",  # auditory pattern when sonified
        "mult",  # multiplicative function (number-theoretic)
        "walk",  # walks / lattice paths
        "eigen",  # eigenvalue spectrum
        "fini",  # finitely many terms
        "full",  # complete sequence shown
        "word",  # word-related (DNA, abstract symbols)
    }
)

# Universal content keyword (~93% of curated set) — content in principle but
# near-zero discriminative signal. Dropped from embed text and excluded from
# eval to avoid saturating intra-class similarity.
UNIVERSAL_CONTENT_KEYWORDS: frozenset[str] = frozenset({"nonn"})

# Editorial keywords: judgments, recency, curation. Kept in embed text. Not
# used as eval signal (correlated with curator attention and %Y graph).
EDITORIAL_KEYWORDS: frozenset[str] = frozenset(
    {
        "core",  # foundational sequences (curated seed)
        "nice",  # exceptionally good entries (curated seed)
        "easy",  # easy to compute
        "hard",  # hard to compute
        "more",  # editor wants more terms
        "less",  # less interesting (rare)
        "obsc",  # obscure
        "changed",  # recently changed
        "new",  # recently added
    }
)

# Hard-exclude keywords: never in any scope.
EXCLUDE_KEYWORDS: frozenset[str] = frozenset({"dead", "dupe", "uned", "dumb"})

# Sanity: the three sets are disjoint.
assert not (CONTENT_KEYWORDS & UNIVERSAL_CONTENT_KEYWORDS), "overlap content/universal"
assert not (CONTENT_KEYWORDS & EDITORIAL_KEYWORDS), "overlap content/editorial"
assert not (CONTENT_KEYWORDS & EXCLUDE_KEYWORDS), "overlap content/exclude"
assert not (EDITORIAL_KEYWORDS & EXCLUDE_KEYWORDS), "overlap editorial/exclude"

# Convenience: every keyword that should be REMOVED from build_embed_text.
EMBED_TEXT_DROP: frozenset[str] = CONTENT_KEYWORDS | UNIVERSAL_CONTENT_KEYWORDS | EXCLUDE_KEYWORDS
