"""Pairwise features for the matching model.

Two families, computed in different places:

* **Text features** (``text_features``) - per pair, from the two records'
  strings. The hot loop; ``pipeline.build`` runs it in worker processes.
    - *Similarity*: how alike are the two records (token, character, fuzzy,
      phonetic skeleton across scripts).
    - *Conflict*: evidence they are *different*: disagreeing house numbers,
      extra core words. Error analysis showed the dominant false positive is
      "same building, different unit" (Flat 503 vs Flat 508), which pure
      similarity features cannot express.
* **Context features** (``context_features``) - vectorised over the whole
  blocking output: how the candidate ranks inside its Source-1 entity's list,
  and how contested it is by *other* Source-1 entities. Ground truth gives every
  Source-2/3 record to exactly one Source-1 entity, so contention is
  informative. Needs blocking run over *all* Source-1 records to be faithful.

``FEATURE_NAMES = TEXT_FEATURE_NAMES + CONTEXT_FEATURE_NAMES`` is the column
order of every feature matrix; both lists must stay aligned with the functions
below (``pipeline.build`` asserts the widths).
"""
from __future__ import annotations

import numpy as np

import ertext

try:  # optional, much faster and more accurate than the fallback
    from rapidfuzz import fuzz

    def _ratio(a, b):
        return fuzz.ratio(a, b) / 100.0

    def _token_set(a, b):
        return fuzz.token_set_ratio(a, b) / 100.0

    def _partial(a, b):
        return fuzz.partial_ratio(a, b) / 100.0

    HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    from difflib import SequenceMatcher

    def _ratio(a, b):
        return SequenceMatcher(None, a[:80], b[:80]).ratio()

    def _token_set(a, b):
        return SequenceMatcher(None, " ".join(sorted(a.split()))[:80],
                               " ".join(sorted(b.split()))[:80]).ratio()

    def _partial(a, b):
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        return SequenceMatcher(None, short[:60], long[:60]).ratio()

    HAVE_RAPIDFUZZ = False

TEXT_FEATURE_NAMES = [
    # similarity - name
    "name_core_jacc", "name_core_cont", "name_all_jacc", "name_tri_jacc",
    "name_sorted_eq", "name_initials_eq", "name_len_ratio", "name_core_shared",
    "name_ratio", "name_token_set", "name_partial",
    # similarity - name, script-independent (romanised consonant skeletons) and
    # web-ified ('womenshealthallied.com' vs 'Womens Health Allied')
    "name_skel_jacc", "name_skel_cont", "name_skel_ratio", "name_compact_partial",
    # similarity - address
    "addr_jacc", "addr_cont", "addr_tri_jacc", "addr_token_set",
    "addr_num_cont", "addr_num_any", "addr_skel_cont",
    # conflict
    "num_conflict", "extra_core", "missing_core", "core_disjoint",
    # missingness / script
    "addr_empty_cand", "addr_empty_s1", "cand_script_indic", "script_mismatch",
    # interactions
    "name_only_evidence", "addr_only_evidence",
]

CONTEXT_FEATURE_NAMES = [
    "block_score", "n_shared", "rank", "score_ratio_top", "score_gap_next",
    "n_candidates", "is_s3", "target_best_ratio", "target_n_claims",
    # reverse index: where does this Source-1 entity rank among everyone that
    # retrieved the same target, and by how much does it beat the runner-up?
    "rev_rank", "target_margin",
    # name channel of blocking (name tokens, concatenation keys, skeletons)
    "name_block_score", "name_block_shared", "name_block_rank",
    "name_block_ratio_top",
]

FEATURE_NAMES = TEXT_FEATURE_NAMES + CONTEXT_FEATURE_NAMES


class RecordView:
    """Pre-computed token views of a record (built once, reused per pair).

    ``amap`` is the mined alias lexicon; applying it here means Devanagari and
    Latin spellings of the same word become the same token before any
    similarity is computed.
    """

    __slots__ = ("name", "addr", "ntok", "core", "atok", "num", "ntri", "atri",
                 "script", "nstr", "astr", "skel", "skstr", "compact", "askel")

    def __init__(self, name, addr, amap=None):
        self.name = name or ""
        self.addr = addr or ""
        core_seq = ertext.canonical(ertext.core_name_tokens(self.name), amap)
        self.ntok = set(ertext.canonical(ertext.name_tokens(self.name), amap))
        self.core = set(core_seq)
        self.atok = set(ertext.canonical(ertext.addr_tokens(self.addr), amap))
        self.num = ertext.numeric_tokens(self.atok)
        self.ntri = ertext.char_ngrams(self.name, 3)
        self.atri = ertext.char_ngrams(self.addr, 3)
        self.script = ertext.script_of(self.name)
        # canonicalised strings, so fuzzy ratios also benefit from the lexicon
        self.nstr = " ".join(sorted(self.ntok))
        self.astr = " ".join(sorted(self.atok))
        sk = [ertext.skeleton(t) for t in core_seq]
        self.skel = {s for s in sk if len(s) >= 2}
        self.skstr = " ".join(sorted(self.skel))
        self.compact = "".join(ertext.romanize(t) for t in core_seq)
        self.askel = {s for s in (ertext.skeleton(t) for t in self.atok
                                  if not t.isdigit()) if len(s) >= 2}


def initials(core):
    return "".join(sorted(t[0] for t in core if t))


def text_features(a: RecordView, b: RecordView):
    name_core_cont = ertext.containment(a.core, b.core)
    addr_cont = ertext.containment(a.atok, b.atok)
    la, lb = len(a.name), len(b.name)

    return [
        # ---- name similarity ----
        ertext.jaccard(a.core, b.core),
        name_core_cont,
        ertext.jaccard(a.ntok, b.ntok),
        ertext.jaccard(a.ntri, b.ntri),
        1.0 if (a.core and a.core == b.core) else 0.0,
        1.0 if (a.core and b.core and initials(a.core) == initials(b.core)) else 0.0,
        min(la, lb) / max(la, lb, 1),
        float(len(a.core & b.core)),
        _ratio(a.nstr, b.nstr),
        _token_set(a.nstr, b.nstr),
        _partial(a.nstr, b.nstr),
        # ---- name similarity, script-independent ----
        ertext.jaccard(a.skel, b.skel),
        ertext.containment(a.skel, b.skel),
        _ratio(a.skstr, b.skstr) if (a.skstr and b.skstr) else 0.0,
        _partial(a.compact, b.compact) if (a.compact and b.compact) else 0.0,
        # ---- address similarity ----
        ertext.jaccard(a.atok, b.atok),
        addr_cont,
        ertext.jaccard(a.atri, b.atri),
        _token_set(a.astr, b.astr),
        ertext.containment(a.num, b.num),
        1.0 if (a.num & b.num) else 0.0,
        ertext.containment(a.askel, b.askel),
        # ---- conflict ----
        # both sides carry house/plot numbers and none of them agree:
        # the "Flat 503 vs Flat 508 in the same tower" false positive.
        1.0 if (a.num and b.num and not (a.num & b.num)) else 0.0,
        float(len(b.core - a.core)),
        float(len(a.core - b.core)),
        1.0 if (a.core and b.core and not (a.core & b.core)) else 0.0,
        # ---- missingness / script ----
        1.0 if not b.atok else 0.0,
        1.0 if not a.atok else 0.0,
        1.0 if b.script == "indic" else 0.0,
        1.0 if (a.script != b.script and "none" not in (a.script, b.script)) else 0.0,
        # ---- interactions ----
        1.0 if (name_core_cont >= 0.8 and addr_cont < 0.2) else 0.0,
        1.0 if (addr_cont >= 0.8 and name_core_cont < 0.2) else 0.0,
    ]


class ContextBuilder:
    """Context features for the rows of one blocking sidecar part.

    ``sc`` holds the part's columns (``s1, cand, score, ...``); rows of one
    Source-1 entity are contiguous and rank-ordered, which is how
    ``blocking.py`` writes them. ``meta`` is ``blocking.load_meta``: per-target
    best / second-best score and claim count, accumulated by blocking over
    *every* Source-1 query - so contention is faithful even when only a sample
    of entities was stored, and no global sort over 100M+ pairs is needed.
    """

    def __init__(self, sc, meta):
        self.sc = sc
        s1, score = sc["s1"], sc["score"]
        n = len(s1)
        self.n = n
        # ---- within each source-1 entity's list ----------------------------
        self.start = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]]) if n else \
            np.zeros(0, np.int64)
        self.size = np.diff(np.r_[self.start, n])
        self.grp = np.repeat(np.arange(len(self.start), dtype=np.int32), self.size)
        self.top = score[self.start] if n else np.zeros(0, np.float32)
        self.ntop = (np.maximum.reduceat(sc["nscore"], self.start)
                     if "nscore" in sc and n else None)
        # ---- across entities: from blocking's streaming statistics ---------
        t = np.searchsorted(meta["t_codes"], sc["cand"])
        self.tbest = meta["t_best"][t]
        self.tsecond = meta["t_second"][t]
        self.tclaims = meta["t_claims"][t]

    def features(self, rows):
        """float32 (len(rows), len(CONTEXT_FEATURE_NAMES)) for part rows."""
        sc = self.sc
        rows = np.asarray(rows)
        score = sc["score"][rows].astype(np.float32)
        g = self.grp[rows]
        top = self.top[g]
        # next score in the same entity's list (rows are rank-ordered)
        nxt_row = np.minimum(rows + 1, self.n - 1)
        same = (nxt_row != rows) & (self.grp[nxt_row] == g)
        nxt = np.where(same, sc["score"][nxt_row], 0.0)
        best = self.tbest[rows]
        second = self.tsecond[rows]
        eps = 1e-4
        # 0 = best claimant, 1 = runner-up, 2 = further down
        rev = np.where(score >= best - eps, 0, np.where(score >= second - eps, 1, 2))
        best_other = np.where(rev == 0, second, best)

        out = np.empty((len(rows), len(CONTEXT_FEATURE_NAMES)), np.float32)
        out[:, 0] = score
        out[:, 1] = sc["shared"][rows]
        out[:, 2] = sc["rank"][rows]
        out[:, 3] = score / np.maximum(top, 1e-6)
        out[:, 4] = score - nxt
        out[:, 5] = self.size[g]
        out[:, 6] = sc["cand"][rows] >= 3_000_000_000   # blocking.encode_id: S3-
        out[:, 7] = score / np.maximum(best, 1e-6)
        out[:, 8] = self.tclaims[rows]
        out[:, 9] = rev
        # >1: this entity beats every other claimant; capped when unopposed
        out[:, 10] = np.minimum(score / np.maximum(best_other, 1e-6), 10.0)
        if self.ntop is not None:
            nsc = sc["nscore"][rows].astype(np.float32)
            out[:, 11] = nsc
            out[:, 12] = sc["nshared"][rows]
            out[:, 13] = sc["nrank"][rows]
            out[:, 14] = nsc / np.maximum(self.ntop[g], 1e-6)
        else:  # single-channel sidecar
            out[:, 11:15] = 0.0
        return out
