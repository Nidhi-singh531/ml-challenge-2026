"""Extra pair features, kept apart from the frozen 47-feature schema.

* Number evidence: ``num_conflict`` fires only when NO number is shared, so a
  shared building or postcode hides a different flat number, and alphanumeric
  numbers ('600a') are ignored entirely.
* Cross-script core names: ``core_name_tokens`` drops legal suffixes *before*
  alias mapping, so an Indic 'प्राइवेट' becomes 'private' and stays in the core
  while the English 'Private' is dropped. Here the core is taken after mapping.
"""
from __future__ import annotations

import re

import ertext
from features import RecordView

NUM_NAMES = ["num_exact_shared", "num_a_only", "num_b_only", "unit_conflict",
             "alnum_match", "alnum_conflict"]
CORE2_NAMES = ["core2_jacc", "core2_cont", "core2_extra", "core2_missing", "core2_disjoint"]


def _tolerant(t):
    s = t.lstrip("0") or "0"
    return {s, t[1:].lstrip("0") or "0"} if len(t) > 3 else {s}


def _numbers(v: RecordView):
    """Digit runs of every token with a digit ('600a' -> '600'), with tolerant forms."""
    raw = {m for t in v.atok for m in re.findall(r"\d+", t)}
    tol = set().union(*(_tolerant(t) for t in raw)) if raw else set()
    return raw, tol


def number_evidence(a: RecordView, b: RecordView):
    if not a.atok or not b.atok:
        return [0.0] * len(NUM_NAMES)
    da, ta = _numbers(a)
    db, tb = _numbers(b)
    exact = len({t.lstrip("0") for t in da} & {t.lstrip("0") for t in db})
    a_only = sum(1 for t in da if not (_tolerant(t) & tb))
    b_only = sum(1 for t in db if not (_tolerant(t) & ta))
    shared = bool(ta & tb)
    unit = 1.0 if (shared and a_only and b_only) else 0.0
    aa = {t for t in a.atok if any(c.isdigit() for c in t) and any(c.isalpha() for c in t)}
    ab = {t for t in b.atok if any(c.isdigit() for c in t) and any(c.isalpha() for c in t)}
    return [float(exact), float(a_only), float(b_only), unit,
            1.0 if aa & ab else 0.0, 1.0 if (aa and ab and not aa & ab) else 0.0]


def core_after_alias(v: RecordView):
    return {t for t in v.ntok if t not in ertext.LEGAL_SUFFIXES}


def core2_features(a: RecordView, b: RecordView):
    ca, cb = core_after_alias(a), core_after_alias(b)
    return [ertext.jaccard(ca, cb), ertext.containment(ca, cb), float(len(cb - ca)),
            float(len(ca - cb)), 1.0 if (ca and cb and not ca & cb) else 0.0]


_AMAP = None


def init(aliases):
    global _AMAP
    _AMAP = ertext.load_aliases(aliases)


def chunk_rows(chunk):
    """chunk: [((name, addr), (name, addr))] -> rows of NUM_NAMES + CORE2_NAMES."""
    import numpy as np
    rows, cache = [], {}
    for s1, cand in chunk:
        a = cache.get(s1) or cache.setdefault(s1, RecordView(*s1, _AMAP))
        b = RecordView(*cand, _AMAP)
        rows.append(number_evidence(a, b) + core2_features(a, b))
    return np.asarray(rows, np.float32).reshape(-1, len(NUM_NAMES) + len(CORE2_NAMES))


def pair_rows(left, right, get_text, pool, chunk=4000):
    import numpy as np
    items = [(get_text(int(l)), get_text(int(r))) for l, r in zip(left, right)]
    parts = pool.map(chunk_rows, [items[i:i + chunk] for i in range(0, len(items), chunk)])
    return np.concatenate(parts) if parts else np.zeros((0, len(NUM_NAMES) + len(CORE2_NAMES)), np.float32)
