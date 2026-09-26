"""Optional classifier-only evidence; never used by blocking."""
from collections import Counter

import numpy as np

import ertext
from blocking import iter_source


FEATURE_NAMES = [
    "s1_core_name_frequency", "cand_core_name_frequency", "core_name_count_ratio",
    "addr_tokens_s1", "addr_tokens_cand", "addr_coverage_s1", "addr_coverage_cand",
    "addr_unshared_cand", "addr_numbers_s1", "addr_numbers_cand",
    "addr_numbers_unshared_cand", "addr_alpha_disjoint",
]


def name_key(name, aliases):
    return tuple(sorted(set(ertext.canonical(ertext.core_name_tokens(name), aliases))))


class NameCounts:
    """Counts over ALL S1 records, within country, without labels or sampling."""

    def __init__(self, source1, aliases):
        self.aliases = aliases
        self.counts = Counter()
        for _, name, _, country in iter_source(source1):
            key = name_key(name, aliases)
            if key:
                self.counts[country, key] += 1

    def get(self, name, country):
        return self.counts.get((country, name_key(name, self.aliases)), 0)


def pair_features(a, b, a_count, b_count):
    shared = len(a.atok & b.atok)
    aa, ba = a.atok - a.num, b.atok - b.num
    return [
        np.log1p(a_count), np.log1p(b_count),
        min(a_count, b_count) / max(a_count, b_count, 1),
        len(a.atok), len(b.atok), shared / max(len(a.atok), 1),
        shared / max(len(b.atok), 1), len(b.atok - a.atok),
        len(a.num), len(b.num), len(b.num - a.num),
        float(bool(aa and ba and not (aa & ba))),
    ]
