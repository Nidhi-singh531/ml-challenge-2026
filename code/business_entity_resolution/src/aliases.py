"""Mine an alias lexicon (transliteration + abbreviation) from the ground truth.

No external data: every equivalence is induced from pairs the training labels
already say are the same business. Two sources of evidence are combined.

1. **Positional alignment.** When a Source-1 name and a matched name have the
   same number of tokens, token i of one is the same word as token i of the
   other. This is what pins down transliterations:
   ``rial tek..`` <-> ``real technology``.
2. **Pointwise mutual information over co-occurrence.** Address tokens are
   reordered too freely to align positionally, so for those we score every
   (variant, canonical) co-occurrence by ``count(x,y) / (count(x) * P(y))``.
   Raw counts alone would tie ``प्राइवेट`` to both ``private`` and ``limited``;
   dividing by the marginal probability of y breaks that tie correctly.

Output is a two-column TSV consumed by ``ertext.load_aliases``.

    python3 src/aliases.py --data-dir dataset/train --prefix train \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --out work/aliases.tsv --max-clusters 400000
"""
from __future__ import annotations

import argparse
import collections
import os

import ertext
from blocking import iter_source


def is_latin(tok: str) -> bool:
    return any("a" <= c <= "z" for c in tok)


def mine(data_dir, prefix, gt_path, out_path, max_clusters=None,
         min_count=3, min_pmi=4.0, min_consistency=0.30, align_weight=3,
         max_word_ratio=0.5, merge_with=None):
    gt = {}
    with open(gt_path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            ids = [x for x in rest.split(",") if x]
            if ids:
                gt[s1] = ids
            if max_clusters and len(gt) >= max_clusters:
                break
    need = set(gt) | {x for v in gt.values() for x in v}
    print(f"[aliases] {len(gt)} clusters, {len(need)} records to load")

    recs = {}
    for i in (1, 2, 3):
        for rid, name, addr, _ in iter_source(
                os.path.join(data_dir, f"{prefix}_source{i}.tsv")):
            if rid in need:
                recs[rid] = (name, addr)

    cooc = collections.Counter()      # (variant, canonical) -> weighted count
    var_count = collections.Counter()  # variant -> occurrences
    can_count = collections.Counter()  # canonical -> occurrences in source 1
    n_docs = 0

    for s1, ids in gt.items():
        if s1 not in recs:
            continue
        s1_name, s1_addr = recs[s1]
        a_name = ertext.name_tokens(s1_name)
        a_all = set(a_name) | set(ertext.addr_tokens(s1_addr))
        a_latin = {t for t in a_all if is_latin(t)}
        n_docs += 1
        can_count.update(a_latin)

        for cid in ids:
            if cid not in recs:
                continue
            c_name, c_addr = recs[cid]
            b_name = ertext.name_tokens(c_name)
            b_all = set(b_name) | set(ertext.addr_tokens(c_addr))
            # variants worth learning: non-Latin tokens, plus short Latin tokens
            # absent from the Source-1 side (abbreviations such as rd / tx).
            variants = {t for t in b_all
                        if len(t) >= 2 and not t.isdigit()
                        and (not is_latin(t) or (len(t) <= 4 and t not in a_all))}
            if not variants:
                continue
            var_count.update(variants)

            # (1) positional evidence from equal-length name token lists
            if len(a_name) == len(b_name):
                for x, y in zip(b_name, a_name):
                    if x in variants and is_latin(y) and x != y:
                        cooc[(x, y)] += align_weight

            # (2) co-occurrence evidence over the whole record
            for x in variants:
                for y in a_latin:
                    cooc[(x, y)] += 1

    print(f"[aliases] {len(cooc)} candidate pairs from {n_docs} clusters")

    # PMI alone rewards rare canonicals: 'rd' would map to whatever unusual
    # street name it happened to sit next to. Requiring *consistency* as well --
    # y must accompany x in a large share of x's occurrences -- kills those and
    # keeps the real equivalences, whose consistency is near 1.
    best = {}
    for (x, y), c in cooc.items():
        if c < (min_count if not is_latin(x) else 10 * min_count):
            continue  # Latin variants need more evidence: a 4-letter typo seen
                      # in one cluster is not an abbreviation worth learning
        if is_latin(x) and can_count[x] > max_word_ratio * var_count[x]:
            continue  # x is an ordinary word on the Source-1 side too ('new' of
                      # 'New York' -> 'ny'); rewriting it would also turn
                      # 'New Delhi' / 'New Jersey' into 'ny'
        consistency = c / max(var_count[x], 1)
        if consistency < min_consistency:
            continue
        p_y = can_count[y] / max(n_docs, 1)
        pmi = c / (var_count[x] * max(p_y, 1e-9))
        if pmi < min_pmi:
            continue
        score = pmi * consistency
        if x not in best or score > best[x][1]:
            best[x] = (y, score, c)

    n_mined = len(best)
    if merge_with:
        # Bootstrap mode: `best` was mined from pseudo labels on unlabelled
        # data. The supervised lexicon wins every conflict; the bootstrap only
        # contributes variants it has never seen (e.g. French abbreviations).
        base = {}
        with open(merge_with, encoding="utf-8") as f:
            next(f, None)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[0] and p[1]:
                    base[p[0]] = (p[1], float(p[2]) if len(p) > 2 else 0.0,
                                  int(p[3]) if len(p) > 3 else 0)
        new = {x: v for x, v in best.items() if x not in base}
        # a new variant must not point at a token the base lexicon rewrites
        new = {x: v for x, v in new.items() if v[0] not in base}
        print(f"[aliases] bootstrap adds {len(new)} new variants to "
              f"{len(base)} from {merge_with}")
        for x, (y, s, c) in sorted(new.items(), key=lambda kv: -kv[1][2])[:25]:
            print(f"  NEW {x!r:>24} -> {y!r:<20} (count {c}, score {s:.1f})")
        best = {**base, **new}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("variant\tcanonical\tscore\tcount\n")
        for x, (y, pmi, c) in sorted(best.items(), key=lambda kv: -kv[1][1]):
            f.write(f"{x}\t{y}\t{pmi:.2f}\t{c}\n")
    print(f"[aliases] wrote {len(best)} aliases ({n_mined} mined here) -> {out_path}")

    print("\n[aliases] sample:")
    for x, (y, pmi, c) in list(sorted(best.items(), key=lambda kv: -kv[1][2]))[:25]:
        print(f"  {x!r:>30} -> {y!r:<20} (count {c}, score {pmi:.1f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--prefix", default="train")
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-clusters", type=int)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--min-pmi", type=float, default=4.0)
    ap.add_argument("--min-consistency", type=float, default=0.30)
    ap.add_argument("--merge-with",
                    help="bootstrap mode: --ground-truth is pseudo labels "
                         "(pipeline.py predict --confident-out); add only "
                         "variants this base lexicon lacks")
    a = ap.parse_args()
    mine(a.data_dir, a.prefix, a.ground_truth, a.out, a.max_clusters,
         a.min_count, a.min_pmi, a.min_consistency, merge_with=a.merge_with)
