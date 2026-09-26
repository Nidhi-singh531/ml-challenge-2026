"""Carve a small, self-consistent subset of the training data.

Useful for iterating on the pipeline in minutes instead of hours: it keeps
``--clusters`` Source-1 entities, all of their true matches, and a random pool
of distractor Source-2/3 records so precision is still exercised.

    python3 src/make_subset.py --data-dir dataset/train --out dataset/mini \
        --clusters 8000 --distractors 120000
"""
import argparse
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="train")
    ap.add_argument("--clusters", type=int, default=8000)
    ap.add_argument("--distractors", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    gt_in = os.path.join(a.data_dir, f"{a.prefix}_ground_truth.tsv")
    gt = {}
    with open(gt_in, encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if len(gt) >= a.clusters:
                break
            s1, _, rest = line.rstrip("\n").partition("\t")
            gt[s1] = [x for x in rest.split(",") if x]
    keep_targets = {x for v in gt.values() for x in v}

    with open(os.path.join(a.out, f"{a.prefix}_ground_truth.tsv"), "w",
              encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for k, v in gt.items():
            f.write(f"{k}\t{','.join(v)}\n")

    for src in (1, 2, 3):
        src_in = os.path.join(a.data_dir, f"{a.prefix}_source{src}.tsv")
        out = os.path.join(a.out, f"{a.prefix}_source{src}.tsv")
        kept = extra = 0
        with open(src_in, encoding="utf-8") as fi, open(out, "w", encoding="utf-8") as fo:
            fo.write(next(fi))
            for line in fi:
                rid = line.split("\t", 1)[0]
                if src == 1:
                    take = rid in gt
                else:
                    take = rid in keep_targets
                    if not take and extra < a.distractors and random.random() < 0.05:
                        take, extra = True, extra + 1
                if take:
                    fo.write(line)
                    kept += 1
        print(f"{out}: {kept} rows")


if __name__ == "__main__":
    main()
