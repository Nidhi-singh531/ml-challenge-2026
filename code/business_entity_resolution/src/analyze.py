"""Diagnostics: where is the score actually being lost?

Two sub-commands.

``blocking``  Recall of the candidate set as a function of K, plus a sample of
              the true pairs that blocking never retrieved. This is the ceiling
              on everything downstream — if recall@K is 0.974, no classifier
              change can recover those links.

``errors``    Decomposes the validation loss into (a) recall already lost in
              blocking, (b) false positives the classifier added, (c) true
              candidates the threshold rejected. Prints feature importances and
              concrete examples of each error type, because the examples are
              what tell you which feature to write next.
"""
from __future__ import annotations

import argparse
import pickle
import random

import numpy as np

from blocking import decode_id, load_scores
from pipeline import (assign, load_ground_truth_codes, load_pairs, load_texts,
                      macro_f05, pair_country, thresholds_for, val_split)


def pair_key(s1, cand):
    """Unique int64 per (S1 code, S2/S3 code): 30 bits of S1 number, 32 of cand."""
    return ((s1 % 1_000_000_000) << 32) | cand


def cmd_blocking(cand_scores, gt_path, ks, data_dir=None, prefix="train",
                 n_examples=12):
    sc = load_scores(cand_scores)
    s1, cand, rank = sc["s1"], sc["cand"], sc["rank"]
    queried = sc.get("stored", sc.get("queried"))  # entities whose pairs were kept
    queried = np.unique(s1) if queried is None else queried
    gt = load_ground_truth_codes(gt_path, set(queried.tolist()))

    # label every blocking pair, then recall@K is a cumulative count by rank
    tl = np.fromiter((l for l, v in gt.items() for _ in v), np.int64)
    tr = np.fromiter((r for v in gt.values() for r in v), np.int64)
    total = len(tl)
    key_true = np.sort(pair_key(tl, tr))
    key_pair = pair_key(s1, cand)
    pos = np.searchsorted(key_true, key_pair)
    pos[pos >= len(key_true)] = 0
    hit = key_true[pos] == key_pair
    print(f"[blocking] {len(queried)} source-1 entities, {total} true links, "
          f"{len(s1)} candidate pairs")
    for k in ks:
        m = rank < k
        avg = m.sum() / max(len(queried), 1)
        print(f"  recall@{k:<3d} = {hit[m].sum() / max(total, 1):.4f}   "
              f"({avg:.1f} candidates per entity, ranked by `all` score)")
    if "nrank" in sc:
        by_name = sc["nrank"] < 999
        print(f"  name channel alone     : {hit[by_name].sum() / max(total, 1):.4f}")
        print(f"  union of both channels : {hit.sum() / max(total, 1):.4f}   "
              f"({len(s1) / max(len(queried), 1):.1f} candidates per entity)")

    found = {}
    for l, r in zip(s1[hit].tolist(), cand[hit].tolist()):
        found.setdefault(l, set()).add(r)
    missed = [(l, v - found.get(l, set())) for l, v in gt.items()]
    missed = [(l, m) for l, m in missed if m]
    print(f"\n[blocking] {sum(len(m) for _, m in missed)} true links never "
          f"retrieved, over {len(missed)} entities")

    if data_dir and missed:
        random.seed(0)
        sample = random.sample(missed, min(n_examples, len(missed)))
        recs = load_texts(data_dir, prefix,
                          {l for l, _ in sample} | {x for _, m in sample for x in m})
        for l, m in sample:
            n, a, _ = recs.get(l, ("?", "?", ""))
            print(f"\n  S1 {n!r} | {a!r}")
            for x in sorted(m):
                xn, xa, _ = recs.get(x, ("?", "?", ""))
                print(f"     MISS {decode_id(x)[:2]} {xn!r} | {xa!r}")


def cmd_errors(pairs_npz, model_path, gt_path, data_dir=None, prefix="train",
               n_examples=10):
    P = load_pairs(pairs_npz)
    X, y, left, right = P["X"], P["y"].astype(int), P["left"], P["right"]
    names = list(P["names"])
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    clf = bundle["model"]

    val_ids = val_split(P["entities"])
    is_val = np.isin(left, np.fromiter(val_ids, np.int64))
    Xv, yv, lv, rv = X[is_val], y[is_val], left[is_val], right[is_val]
    prob = clf.predict_proba(Xv)[:, 1]
    thr = thresholds_for(bundle, pair_country(lv, P["entities"], P["entity_country"]))
    gt = load_ground_truth_codes(gt_path, val_ids)
    gt_val = {k: gt.get(k, set()) for k in val_ids}

    # --- ceilings -----------------------------------------------------------
    oracle = {}
    for i in np.flatnonzero(yv == 1):
        oracle.setdefault(int(lv[i]), set()).add(int(rv[i]))
    pred = assign(lv, rv, prob, thr, bundle["one_to_one"])
    s_or, s_cur = macro_f05(oracle, gt_val), macro_f05(pred, gt_val)
    print(f"[errors] validation entities: {len(gt_val)}")
    print(f"  oracle over candidates (perfect classifier) : {s_or:.4f}")
    print(f"  current model                                : {s_cur:.4f}")
    print(f"  headroom inside the candidate set            : {s_or - s_cur:.4f}")

    # --- per-entity error taxonomy -------------------------------------------
    perfect = fp_only = fn_only = both = empty_wrong = 0
    fp_rows, fn_rows = [], []
    idx_by_s1 = {}
    for i, s1 in enumerate(lv.tolist()):
        idx_by_s1.setdefault(s1, []).append(i)
    for s1, truth in gt_val.items():
        got = pred.get(s1, set())
        fp, fn = got - truth, truth - got
        if not fp and not fn:
            perfect += 1
        elif fp and fn:
            both += 1
        elif fp:
            fp_only += 1
            if not truth:
                empty_wrong += 1
        else:
            fn_only += 1
        for i in idx_by_s1.get(s1, []):
            if rv[i] in fp:
                fp_rows.append((s1, int(rv[i]), prob[i]))
            elif rv[i] in fn:
                fn_rows.append((s1, int(rv[i]), prob[i]))
    n = max(len(gt_val), 1)
    print(f"\n[errors] entities fully correct : {perfect} ({perfect / n:.3f})")
    print(f"         false positives only   : {fp_only} "
          f"(of which {empty_wrong} are true singletons wrongly merged)")
    print(f"         missed matches only    : {fn_only}")
    print(f"         both                   : {both}")
    print(f"\n[errors] {len(fn_rows)} true links were candidates but scored below "
          f"the threshold; {len(fp_rows)} wrong links scored above it")

    # --- feature importance ---------------------------------------------------
    imp = getattr(clf, "feature_importances_", None)
    if imp is not None:
        print("\n[errors] feature importance (high -> low)")
        for j in np.argsort(-np.asarray(imp)):
            print(f"  {names[j]:<22s} {imp[j]:.0f}")

    # --- examples -------------------------------------------------------------
    if data_dir:
        random.seed(0)
        samples = {lab: random.sample(rows, min(n_examples, len(rows)))
                   for lab, rows in (("FALSE POSITIVE", fp_rows),
                                     ("MISSED MATCH", fn_rows))}
        need = {x for rows in samples.values() for s1, c, _ in rows for x in (s1, c)}
        recs = load_texts(data_dir, prefix, need)
        for label, rows in samples.items():
            print(f"\n=== {label} examples ===")
            for s1, c, p in rows:
                sn, sa, _ = recs.get(s1, ("?", "?", ""))
                cn, ca, _ = recs.get(c, ("?", "?", ""))
                print(f"  p={p:.3f}")
                print(f"    S1 {sn!r} | {sa!r}")
                print(f"    {decode_id(c)[:2]} {cn!r} | {ca!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("blocking")
    b.add_argument("--cand-scores", required=True)
    b.add_argument("--ground-truth", required=True)
    b.add_argument("--data-dir")
    b.add_argument("--prefix", default="train")
    b.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 30, 50, 75, 100])

    e = sub.add_parser("errors")
    e.add_argument("--pairs", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--ground-truth", required=True)
    e.add_argument("--data-dir")
    e.add_argument("--prefix", default="train")

    a = ap.parse_args()
    if a.cmd == "blocking":
        cmd_blocking(a.cand_scores, a.ground_truth, a.ks, a.data_dir, a.prefix)
    else:
        cmd_errors(a.pairs, a.model, a.ground_truth, a.data_dir, a.prefix)
