"""Stage 2: re-score every candidate with its entity's whole candidate list.

Stage 1 scores a pair in isolation. A true cluster is a set of noisy variants of
one business, so its S2/S3 members also resemble *each other*; a same-building
false positive resembles the Source-1 address but none of the other members.
Stage 2 adds (a) list features from stage-1 probabilities, out-of-fold on dev,
and (b) candidate-to-candidate coherence, then decodes each entity's list either
by per-country threshold or by exact expected-F0.5 subset selection.

    python stage2.py dev      # OOF, features, train, honest A/B evaluation
    python stage2.py predict  # apply to the test pair file(s)
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import time

import numpy as np

import ertext
import extra_features as E
import features as F
import pipeline as P
from blocking import decode_id, iter_source, write_candidate_tsv

SRC_BASE = 1_000_000_000
LIST_NAMES = ["p1", "p1_rank", "p1_gap", "ent_pmax", "ent_p2nd", "ent_psum",
              "ent_n50", "ent_n", "best_in_source"]
COH_NAMES = ["coh_n_strong", "coh_name_max", "coh_name_tri_pw", "coh_addr_max",
             "coh_numconf_frac", "coh_support", "coh_top_name"]
STRONG_P, MAX_STRONG, MIN_P = 0.3, 8, 0.02
MAX_DECODE = 15
BETA2 = 0.25


# --------------------------------------------------------------------------- #
# list features
# --------------------------------------------------------------------------- #
def list_features(left, right, p):
    """Per-pair features of the entity's stage-1 probability list (input order)."""
    o = np.lexsort((-p, left))
    l, r, q = left[o], right[o], p[o].astype(np.float32)
    n = len(l)
    if n == 0:
        return np.zeros((0, len(LIST_NAMES)), np.float32)
    starts = np.flatnonzero(np.r_[True, l[1:] != l[:-1]])
    sizes = np.diff(np.r_[starts, n])
    first = np.repeat(starts, sizes)
    rank = np.arange(n) - first
    pmax = q[first]
    p2 = np.where(sizes > 1, q[np.minimum(starts + 1, n - 1)], 0.0)
    p2nd = np.repeat(p2, sizes)
    gap = np.where(rank == 0, q - p2nd, q - pmax)
    psum = np.repeat(np.add.reduceat(q, starts), sizes)
    n50 = np.repeat(np.add.reduceat((q >= 0.5).astype(np.float32), starts), sizes)
    src = r // SRC_BASE
    o2 = np.lexsort((-q, src, l))
    head = np.r_[True, (l[o2][1:] != l[o2][:-1]) | (src[o2][1:] != src[o2][:-1])]
    best = np.zeros(n, np.float32)
    best[o2[head]] = 1.0
    out = np.column_stack([q, rank, gap, pmax, p2nd, psum, n50,
                           np.repeat(sizes, sizes), best]).astype(np.float32)
    res = np.empty_like(out)
    res[o] = out
    return res


# --------------------------------------------------------------------------- #
# coherence features (worker processes)
# --------------------------------------------------------------------------- #
_AM = None


def _init(aliases):
    global _AM
    _AM = ertext.load_aliases(aliases)
    E._AMAP = _AM


def _sims(a, b):
    conflict = 1.0 if (a.num and b.num and not (a.num & b.num)) else 0.0
    return (F._token_set(a.nstr, b.nstr), ertext.jaccard(a.ntri, b.ntri),
            F._token_set(a.astr, b.astr), conflict)


def coherence_rows(cands):
    """cands: [(name, addr, p)] sorted by p descending -> (n, len(COH_NAMES))."""
    n = len(cands)
    rows = np.zeros((n, len(COH_NAMES)), np.float32)
    strong = [i for i in range(min(n, MAX_STRONG)) if cands[i][2] >= STRONG_P]
    active = [i for i in range(n) if cands[i][2] >= MIN_P]
    views = {i: F.RecordView(cands[i][0], cands[i][1], _AM)
             for i in set(active) | set(strong)}
    cache = {}
    for i in range(n):
        if i not in views or cands[i][2] < MIN_P:
            rows[i, 0] = -1.0
            continue
        others = [j for j in strong if j != i]
        rows[i, 0] = len(others)
        if not others:
            continue
        sims = []
        for j in others:
            key = (min(i, j), max(i, j))
            s = cache.get(key)
            if s is None:
                s = cache[key] = _sims(views[i], views[j])
            sims.append(s)
        s = np.asarray(sims, np.float32)
        w = np.asarray([cands[j][2] for j in others], np.float32)
        rows[i, 1] = s[:, 0].max()
        rows[i, 2] = float((s[:, 1] * w).sum() / w.sum())
        rows[i, 3] = s[:, 2].max()
        rows[i, 4] = s[:, 3].mean()
        rows[i, 5] = float(((s[:, 0] >= 0.8) & (w >= 0.5)).sum())
        rows[i, 6] = s[0, 0]
    return rows


def _coh_chunk(chunk):
    out = [coherence_rows(c) for c in chunk]
    return np.concatenate(out) if out else np.zeros((0, len(COH_NAMES)), np.float32)


JOINT_NAMES = ["joint_max", "joint_support_p", "joint_no_addr"]


def joint_rows(cands):
    """Does ONE other strong candidate agree with this one on name AND address?

    min(name, address) similarity per supporter, maximised over supporters; an
    empty address on either side gives no joint evidence (flagged, not a conflict).
    """
    n = len(cands)
    rows = np.zeros((n, len(JOINT_NAMES)), np.float32)
    strong = [i for i in range(min(n, MAX_STRONG)) if cands[i][2] >= STRONG_P]
    views = {}

    def view(i):
        if i not in views:
            views[i] = F.RecordView(cands[i][0], cands[i][1], _AM)
        return views[i]

    for i in range(n):
        if cands[i][2] < MIN_P:
            rows[i] = (-1.0, 0.0, 1.0)
            continue
        best, best_p, any_addr = 0.0, 0.0, False
        for j in strong:
            if j == i:
                continue
            a, b = view(i), view(j)
            if not a.atok or not b.atok:
                continue
            any_addr = True
            v = min(F._token_set(a.nstr, b.nstr), F._token_set(a.astr, b.astr))
            if v > best:
                best, best_p = v, cands[j][2]
        rows[i] = (best, best_p, 0.0 if any_addr else 1.0)
    return rows


def _joint_chunk(chunk):
    out = [joint_rows(c) for c in chunk]
    return np.concatenate(out) if out else np.zeros((0, len(JOINT_NAMES)), np.float32)


def coherence(left, right, p, get_text, pool, chunk_entities=1500, fn=_coh_chunk):
    o = np.lexsort((-p, left))
    l, r, q = left[o], right[o], p[o]
    starts = np.r_[np.flatnonzero(np.r_[True, l[1:] != l[:-1]]), len(l)] if len(l) \
        else np.zeros(1, np.int64)
    chunks, cur = [], []
    for lo, hi in zip(starts[:-1], starts[1:]):
        cur.append([(*get_text(int(r[k])), float(q[k])) for k in range(lo, hi)])
        if len(cur) >= chunk_entities:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    parts = pool.map(fn, chunks) if pool is not None else map(fn, chunks)
    rows = np.concatenate(list(parts)) if chunks else np.zeros((0, 0), np.float32)
    res = np.empty_like(rows)
    res[o] = rows
    return res


def stage2_matrix(X, left, right, p1, get_text, pool):
    return np.hstack([X, list_features(left, right, p1),
                      coherence(left, right, p1, get_text, pool)]).astype(np.float32)


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def pb_pmf(q):
    pmf = np.zeros(len(q) + 1)
    pmf[0] = 1.0
    for x in q:
        pmf[1:] = pmf[1:] * (1 - x) + pmf[:-1] * x
        pmf[0] *= 1 - x
    return pmf


def poisson_pmf(lam, kmax=12):
    k = np.arange(kmax + 1)
    return np.exp(-lam + k * math.log(lam) - np.array([math.lgamma(i + 1) for i in k])) \
        if lam > 0 else np.r_[1.0, np.zeros(kmax)]


def best_k(q, lam):
    """Exact expected-F0.5-optimal size k of the top-k set for sorted-desc q.

    With t true among the chosen k and u true links outside them,
    F0.5 = (1+b2) t / (b2 (t+u) + k); an empty prediction scores 1 iff t+u = 0.
    Unchosen candidates are Bernoulli(q); links blocking never found ~ Poisson(lam).
    """
    n = len(q)
    miss = poisson_pmf(lam)
    suffix = [None] * (n + 1)
    suffix[n] = miss
    for k in range(n - 1, -1, -1):
        suffix[k] = np.convolve(suffix[k + 1], [1 - q[k], q[k]])
    best, best_v = 0, suffix[0][0]
    tp = np.array([1.0])
    for k in range(1, n + 1):
        tp = np.convolve(tp, [1 - q[k - 1], q[k - 1]])
        u = suffix[k]
        t = np.arange(len(tp))[:, None]
        uu = np.arange(len(u))[None, :]
        v = float((tp[:, None] * u[None, :] * ((1 + BETA2) * t / (BETA2 * (t + uu) + k))).sum())
        if v > best_v:
            best, best_v = k, v
    return best, best_v


def decode_ef(left, right, q, lam_of_left):
    """Per-entity expected-F0.5 selection, then one-to-one on the targets."""
    o = np.lexsort((-q, left))
    l, r, qq = left[o], right[o], q[o]
    starts = np.r_[np.flatnonzero(np.r_[True, l[1:] != l[:-1]]), len(l)]
    keep = []
    for lo, hi in zip(starts[:-1], starts[1:]):
        if qq[lo] < 0.05:
            continue
        hi = min(hi, lo + MAX_DECODE)
        k, _ = best_k(qq[lo:hi], lam_of_left(int(l[lo])))
        keep.extend(range(lo, lo + k))
    idx = np.asarray(keep, np.int64)
    return P.assign(l[idx], r[idx], qq[idx], 0.0, True)


def lam_lookup(entities, ent_country, lam, default):
    ec = dict(zip(entities.tolist(), ent_country.tolist()))
    return lambda e: lam.get(ec.get(e), default)


# --------------------------------------------------------------------------- #
# dev: OOF stage 1, stage-2 training, honest evaluation
# --------------------------------------------------------------------------- #
def oof_stage1(X, y, left, train_ents, n_folds=5, seed=0):
    ents = np.array(sorted(train_ents), np.int64)
    fold = np.random.default_rng(seed).permutation(len(ents)) % n_folds
    is_tr = np.isin(left, ents)
    pf = np.full(len(left), -1)
    pf[is_tr] = fold[np.searchsorted(ents, left[is_tr])]
    p = np.full(len(left), np.nan, np.float32)
    for k in range(n_folds):
        fit, out = is_tr & (pf != k), is_tr & (pf == k)
        clf = P._make_clf()
        clf.fit(X[fit], y[fit])
        p[out] = clf.predict_proba(X[out])[:, 1]
        print(f"[stage2] OOF fold {k}: fit {int(fit.sum())} / predict {int(out.sum())}", flush=True)
    return p, pf


def subset_pred(pred, ents):
    return {e: pred.get(e, set()) for e in ents}


def score_on(pred, gt, ents):
    return P.macro_f05(subset_pred(pred, ents), {e: gt[e] for e in ents})


def tune_thresholds(left, right, p, country, gt, ents, ent_c):
    thr = {}
    for c in sorted(set(country.tolist())):
        m = country == c
        ce = [e for e in ents if ent_c[e] == c]
        gl = {e: gt[e] for e in ce}
        mm = m & np.isin(left, ce)
        thr[c], _, _ = P.sweep(left[mm], right[mm], p[mm], gl, True)
    return thr


def dev(args):
    t0 = time.time()
    D = P.load_pairs(args.pairs)
    X, y, left, right = D["X"], D["y"].astype(int), D["left"], D["right"]
    entities, ent_country = D["entities"], D["entity_country"]
    ent_c = dict(zip(entities.tolist(), ent_country.tolist()))
    with open(args.heldout_model, "rb") as f:
        held = pickle.load(f)
    if list(D["names"]) != list(held["features"]):
        raise ValueError("dev pair schema differs from the held-out stage-1 model")
    val = np.asarray(held["validation_ids"], np.int64)
    train_ents = np.setdiff1d(entities, val)
    is_val = np.isin(left, val)
    print(f"[stage2] {len(y)} pairs, {len(train_ents)} train / {len(val)} holdout entities")

    p1, _ = oof_stage1(X, y, left, train_ents)
    p1[is_val] = held["model"].predict_proba(X[is_val])[:, 1]

    if os.path.exists(args.cache):
        Z = np.load(args.cache)["Z"]
        print(f"[stage2] loaded features {args.cache}")
    else:
        need = set(right.tolist())
        texts = P.load_texts(args.data_dir, args.prefix, need)
        with mp.get_context("spawn").Pool(args.workers, _init, (args.aliases,)) as pool:
            Z = stage2_matrix(X, left, right, p1, lambda c: texts[c][:2], pool)
        np.savez(args.cache, Z=Z)
        print(f"[stage2] features {Z.shape} ({time.time() - t0:.0f}s)")
    names2 = list(D["names"]) + LIST_NAMES + COH_NAMES
    if args.extra_numbers:
        if not os.path.exists(args.extra_numbers):
            texts = P.load_texts(args.data_dir, args.prefix, set(left.tolist()) | set(right.tolist()))
            with mp.get_context("spawn").Pool(args.workers, _init, (args.aliases,)) as pool:
                np.savez(args.extra_numbers, X=E.pair_rows(left, right, lambda c: texts[c][:2], pool))
        Z = np.hstack([Z, np.load(args.extra_numbers)["X"][:, :len(E.NUM_NAMES)]])
        names2 += E.NUM_NAMES

    clf2 = P._make_clf(args.n_estimators)
    clf2.fit(Z[~is_val], y[~is_val])
    p2 = clf2.predict_proba(Z[is_val])[:, 1]
    imp = sorted(zip(clf2.feature_importances_, names2), reverse=True)[:20]
    print("[stage2] top importances:", ", ".join(f"{n}={v}" for v, n in imp))

    gt = P.load_ground_truth_codes(args.gt, set(val.tolist()))
    gt = {e: gt.get(e, set()) for e in val.tolist()}
    halves = np.random.default_rng(1).permutation(sorted(val.tolist()))
    A, B = sorted(halves[: len(halves) // 2].tolist()), sorted(halves[len(halves) // 2:].tolist())
    lv, rv = left[is_val], right[is_val]
    cv = P.pair_country(lv, entities, ent_country)
    p1v = p1[is_val]
    inA = np.isin(lv, A)
    report = {}

    def thr_eval(p, tag):
        thr = tune_thresholds(lv, rv, p, cv, gt, A, ent_c)
        t = np.array([thr[c] for c in cv], np.float32)
        pred = P.assign(lv, rv, p, t, True)
        report[tag] = {"A": score_on(pred, gt, A), "B": score_on(pred, gt, B),
                       "all": score_on(pred, gt, val.tolist()), "thresholds": thr}
        print(f"[stage2] {tag}: A {report[tag]['A']:.5f}  B {report[tag]['B']:.5f}  "
              f"all {report[tag]['all']:.5f}  thr {thr}", flush=True)
        return thr

    thr1 = thr_eval(p1v, "stage1_threshold")
    thr2 = thr_eval(p2, "stage2_threshold")

    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p2[inA], y[is_val][inA])
    q = iso.predict(p2).astype(np.float32)
    grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35]
    lam = {}
    for c in sorted(set(cv.tolist())):
        ce = [e for e in A if ent_c[e] == c]
        m = np.isin(lv, ce)
        best = max(grid, key=lambda g: score_on(
            decode_ef(lv[m], rv[m], q[m], lambda e: g), gt, ce))
        lam[c] = best
    pred = decode_ef(lv, rv, q, lam_lookup(entities, ent_country, lam, min(lam.values())))
    report["stage2_ef"] = {"A": score_on(pred, gt, A), "B": score_on(pred, gt, B),
                           "all": score_on(pred, gt, val.tolist()), "lambda": lam}
    print(f"[stage2] stage2_ef: A {report['stage2_ef']['A']:.5f}  B {report['stage2_ef']['B']:.5f}  "
          f"all {report['stage2_ef']['all']:.5f}  lambda {lam}", flush=True)

    # Final decode parameters for test: refit on the whole holdout.
    method = "ef" if report["stage2_ef"]["B"] > report["stage2_threshold"]["B"] else "threshold"
    iso_all = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p2, y[is_val])
    thr_all = tune_thresholds(lv, rv, p2, cv, gt, val.tolist(), ent_c)
    qa = iso_all.predict(p2).astype(np.float32)
    lam_all = {}
    for c in sorted(set(cv.tolist())):
        ce = [e for e in val.tolist() if ent_c[e] == c]
        m = np.isin(lv, ce)
        lam_all[c] = max(grid, key=lambda g: score_on(
            decode_ef(lv[m], rv[m], qa[m], lambda e: g), gt, ce))
    bundle = {"stage1": args.stage1_model, "model": clf2, "features": names2,
              "method": method, "calibrator": iso_all,
              "thresholds": thr_all, "default_threshold": max(thr_all.values()),
              "lambda": lam_all, "default_lambda": min(lam_all.values()),
              "extra_numbers": bool(args.extra_numbers), "report": report}
    with open(args.out, "wb") as f:
        pickle.dump(bundle, f)
    report["method"] = method
    report["final_thresholds"], report["final_lambda"] = thr_all, lam_all
    with open(args.out[:-4] + "_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"[stage2] selected {method}; saved {args.out} ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- #
# predict on test
# --------------------------------------------------------------------------- #
def predict(args):
    t0 = time.time()
    with open(args.model, "rb") as f:
        b2 = pickle.load(f)
    with open(b2["stage1"], "rb") as f:
        b1 = pickle.load(f)
    extra = b2.get("extra_numbers", False)
    # number features compare the S1 record with the candidate, so S1 texts are needed too
    store = P.TextStore(args.data_dir, args.prefix, which=(1, 2, 3) if extra else (2, 3))
    print(f"[stage2] texts loaded ({time.time() - t0:.0f}s)", flush=True)
    L, R, Q, ents, ecs = [], [], [], [], []
    mans = [P.load_manifest(m) for m in args.pairs]
    # a later manifest (e.g. one country re-blocked) replaces those entities' lists
    later = [np.concatenate([m["entities"] for m in mans[i + 1:]]) if i + 1 < len(mans)
             else np.empty(0, np.int64) for i in range(len(mans))]
    with mp.get_context("spawn").Pool(args.workers, _init, (args.aliases,)) as pool:
        for manifest, man, drop in zip(args.pairs, mans, later):
            if list(man["names"]) != list(b1["features"]):
                raise ValueError(f"{manifest}: feature schema differs from stage-1 model")
            keep_e = ~np.isin(man["entities"], drop)
            ents.append(man["entities"][keep_e]), ecs.append(man["entity_country"][keep_e])
            for path in P.part_paths(manifest):
                d = np.load(path)
                m = ~np.isin(d["left"], drop)
                if not m.any():
                    continue
                X, left, right = d["X"][m], d["left"][m], d["right"][m]
                p1 = b1["model"].predict_proba(X)[:, 1].astype(np.float32)
                Z = stage2_matrix(X, left, right, p1, store.get, pool)
                if extra:
                    Xn = E.pair_rows(left, right, store.get, pool)[:, :len(E.NUM_NAMES)]
                    Z = np.hstack([Z, Xn])
                p2 = b2["model"].predict_proba(Z)[:, 1]
                L.append(left), R.append(right), Q.append(p2.astype(np.float32))
                print(f"[stage2] {os.path.basename(path)}: {len(left)} pairs "
                      f"({time.time() - t0:.0f}s)", flush=True)
    # list and coherence features assume each entity's whole list is in one shard
    if sum(len(np.unique(x)) for x in L) != len(np.unique(np.concatenate(L))):
        raise ValueError("an entity's candidates are split across shards")
    left, right, p2 = np.concatenate(L), np.concatenate(R), np.concatenate(Q)
    entities, ent_country = np.concatenate(ents), np.concatenate(ecs)
    country = P.pair_country(left, entities, ent_country)
    method = args.method or b2["method"]
    for c in sorted(set(ent_country.tolist())):
        seen = c in b2["thresholds"]
        print(f"[stage2] {c}: threshold {b2['thresholds'].get(c, b2['default_threshold']):.3f}, "
              f"lambda {b2['lambda'].get(c, b2['default_lambda'])} "
              f"({'learned' if seen else 'UNSEEN -> conservative'})")
    if method == "ef":
        q = b2["calibrator"].predict(p2).astype(np.float32)
        pred = decode_ef(left, right, q, lam_lookup(entities, ent_country, b2["lambda"],
                                                    b2["default_lambda"]))
    else:
        # test has more unmatched targets than train; the leaderboard preferred +0.10
        t = np.minimum(np.array([b2["thresholds"].get(c, b2["default_threshold"]) for c in country])
                       + args.threshold_shift, 0.995)
        pred = P.assign(left, right, p2, t, True)
    np.savez(args.out[:-4] + "_p2.npz", left=left, right=right, p2=p2)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        n_empty = 0
        for rid, _, _, _ in iter_source(args.source1):
            ids = sorted(decode_id(x) for x in pred.get(P.encode_id(rid), ()))
            n_empty += not ids
            f.write(f"{rid}\t{','.join(ids)}\n")
    print(f"[stage2] method {method}: {sum(map(len, pred.values()))} links, "
          f"{n_empty} empty -> {args.out} ({time.time() - t0:.0f}s)")
    if args.candidates_out:
        o = np.argsort(left, kind="stable")
        ls, rs = left[o], right[o]
        starts = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1]])
        cands = {int(ls[lo]): rs[lo:hi] for lo, hi in zip(starts, np.r_[starts[1:], len(ls)])}
        write_candidate_tsv(args.candidates_out,
                            [r for r, _, _, _ in iter_source(args.source1)], cands)
        print(f"[stage2] candidate set -> {args.candidates_out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev")
    d.add_argument("--pairs", default="work/dev/pairs200k_filtered.npz")
    d.add_argument("--heldout-model", default="work/dev/model200k_larger_baseline.pkl")
    d.add_argument("--stage1-model", default="work/dev/model_optimized.pkl")
    d.add_argument("--gt", default="dataset/train/train_ground_truth.tsv")
    d.add_argument("--data-dir", default="dataset/train")
    d.add_argument("--prefix", default="train")
    d.add_argument("--aliases", default="work/aliases_full.tsv")
    d.add_argument("--cache", default="work/dev/stage2_dev_features.npz")
    d.add_argument("--extra-numbers", help="cache of extra_features rows; appends the number features")
    d.add_argument("--out", default="work/dev/stage2.pkl")
    d.add_argument("--workers", type=int, default=3)
    d.add_argument("--n-estimators", type=int, default=500)
    p = sub.add_parser("predict")
    p.add_argument("--pairs", nargs="+", default=["work/pairs_test.npz"])
    p.add_argument("--model", default="work/dev/stage2.pkl")
    p.add_argument("--data-dir", default="dataset/test")
    p.add_argument("--prefix", default="test")
    p.add_argument("--aliases", default="work/aliases_full.tsv")
    p.add_argument("--source1", default="dataset/test/test_source1.tsv")
    p.add_argument("--out", default="output/matching_results.tsv")
    p.add_argument("--candidates-out")
    p.add_argument("--method", choices=["ef", "threshold"])
    p.add_argument("--threshold-shift", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    dev(a) if a.cmd == "dev" else predict(a)


if __name__ == "__main__":
    main()
