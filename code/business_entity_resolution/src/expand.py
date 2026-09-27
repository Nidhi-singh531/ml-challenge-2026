"""Cluster expansion: recover true links that blocking's top-K cut dropped.

Nearly half of the true links blocking misses share an exact *name key* with the
Source-1 record or with a record already matched to it (identical Devanagari
names, 'winadvisors.com' vs 'Win Advisors'); they lost the top-K race to
records with more address overlap. We index three keys of every S2/S3 record
(canonical core tokens, consonant skeleton, compact romanised string), look up
keys of rare frequency from each entity's S1 record and matched members, and
score the new pairs with a model trained on dev labels.

    python expand.py index   --data-dir dataset/train --prefix train --out work/dev/keyidx_train.npz
    python expand.py dev     # generate on dev, train, honest A/B evaluation
    python expand.py predict # add accepted expansions to the stage-2 test output
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import time
from collections import defaultdict

import numpy as np

import ertext
import features as F
import pipeline as P
import stage2 as S
from blocking import decode_id, encode_id, iter_source, write_candidate_tsv

KEY_TYPES = ("core", "skel", "compact")
FMAX = 20          # a key shared by more records than this is a common name
MAX_NEW = 30       # new candidates per entity, rarest keys first
EXP_NAMES = ["hit_core_s1", "hit_skel_s1", "hit_compact_s1", "hit_core_m", "hit_skel_m",
             "hit_compact_m", "n_member_hits", "log_min_freq", "best_member_p",
             "ent_n_pred", "ent_pmax", "ent_psum", "is_s3",
             "mem_name_max", "mem_tri_max", "mem_addr_max", "mem_conflict_min"]
_AMAP = None


def _init(aliases):
    global _AMAP
    _AMAP = ertext.load_aliases(aliases)
    S._AM = _AMAP
    P._AMAP = _AMAP


def name_keys3(name, amap):
    core = ertext.canonical(ertext.core_name_tokens(name), amap)
    skel = sorted({s for s in (ertext.skeleton(t) for t in core) if len(s) >= 2})
    return (" ".join(sorted(set(core))), " ".join(skel),
            "".join(ertext.romanize(t) for t in core))


def key_hash(country, kind, value):
    return int.from_bytes(hashlib.blake2b(f"{country}|{kind}|{value}".encode("utf-8"),
                                          digest_size=8).digest(), "little", signed=True)


def record_hashes(country, name, amap):
    return [key_hash(country, k, v) if len(v) >= 3 else None
            for k, v in zip(KEY_TYPES, name_keys3(name, amap))]


def _hash_chunk(rows):
    out_h, out_c = [], []
    for code, name, country in rows:
        for h in record_hashes(country, name, _AMAP):
            if h is not None:
                out_h.append(h)
                out_c.append(code)
    return np.asarray(out_h, np.int64), np.asarray(out_c, np.int64)


def build_index(data_dir, prefix, out, aliases, workers):
    t0 = time.time()

    def chunks():
        cur = []
        for i in (2, 3):
            for rid, name, _, ctry in iter_source(os.path.join(data_dir, f"{prefix}_source{i}.tsv")):
                cur.append((encode_id(rid), name, ctry))
                if len(cur) >= 20000:
                    yield cur
                    cur = []
        if cur:
            yield cur

    H, C = [], []
    with mp.get_context("spawn").Pool(workers, _init, (aliases,)) as pool:
        for h, c in pool.imap(_hash_chunk, chunks()):
            H.append(h), C.append(c)
    h, c = np.concatenate(H), np.concatenate(C)
    o = np.argsort(h, kind="stable")
    np.savez(out, h=h[o], c=c[o])
    print(f"[expand] indexed {len(h)} keys of S2/S3 -> {out} ({time.time() - t0:.0f}s)")


class KeyIndex:
    def __init__(self, path):
        d = np.load(path)
        self.h, self.c = d["h"], d["c"]

    def lookup(self, hashes):
        hashes = np.asarray(hashes, np.int64)
        lo = np.searchsorted(self.h, hashes, "left")
        hi = np.searchsorted(self.h, hashes, "right")
        return lo, hi


# --------------------------------------------------------------------------- #
# candidate generation
# --------------------------------------------------------------------------- #
def generate(entities, ent_country, seeds, existing, s1_text, idx, amap):
    """seeds: {entity: [(member_code, p, member_name)]} (S1 itself is added here).

    Returns rows: (entity, target, key-hit bits, n_member_hits, min_freq, best_member_p).
    """
    ec = dict(zip(entities.tolist(), ent_country.tolist()))
    out = []
    for e in entities.tolist():
        country = ec[e]
        name = s1_text(e)[0]
        seed_list = [(e, 1.0, name, True)] + [(m, p, nm, False) for m, p, nm in seeds.get(e, ())]
        found = {}
        have = existing.get(e, set())
        for code, p, nm, is_s1 in seed_list:
            hs = record_hashes(country, nm, amap)
            valid = [(k, h) for k, h in enumerate(hs) if h is not None]
            if not valid:
                continue
            lo, hi = idx.lookup([h for _, h in valid])
            for (k, _), a, b in zip(valid, lo, hi):
                freq = b - a
                if freq == 0 or freq > FMAX:
                    continue
                for t in idx.c[a:b].tolist():
                    if t in have or t == code:
                        continue
                    r = found.get(t)
                    if r is None:
                        r = found[t] = [0] * 6 + [0, FMAX + 1, 0.0, set()]
                    r[k + (0 if is_s1 else 3)] = 1
                    if not is_s1:
                        r[9].add(code)
                        r[8] = max(r[8], p)
                    r[7] = min(r[7], freq)
        for t, r in sorted(found.items(), key=lambda kv: kv[1][7])[:MAX_NEW]:
            out.append((e, t, r[:6], len(r[9]), r[7], r[8], sorted(r[9])))
    return out


def _feat_chunk(chunk):
    """chunk: [(s1 (name, addr), cand (name, addr), [member (name, addr)])]."""
    rows = []
    for s1, cand, members in chunk:
        a, b = F.RecordView(*s1, _AMAP), F.RecordView(*cand, _AMAP)
        tf = F.text_features(a, b)
        sims = [S._sims(F.RecordView(*m, _AMAP), b) for m in members] or [(0.0, 0.0, 0.0, 1.0)]
        s = np.asarray(sims, np.float32)
        rows.append(tf + [float(s[:, 0].max()), float(s[:, 1].max()), float(s[:, 2].max()),
                          float(s[:, 3].min())])
    return np.asarray(rows, np.float32).reshape(-1, len(F.TEXT_FEATURE_NAMES) + 4)


def featurize(gen, get_text, ent_stats, pool, chunk=2000):
    if not gen:
        return np.zeros((0, len(F.TEXT_FEATURE_NAMES) + len(EXP_NAMES)), np.float32)
    items = [(get_text(e), get_text(t), [get_text(m) for m in mem][:6])
             for e, t, _, _, _, _, mem in gen]
    parts = pool.map(_feat_chunk, [items[i:i + chunk] for i in range(0, len(items), chunk)])
    tfc = np.concatenate(parts)
    nt = len(F.TEXT_FEATURE_NAMES)
    meta = np.array([bits + [n_m, np.log(freq), best_p, *ent_stats(e), float(t // S.SRC_BASE == 3)]
                     for e, t, bits, n_m, freq, best_p, _ in gen], np.float32)
    return np.hstack([tfc[:, :nt], meta, tfc[:, nt:]]).astype(np.float32)


def entity_stats(left, p, members):
    o = np.argsort(left, kind="stable")
    l, q = left[o], p[o]
    starts = np.flatnonzero(np.r_[True, l[1:] != l[:-1]])
    pmax = np.maximum.reduceat(q, starts) if len(l) else np.zeros(0)
    psum = np.add.reduceat(q, starts) if len(l) else np.zeros(0)
    st = dict(zip(l[starts].tolist(), zip(pmax.tolist(), psum.tolist())))
    return lambda e: (float(len(members.get(e, ()))), *st.get(e, (0.0, 0.0)))


# --------------------------------------------------------------------------- #
# dev
# --------------------------------------------------------------------------- #
def dev(args):
    t0 = time.time()
    amap = ertext.load_aliases(args.aliases)
    D = P.load_pairs(args.pairs, with_X=False)
    y, left, right = D["y"].astype(int), D["left"], D["right"]
    entities, ent_country = D["entities"], D["entity_country"]
    b2 = pickle.load(open(args.stage2, "rb"))
    held = pickle.load(open(args.heldout_model, "rb"))
    val = np.asarray(held["validation_ids"], np.int64)
    Z = np.load(args.stage2_features)["Z"]
    is_val = np.isin(left, val)
    # members: final stage-2 predictions on the holdout (as at test time);
    # out-of-fold stage-1 p >= 0.7 on training entities (no in-sample optimism)
    p_mem = Z[:, len(D["names"])].copy()
    p2v = b2["model"].predict_proba(Z[is_val])[:, 1]
    p_mem[is_val] = p2v
    cty = P.pair_country(left, entities, ent_country)
    thr = np.array([b2["thresholds"].get(c, b2["default_threshold"]) for c in cty], np.float32)
    thr[~is_val] = 0.7
    pred = P.assign(left, right, p_mem, thr, True)
    store = P.TextStore(args.data_dir, args.prefix)
    print(f"[expand] texts loaded ({time.time() - t0:.0f}s)", flush=True)
    existing = defaultdict(set)
    for l, r in zip(left.tolist(), right.tolist()):
        existing[l].add(r)
    pm = dict(zip(zip(left.tolist(), right.tolist()), p_mem.tolist()))
    seeds = {e: [(m, pm[(e, m)], store.get(m)[0]) for m in ms] for e, ms in pred.items()}
    idx = KeyIndex(args.index)
    gen = generate(entities, ent_country, seeds, existing, store.get, idx, amap)
    print(f"[expand] {len(gen)} new pairs for {len(entities)} entities "
          f"({time.time() - t0:.0f}s)", flush=True)
    gt = P.load_ground_truth_codes(args.gt, set(entities.tolist()))
    yy = np.array([t in gt.get(e, ()) for e, t, *_ in gen], np.int8)
    missed = sum(len(v - existing[k]) for k, v in gt.items())
    print(f"[expand] positives among new pairs: {int(yy.sum())} of {missed} links "
          f"blocking/prefilter missed", flush=True)
    with mp.get_context("spawn").Pool(args.workers, _init, (args.aliases,)) as pool:
        Xe = featurize(gen, store.get, entity_stats(left, p_mem, pred), pool)
    ge = np.array([g[0] for g in gen], np.int64)
    gr = np.array([g[1] for g in gen], np.int64)
    ev = np.isin(ge, val)
    clf = P._make_clf(400)
    clf.fit(Xe[~ev], yy[~ev])
    pe = clf.predict_proba(Xe[ev])[:, 1]
    print(f"[expand] features {Xe.shape}; fitted ({time.time() - t0:.0f}s)", flush=True)

    gtv = {e: gt.get(e, set()) for e in val.tolist()}
    halves = np.random.default_rng(1).permutation(sorted(val.tolist()))
    A, B = sorted(halves[: len(halves) // 2].tolist()), sorted(halves[len(halves) // 2:].tolist())
    base = {e: s for e, s in pred.items() if e in set(val.tolist())}
    claimed = {r for s in pred.values() for r in s}
    gv, rv = ge[ev], gr[ev]

    def with_exp(t):
        out = {e: set(s) for e, s in base.items()}
        o = np.argsort(-pe)
        taken = set(claimed)
        for i in o:
            if pe[i] < t:
                break
            r = int(rv[i])
            if r in taken:
                continue
            taken.add(r)
            out.setdefault(int(gv[i]), set()).add(r)
        return out

    grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.01]
    sc = {t: S.score_on(with_exp(t), gtv, A) for t in grid}
    t_best = max(grid, key=lambda t: sc[t])
    rep = {"base_A": S.score_on(base, gtv, A), "base_B": S.score_on(base, gtv, B),
           "exp_A": sc[t_best], "exp_B": S.score_on(with_exp(t_best), gtv, B),
           "threshold": t_best, "grid_A": sc,
           "new_pairs": len(gen), "new_positives": int(yy.sum())}
    print(f"[expand] base A {rep['base_A']:.5f} B {rep['base_B']:.5f} | with expansion "
          f"(t={t_best}) A {rep['exp_A']:.5f} B {rep['exp_B']:.5f}", flush=True)
    clf_all = P._make_clf(400)
    clf_all.fit(Xe, yy)
    with open(args.out, "wb") as f:
        pickle.dump({"model": clf_all, "threshold": t_best, "fmax": FMAX, "max_new": MAX_NEW,
                     "features": list(F.TEXT_FEATURE_NAMES) + EXP_NAMES, "report": rep}, f)
    with open(args.out[:-4] + "_report.json", "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print(f"[expand] saved {args.out} ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- #
# predict on test
# --------------------------------------------------------------------------- #
def predict(args):
    t0 = time.time()
    amap = ertext.load_aliases(args.aliases)
    b2 = pickle.load(open(args.stage2, "rb"))
    be = pickle.load(open(args.model, "rb"))
    d = np.load(args.p2)
    left, right, p2 = d["left"], d["right"], d["p2"]
    ents, ecs = [], []
    for m in args.pairs:
        man = P.load_manifest(m)
        ents.append(man["entities"]), ecs.append(man["entity_country"])
    entities, ent_country = np.concatenate(ents), np.concatenate(ecs)
    entities, first = np.unique(entities, return_index=True)
    ent_country = ent_country[first]
    cty = P.pair_country(left, entities, ent_country)
    thr = np.array([b2["thresholds"].get(c, b2["default_threshold"]) for c in cty], np.float32)
    pred = P.assign(left, right, p2, thr, True)
    store = P.TextStore(args.data_dir, args.prefix)
    print(f"[expand] texts loaded ({time.time() - t0:.0f}s)", flush=True)
    existing = defaultdict(set)
    for l, r in zip(left.tolist(), right.tolist()):
        existing[l].add(r)
    pm = dict(zip(zip(left.tolist(), right.tolist()), p2.tolist()))
    seeds = {e: [(m, pm[(e, m)], store.get(m)[0]) for m in ms] for e, ms in pred.items()}
    idx = KeyIndex(args.index)
    all_s1 = np.array([encode_id(r) for r, _, _, _ in iter_source(args.source1)], np.int64)
    s1_ctry = np.array([c for _, _, _, c in iter_source(args.source1)])
    gen = generate(all_s1, s1_ctry, seeds, existing, store.get, idx, amap)
    print(f"[expand] {len(gen)} new pairs ({time.time() - t0:.0f}s)", flush=True)
    with mp.get_context("spawn").Pool(args.workers, _init, (args.aliases,)) as pool:
        Xe = featurize(gen, store.get, entity_stats(left, p2, pred), pool)
    pe = be["model"].predict_proba(Xe)[:, 1] if len(gen) else np.zeros(0)
    taken = {r for s in pred.values() for r in s}
    added = 0
    for i in np.argsort(-pe):
        if pe[i] < be["threshold"]:
            break
        e, t = gen[i][0], gen[i][1]
        if t in taken:
            continue
        taken.add(t)
        pred.setdefault(e, set()).add(t)
        added += 1
    print(f"[expand] accepted {added} expansion links (threshold {be['threshold']})")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        n_empty = 0
        for rid, _, _, _ in iter_source(args.source1):
            ids = sorted(decode_id(x) for x in pred.get(encode_id(rid), ()))
            n_empty += not ids
            f.write(f"{rid}\t{','.join(ids)}\n")
    print(f"[expand] {sum(map(len, pred.values()))} links, {n_empty} empty -> {args.out}")
    if args.candidates_out:
        cands = {k: np.fromiter(v, np.int64) for k, v in existing.items()}
        for i, g in enumerate(gen):
            if pe[i] >= be["threshold"]:
                cands[g[0]] = np.append(cands.get(g[0], np.zeros(0, np.int64)), g[1])
        write_candidate_tsv(args.candidates_out,
                            [r for r, _, _, _ in iter_source(args.source1)], cands)
        print(f"[expand] candidate set -> {args.candidates_out} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index")
    i.add_argument("--data-dir", required=True)
    i.add_argument("--prefix", required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--aliases", default="work/aliases_full.tsv")
    i.add_argument("--workers", type=int, default=3)
    d = sub.add_parser("dev")
    d.add_argument("--pairs", default="work/dev/pairs200k_filtered.npz")
    d.add_argument("--stage2", default="work/dev/stage2.pkl")
    d.add_argument("--stage2-features", default="work/dev/stage2_dev_features.npz")
    d.add_argument("--heldout-model", default="work/dev/model200k_larger_baseline.pkl")
    d.add_argument("--index", default="work/dev/keyidx_train.npz")
    d.add_argument("--gt", default="dataset/train/train_ground_truth.tsv")
    d.add_argument("--data-dir", default="dataset/train")
    d.add_argument("--prefix", default="train")
    d.add_argument("--aliases", default="work/aliases_full.tsv")
    d.add_argument("--out", default="work/dev/expand.pkl")
    d.add_argument("--workers", type=int, default=3)
    p = sub.add_parser("predict")
    p.add_argument("--p2", required=True)
    p.add_argument("--pairs", nargs="+", default=["work/pairs_test.npz"])
    p.add_argument("--stage2", default="work/dev/stage2.pkl")
    p.add_argument("--model", default="work/dev/expand.pkl")
    p.add_argument("--index", default="work/keyidx_test.npz")
    p.add_argument("--data-dir", default="dataset/test")
    p.add_argument("--prefix", default="test")
    p.add_argument("--aliases", default="work/aliases_full.tsv")
    p.add_argument("--source1", default="dataset/test/test_source1.tsv")
    p.add_argument("--out", required=True)
    p.add_argument("--candidates-out")
    p.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    {"index": lambda: build_index(a.data_dir, a.prefix, a.out, a.aliases, a.workers),
     "dev": lambda: dev(a), "predict": lambda: predict(a)}[a.cmd]()


if __name__ == "__main__":
    main()
