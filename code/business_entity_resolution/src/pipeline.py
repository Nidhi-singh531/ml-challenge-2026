"""Train / score / decide / evaluate for the entity-resolution pipeline.

Commands
--------
  build     candidate pairs + features (+ labels when ground truth exists)
  train     fit the pair classifier, the stage-1 prefilter, and per-country
            F_0.5-optimal thresholds
  predict   score candidates and write matching_results.tsv
  evaluate  macro F_0.5 of a results file against ground truth

Pair files
----------
``build --out X.npz`` writes a small manifest ``X.npz`` (entities, their
countries, feature names) plus ``X.partNNN.npz`` shards holding the feature
matrix. Sharding keeps the full test set (~1.7M entities) inside 16 GB RAM.
Ids are int64-encoded (``blocking.encode_id``) throughout.
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import pickle
import time
from array import array

import numpy as np

import features as F
from blocking import decode_id, encode_id, iter_source, load_meta, \
    score_parts, write_candidate_tsv


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #
def load_ground_truth(path, restrict=None):
    """{s1_id: set(ids)} with string ids (restrict: optional set of s1 ids)."""
    gt = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            if restrict is not None and s1 not in restrict:
                continue
            gt[s1] = {x for x in rest.split(",") if x}
    return gt


def load_ground_truth_codes(path, restrict=None):
    """Same as ``load_ground_truth`` with int64-encoded ids."""
    gt = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            c = encode_id(s1)
            if restrict is not None and c not in restrict:
                continue
            gt[c] = {encode_id(x) for x in rest.split(",") if x}
    return gt


def load_texts(data_dir, prefix, need):
    """{code: (name, addr, country)} for the int-encoded ids in ``need``."""
    recs = {}
    for i in (1, 2, 3):
        for rid, name, addr, ctry in iter_source(
                os.path.join(data_dir, f"{prefix}_source{i}.tsv")):
            c = encode_id(rid)
            if c in need:
                recs[c] = (name, addr, ctry)
    return recs


def part_paths(manifest):
    base = manifest[:-4] if manifest.endswith(".npz") else manifest
    return sorted(glob.glob(f"{base}.part*.npz"))


def load_manifest(manifest):
    d = np.load(manifest, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_pairs(manifest, with_X=True):
    """Concatenate every shard of a pair file (training-sized files only)."""
    cols = {"X": [], "y": [], "left": [], "right": []}
    for p in part_paths(manifest):
        d = np.load(p)
        for k in cols:
            if k == "X" and not with_X:
                continue
            cols[k].append(d[k])
    out = {k: (np.concatenate(v) if v else None) for k, v in cols.items()}
    out.update(load_manifest(manifest))
    return out


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
_AMAP = None


def _init_worker(aliases):
    global _AMAP
    import ertext
    _AMAP = ertext.load_aliases(aliases)


def _text_chunk(chunk):
    """chunk: list of ((name, addr), [(name, addr), ...]) -> float32 matrix."""
    rows = []
    cache = {}
    for (an, aa), cands in chunk:
        a = F.RecordView(an, aa, _AMAP)
        for bn, ba in cands:
            key = (bn, ba)
            b = cache.get(key)
            if b is None:
                b = cache[key] = F.RecordView(bn, ba, _AMAP)
            rows.append(F.text_features(a, b))
    return np.asarray(rows, np.float32).reshape(-1, len(F.TEXT_FEATURE_NAMES))


def prefilter_proba(bundle, ctx):
    pre = bundle.get("prefilter")
    if pre is None:
        return None
    p = np.empty(len(ctx), np.float32)
    for i in range(0, len(ctx), 2_000_000):
        p[i:i + 2_000_000] = pre["model"].predict_proba(ctx[i:i + 2_000_000])[:, 1]
    return p


class TextStore:
    """Every record's "name<TAB>address" in one bytes blob, looked up by code.

    ~1 GB for the full 12M-record train or test set, against ~3 GB for the
    equivalent dict of Python strings.
    """

    def __init__(self, data_dir, prefix, which=(1, 2, 3)):
        codes, offs, ctry_idx = array("q"), array("q", [0]), array("b")
        blob = bytearray()
        self.countries = []
        cmap = {}
        for i in which:
            for rid, name, addr, ctry in iter_source(
                    os.path.join(data_dir, f"{prefix}_source{i}.tsv")):
                codes.append(encode_id(rid))
                blob += f"{name}\t{addr}".encode("utf-8")
                offs.append(len(blob))
                c = cmap.get(ctry)
                if c is None:
                    c = cmap[ctry] = len(self.countries)
                    self.countries.append(ctry)
                ctry_idx.append(c)
        codes = np.frombuffer(codes, np.int64)
        self.order = np.argsort(codes)
        self.sorted_codes = codes[self.order]
        self.offs = np.frombuffer(offs, np.int64)
        self.ctry = np.frombuffer(ctry_idx, np.int8)
        self.blob = bytes(blob)

    def _index(self, code):
        return self.order[np.searchsorted(self.sorted_codes, code)]

    def get(self, code):
        i = self._index(code)
        name, _, addr = self.blob[self.offs[i]:self.offs[i + 1]].decode(
            "utf-8").partition("\t")
        return name, addr

    def country(self, codes):
        return np.array(self.countries, dtype=object)[
            self.ctry[self._index(np.asarray(codes))]].astype(str)


def build(data_dir, prefix, cand_scores_path, out_npz, gt_path=None, max_s1=None,
          aliases=None, workers=None, prefilter=None, part_pairs=2_000_000,
          seed=0, chunk_pairs=4000):
    t0 = time.time()
    meta = load_meta(cand_scores_path)
    parts = score_parts(cand_scores_path)
    entities = np.sort(meta["stored"])
    print(f"[build] {len(parts)} sidecar part(s), {len(entities)} stored of "
          f"{len(meta['queried'])} queried entities")
    if max_s1 and max_s1 < len(entities):
        rng = np.random.default_rng(seed)
        entities = np.sort(rng.choice(entities, size=max_s1, replace=False))

    pre = None
    if prefilter:
        with open(prefilter, "rb") as f:
            pre = pickle.load(f)
        if pre.get("prefilter") is None:
            pre = None

    store = TextStore(data_dir, prefix)
    ent_country = store.country(entities)
    print(f"[build] {len(entities)} entities, texts loaded ({time.time() - t0:.0f}s)")

    gt = load_ground_truth_codes(gt_path, set(entities.tolist())) if gt_path else None

    base = out_npz[:-4] if out_npz.endswith(".npz") else out_npz
    for old in part_paths(out_npz):
        os.remove(old)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    n_part = n_block = n_kept = n_pos = 0
    ctx_pool = mp.get_context("spawn")
    with ctx_pool.Pool(workers, initializer=_init_worker, initargs=(aliases,)) as pool:
        for i_part, part in enumerate(parts):
            # ---- one sidecar part: whole entity lists ----------------------
            d = np.load(part)
            sc = {k: d[k] for k in d.files}
            rows = np.flatnonzero(np.isin(sc["s1"], entities))
            if len(rows) == 0:
                continue
            cb = F.ContextBuilder(sc, meta)
            s1_all, cand_all = sc["s1"], sc["cand"]
            n_block += len(rows)
            ctx = cb.features(rows)
            if pre is not None:
                keep = prefilter_proba(pre, ctx) >= pre["prefilter"]["threshold"]
                rows, ctx = rows[keep], ctx[keep]
            left, right = s1_all[rows], cand_all[rows]
            n_kept += len(rows)

            # ---- text features, in parallel --------------------------------
            chunks, cur, cur_n = [], [], 0
            starts = np.flatnonzero(np.r_[True, left[1:] != left[:-1]]) \
                if len(left) else np.zeros(0, np.int64)
            for lo, hi in zip(starts, np.r_[starts[1:], len(left)]):
                cur.append((store.get(left[lo]),
                            [store.get(c) for c in right[lo:hi]]))
                cur_n += hi - lo
                if cur_n >= chunk_pairs:
                    chunks.append(cur)
                    cur, cur_n = [], 0
            if cur:
                chunks.append(cur)
            T = [t for t in pool.imap(_text_chunk, chunks, chunksize=1)]
            T = np.concatenate(T) if T else \
                np.zeros((0, len(F.TEXT_FEATURE_NAMES)), np.float32)
            assert len(T) == len(left)
            X = np.hstack([T, ctx])
            assert X.shape[1] == len(F.FEATURE_NAMES), (
                "text + context widths do not match FEATURE_NAMES - keep aligned")
            y = np.zeros(0, np.int8)
            if gt is not None:
                y = np.fromiter((c in gt.get(l, ()) for l, c in
                                 zip(left.tolist(), right.tolist())), np.int8,
                                len(left))
                n_pos += int(y.sum())
            np.savez(f"{base}.part{n_part:03d}.npz", X=X, y=y, left=left,
                     right=right)
            n_part += 1
            print(f"[build] part {i_part + 1}/{len(parts)}: {n_kept}/{n_block} "
                  f"pairs kept ({time.time() - t0:.0f}s)", flush=True)
            del sc, cb, d

    np.savez(out_npz, entities=entities, entity_country=ent_country,
             names=np.asarray(F.FEATURE_NAMES), n_parts=n_part)
    print(f"[build] {n_kept} pairs, {len(F.FEATURE_NAMES)} features, "
          f"{n_part} part(s) -> {out_npz} ({time.time() - t0:.0f}s)")
    if gt is not None:
        n_true = sum(len(v) for v in gt.values())
        print(f"[build] positives {n_pos}; recall ceiling {n_pos / max(n_true, 1):.4f}"
              f"{' (after prefilter)' if pre is not None else ''}")


# --------------------------------------------------------------------------- #
# metric + decision
# --------------------------------------------------------------------------- #
def macro_f05(pred: dict, gt: dict) -> float:
    beta2 = 0.25
    total = 0.0
    for s1, truth in gt.items():
        got = pred.get(s1, set())
        if not truth and not got:
            total += 1.0
            continue
        if not truth or not got:
            continue
        tp = len(got & truth)
        if tp == 0:
            continue
        p, r = tp / len(got), tp / len(truth)
        total += (1 + beta2) * p * r / (beta2 * p + r)
    return total / max(len(gt), 1)


def assign(left, right, prob, threshold, one_to_one=True):
    """Threshold + optional global one-to-one constraint on the S2/S3 side.

    Every Source-2/3 record in the training ground truth belongs to exactly one
    Source-1 entity, so a target claimed by two entities is necessarily one
    false positive. Keeping only the highest-probability claim is a free
    precision win under F_0.5. ``threshold`` may be a scalar or per-pair array.
    """
    keep = prob >= threshold
    idx = np.flatnonzero(keep)
    if one_to_one and len(idx):
        idx = idx[np.argsort(-prob[idx], kind="stable")]
        _, first = np.unique(right[idx], return_index=True)
        idx = idx[np.sort(first)]
    pred = {}
    for l, r in zip(left[idx].tolist(), right[idx].tolist()):
        pred.setdefault(l, set()).add(r)
    return pred


def pair_country(left, entities, ent_country):
    order = np.argsort(entities)
    pos = np.searchsorted(entities[order], left)
    return ent_country[order][pos]


def thresholds_for(bundle, countries):
    thr = bundle["thresholds"]
    default = bundle["default_threshold"]
    return np.array([thr.get(c, default) for c in countries], np.float32)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def _make_clf(n_estimators=500):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=n_estimators, learning_rate=0.06,
                              num_leaves=63, min_child_samples=50, subsample=0.9,
                              subsample_freq=1, colsample_bytree=0.9, n_jobs=-1,
                              verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier as HGB
        return HGB(max_iter=n_estimators, learning_rate=0.08, max_leaf_nodes=63,
                   min_samples_leaf=50, early_stopping=False)


def val_split(entities, val_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    return set(rng.choice(entities, size=int(len(entities) * val_frac),
                          replace=False).tolist())


GRID = np.concatenate([np.arange(0.05, 0.90, 0.025), np.arange(0.90, 0.995, 0.01)])


def sweep(left, right, prob, gt_val, one_to_one, grid=GRID):
    scores = [macro_f05(assign(left, right, prob, t, one_to_one), gt_val)
              for t in grid]
    j = int(np.argmax(scores))
    return float(grid[j]), scores[j], scores


def train(pairs_npz, gt_path, model_out, val_frac=0.25, one_to_one=True,
          refit=False, n_estimators=500, prefilter_recall=0.999,
          unseen="max", drop=()):
    P = load_pairs(pairs_npz)
    X, y, left, right = P["X"], P["y"].astype(int), P["left"], P["right"]
    entities, ent_country = P["entities"], P["entity_country"]
    names = list(P["names"])
    if drop:
        # ablation only: zeroing keeps column positions (and the prefilter's
        # context slice) intact, and a constant column is never split on
        unknown = set(drop) - set(names)
        assert not unknown, f"unknown features {unknown}"
        for f in drop:
            X[:, names.index(f)] = 0.0
        print(f"[train] ablation: dropped {sorted(drop)}")
    gt = load_ground_truth_codes(gt_path, set(entities.tolist()))
    for e in entities.tolist():  # entities with no truth row are singletons
        gt.setdefault(e, set())

    val_ids = val_split(entities, val_frac)
    is_val = np.isin(left, np.fromiter(val_ids, np.int64))
    country = pair_country(left, entities, ent_country)

    clf = _make_clf(n_estimators)
    clf.fit(X[~is_val], y[~is_val])
    print(f"[train] fitted {type(clf).__name__} on {int((~is_val).sum())} pairs")

    prob = clf.predict_proba(X[is_val])[:, 1]
    lv, rv, cv = left[is_val], right[is_val], country[is_val]
    gt_val = {k: v for k, v in gt.items() if k in val_ids}

    # ---- thresholds: global, then per country ------------------------------
    t_glob, s_glob, curve = sweep(lv, rv, prob, gt_val, one_to_one)
    for t, s in zip(GRID, curve):
        if abs(t - t_glob) < 0.101:
            print(f"[train]   threshold {t:.3f} -> macro F0.5 {s:.4f}")
    print(f"[train] global: best macro F0.5 {s_glob:.4f} at threshold {t_glob:.3f}")

    ent_c = dict(zip(entities.tolist(), ent_country.tolist()))
    thresholds = {}
    for c in sorted(set(ent_country.tolist())):
        m = cv == c
        gt_c = {k: v for k, v in gt_val.items() if ent_c[k] == c}
        t_c, s_c, _ = sweep(lv[m], rv[m], prob[m], gt_c, one_to_one)
        thresholds[c] = t_c
        print(f"[train] {c}: {len(gt_c)} val entities, best {s_c:.4f} at {t_c:.3f} "
              f"(global threshold gives "
              f"{macro_f05(assign(lv[m], rv[m], prob[m], t_glob, one_to_one), gt_c):.4f})")
    thr_pair = np.array([thresholds[c] for c in cv], np.float32)
    s_pc = macro_f05(assign(lv, rv, prob, thr_pair, one_to_one), gt_val)
    print(f"[train] per-country thresholds -> macro F0.5 {s_pc:.4f} "
          f"(global {s_glob:.4f})")
    # A country never seen in training (France at test time) cannot be
    # validated, so it gets the most conservative learned threshold.
    default = max(thresholds.values()) if unseen == "max" else t_glob

    # ---- oracle over candidates --------------------------------------------
    oracle = {}
    for l, r in zip(lv[y[is_val] == 1].tolist(), rv[y[is_val] == 1].tolist()):
        oracle.setdefault(l, set()).add(r)
    print(f"[train] oracle over candidates {macro_f05(oracle, gt_val):.4f}")

    # ---- stage-1 prefilter on context features only ------------------------
    # Cheap (no text needed), so the test build can drop most blocking pairs
    # before the expensive text features. Threshold keeps `prefilter_recall`
    # of the validation pairs the full model accepts.
    n_text = len(F.TEXT_FEATURE_NAMES)
    pre = _make_clf(200)
    pre.fit(X[~is_val, n_text:], y[~is_val])
    p1 = pre.predict_proba(X[is_val, n_text:])[:, 1]
    accepted = (prob >= thr_pair) & (y[is_val] == 1)
    t1 = float(np.quantile(p1[accepted], 1 - prefilter_recall)) if accepted.any() else 0.0
    kept = p1 >= t1
    s_pre = macro_f05(assign(lv[kept], rv[kept], prob[kept], thr_pair[kept],
                             one_to_one), gt_val)
    print(f"[train] prefilter threshold {t1:.5f} keeps {kept.mean():.3f} of pairs; "
          f"macro F0.5 after prefilter {s_pre:.4f}")

    if refit:
        clf = _make_clf(n_estimators)
        clf.fit(X, y)
        pre = _make_clf(200)
        pre.fit(X[:, n_text:], y)
        print(f"[train] refitted on all {len(y)} pairs")

    with open(model_out, "wb") as f:
        pickle.dump({"model": clf, "threshold": t_glob, "thresholds": thresholds,
                     "default_threshold": default, "one_to_one": one_to_one,
                     "features": names,
                     "prefilter": {"model": pre, "threshold": t1}}, f)
    print(f"[train] saved {model_out}; thresholds {thresholds}, "
          f"unseen countries -> {default:.3f}")
    return {"val": s_pc, "val_global": s_glob,
            "oracle": macro_f05(oracle, gt_val)}


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
def score_pairs(pairs_npz, clf):
    L, R, Pr = [], [], []
    for p in part_paths(pairs_npz):
        d = np.load(p)
        X = d["X"]
        prob = np.empty(len(X), np.float32)
        for i in range(0, len(X), 500_000):
            prob[i:i + 500_000] = clf.predict_proba(X[i:i + 500_000])[:, 1]
        L.append(d["left"]), R.append(d["right"]), Pr.append(prob)
    return np.concatenate(L), np.concatenate(R), np.concatenate(Pr)


def predict(pairs_npz, model_path, s1_path, out_path, threshold=None,
            candidates_out=None, confident_out=None, confident_threshold=0.98):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    man = load_manifest(pairs_npz)
    left, right, prob = score_pairs(pairs_npz, bundle["model"])
    country = pair_country(left, man["entities"], man["entity_country"])
    if threshold is not None:
        t = np.full(len(left), threshold, np.float32)
    elif "thresholds" in bundle:
        t = thresholds_for(bundle, country)
        for c in sorted(set(man["entity_country"].tolist())):
            src = "learned" if c in bundle["thresholds"] else "UNSEEN -> conservative"
            print(f"[predict] {c}: threshold "
                  f"{bundle['thresholds'].get(c, bundle['default_threshold']):.3f} ({src})")
    else:
        t = np.full(len(left), bundle["threshold"], np.float32)
    pred = assign(left, right, prob, t, bundle["one_to_one"])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        n_empty = 0
        for rid, _, _, _ in iter_source(s1_path):
            ids = sorted(decode_id(x) for x in pred.get(encode_id(rid), ()))
            n_empty += not ids
            f.write(f"{rid}\t{','.join(ids)}\n")
    print(f"[predict] {len(left)} scored pairs -> {out_path} "
          f"({n_empty} entities predicted as singletons)")

    if confident_out:
        # Pseudo ground truth for `aliases.py --ground-truth`: only links the
        # model is nearly sure of, so the lexicon can be re-mined on unlabelled
        # data (this is how aliases for a country absent from training appear).
        conf = assign(left, right, prob, confident_threshold, bundle["one_to_one"])
        with open(confident_out, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for l, rs in conf.items():
                f.write(f"{decode_id(l)}\t{','.join(decode_id(r) for r in rs)}\n")
        print(f"[predict] {sum(len(v) for v in conf.values())} links with "
              f"p >= {confident_threshold} -> {confident_out}")

    if candidates_out:
        # The candidate set the classifier actually scored (post-prefilter).
        cands = {}
        starts = np.flatnonzero(np.r_[True, left[1:] != left[:-1]])
        for lo, hi in zip(starts, np.r_[starts[1:], len(left)]):
            cands[int(left[lo])] = right[lo:hi]
        write_candidate_tsv(candidates_out,
                            [r for r, _, _, _ in iter_source(s1_path)], cands)
        print(f"[predict] candidate set -> {candidates_out}")


def evaluate(results_path, gt_path):
    gt = load_ground_truth(gt_path)
    pred = {}
    with open(results_path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            pred[s1] = {x for x in rest.split(",") if x}
    gt = {k: v for k, v in gt.items() if k in pred}
    for k in pred:
        gt.setdefault(k, set())
    print(f"[evaluate] macro F0.5 = {macro_f05(pred, gt):.4f} over {len(gt)} entities")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--data-dir", required=True)
    b.add_argument("--prefix", default="train")
    b.add_argument("--cand-scores", required=True,
                   help="*_scores.npz sidecar written by blocking.py")
    b.add_argument("--out", required=True)
    b.add_argument("--ground-truth")
    b.add_argument("--max-s1", type=int,
                   help="sample N queried entities (context still uses all)")
    b.add_argument("--aliases")
    b.add_argument("--workers", type=int)
    b.add_argument("--prefilter", help="model.pkl whose stage-1 prefilter to apply")
    b.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("train")
    t.add_argument("--pairs", required=True)
    t.add_argument("--ground-truth", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--no-one-to-one", action="store_true")
    t.add_argument("--refit", action="store_true",
                   help="after validation, refit on all pairs (final model)")
    t.add_argument("--n-estimators", type=int, default=500)
    t.add_argument("--unseen", choices=["max", "global"], default="max",
                   help="threshold for countries absent from training")
    t.add_argument("--drop-features", nargs="+", default=[],
                   help="ablation: neutralise these features")

    p = sub.add_parser("predict")
    p.add_argument("--pairs", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--source1", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float)
    p.add_argument("--candidates-out",
                   help="also write the scored candidate set in submission format")
    p.add_argument("--confident-out",
                   help="also write high-confidence links (pseudo ground truth "
                        "for re-mining aliases on unlabelled data)")
    p.add_argument("--confident-threshold", type=float, default=0.98)

    e = sub.add_parser("evaluate")
    e.add_argument("--results", required=True)
    e.add_argument("--ground-truth", required=True)

    a = ap.parse_args()
    if a.cmd == "build":
        build(a.data_dir, a.prefix, a.cand_scores, a.out, a.ground_truth,
              a.max_s1, a.aliases, a.workers, a.prefilter, seed=a.seed)
    elif a.cmd == "train":
        train(a.pairs, a.ground_truth, a.out, one_to_one=not a.no_one_to_one,
              refit=a.refit, n_estimators=a.n_estimators, unseen=a.unseen,
              drop=a.drop_features)
    elif a.cmd == "predict":
        predict(a.pairs, a.model, a.source1, a.out, a.threshold, a.candidates_out,
                a.confident_out, a.confident_threshold)
    else:
        evaluate(a.results, a.ground_truth)
