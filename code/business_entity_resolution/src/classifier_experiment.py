"""Reproducible fixed-holdout classifier experiment using frozen candidates."""
import argparse
import json
import os
import pickle
import time

import numpy as np

import ambiguity
import features as F
import pipeline as P


ROOT = "work/dev"
BASELINE = f"{ROOT}/model40k.pkl"
GT = "dataset/train/train_ground_truth.tsv"
FROZEN = ["code/business_entity_resolution/src/blocking.py",
          "code/business_entity_resolution/src/ertext.py", "work/aliases_full.tsv"]


def compare_models():
    """Report identical holdout cohorts and paired bootstrap score differences."""
    pairs = f"{ROOT}/pairs200k_ambiguity.npz"
    man = P.load_manifest(pairs)
    val = P.val_split(P.load_manifest(f"{ROOT}/pairs40k.npz")["entities"])
    xs, ls, rs = [], [], []
    for path in P.part_paths(pairs):
        with np.load(path) as d:
            mask = np.isin(d["left"], list(val))
            xs.append(d["X"][mask]); ls.append(d["left"][mask]); rs.append(d["right"][mask])
    x, left, right = np.concatenate(xs), np.concatenate(ls), np.concatenate(rs)
    gt = P.load_ground_truth_codes(GT, val)
    gt = {e: gt.get(e, set()) for e in sorted(val)}
    names = list(man["names"])
    countries = P.pair_country(left, man["entities"], man["entity_country"])
    repeated = set(left[x[:, names.index("s1_core_name_frequency")] > np.log(2) + 1e-6].tolist())
    empty = set(left[x[:, names.index("addr_empty_cand")] > 0].tolist())
    cohorts = {"all": set(val), "shared_core_name": repeated, "empty_address_candidate": empty}
    # Thresholds originally tuned before filtering may no longer be optimal.
    # Include this cheap alternative so extra training must beat it as well.
    with open(BASELINE, "rb") as f:
        calibrated = pickle.load(f)
    cols = [names.index(n) for n in calibrated["features"]]
    prob = calibrated["model"].predict_proba(x[:, cols])[:, 1]
    global_thr, _, _ = P.sweep(left, right, prob, gt, True)
    ent_country = dict(zip(man["entities"].tolist(), man["entity_country"].tolist()))
    thresholds = {}
    for country in sorted(set(countries.tolist())):
        mask = countries == country
        country_gt = {e: truth for e, truth in gt.items() if ent_country[e] == country}
        thresholds[country], _, _ = P.sweep(left[mask], right[mask], prob[mask], country_gt, True)
    calibrated.update(threshold=global_thr, thresholds=thresholds,
                      default_threshold=max(thresholds.values()),
                      validation_ids=np.asarray(sorted(val), np.int64), refit=False,
                      fixed_prefilter_sha256=P.file_sha256(BASELINE))
    calibrated_path = f"{ROOT}/model40k_calibrated.pkl"
    with open(calibrated_path, "wb") as f:
        pickle.dump(calibrated, f)
    report, baseline_scores = {}, None
    for label, model in [("baseline", BASELINE),
                         ("baseline_calibrated", calibrated_path),
                         ("larger_baseline", f"{ROOT}/model200k_larger_baseline.pkl"),
                         ("ambiguity", f"{ROOT}/model200k_ambiguity.pkl")]:
        with open(model, "rb") as f:
            bundle = pickle.load(f)
        cols = [names.index(n) for n in bundle["features"]]
        prob = bundle["model"].predict_proba(x[:, cols])[:, 1]
        pred = P.assign(left, right, prob, P.thresholds_for(bundle, countries), bundle["one_to_one"])
        scores = np.array([P.macro_f05(pred, {e: truth}) for e, truth in gt.items()])
        details = {}
        for cohort, ids in cohorts.items():
            truth = {e: gt[e] for e in ids}
            details[cohort] = {"entities": len(ids), "macro_f05": P.macro_f05(pred, truth),
                               "false_links": sum(len(pred.get(e, set()) - t) for e, t in truth.items()),
                               "missed_links": sum(len(t - pred.get(e, set())) for e, t in truth.items()),
                               "false_singleton_merges": sum(not t and bool(pred.get(e)) for e, t in truth.items())}
        if baseline_scores is None:
            baseline_scores = scores
        else:
            delta = scores - baseline_scores
            rng = np.random.default_rng(0)
            means = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(1000)]
            details["paired_delta_vs_baseline"] = float(delta.mean())
            details["paired_bootstrap_95pct"] = np.quantile(means, [.025, .975]).tolist()
        report[label] = details
    with open(f"{ROOT}/classifier_comparison.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return report


def audit_baseline():
    man = P.load_manifest(f"{ROOT}/pairs40k.npz")
    val = P.val_split(man["entities"])
    with open(BASELINE, "rb") as f:
        bundle = pickle.load(f)
    arrays = {k: [] for k in ("left", "right", "prob", "keep", "empty")}
    for path in P.part_paths(f"{ROOT}/pairs40k.npz"):
        with np.load(path) as d:
            mask = np.isin(d["left"], list(val))
            x = d["X"][mask]
            arrays["left"].append(d["left"][mask])
            arrays["right"].append(d["right"][mask])
            arrays["prob"].append(bundle["model"].predict_proba(x)[:, 1])
            cols = [list(man["names"]).index(n) for n in F.CONTEXT_FEATURE_NAMES]
            arrays["keep"].append(P.prefilter_proba(bundle, x[:, cols]) >= bundle["prefilter"]["threshold"])
            arrays["empty"].append(x[:, list(man["names"]).index("addr_empty_cand")] > 0)
    a = {k: np.concatenate(v) for k, v in arrays.items()}
    gt = P.load_ground_truth_codes(GT, val)
    gt = {e: gt.get(e, set()) for e in val}
    threshold = P.thresholds_for(bundle, P.pair_country(a["left"], man["entities"], man["entity_country"]))
    metrics = {}
    for label, mask in [("unfiltered", np.ones(len(a["left"]), bool)), ("prefiltered", a["keep"])]:
        pred = P.assign(a["left"][mask], a["right"][mask], a["prob"][mask], threshold[mask])
        empty_ids = set(a["left"][a["empty"]].tolist())
        metrics[label] = {
            "macro_f05": P.macro_f05(pred, gt),
            "false_singleton_merges": sum(not truth and bool(pred.get(e)) for e, truth in gt.items()),
            "false_links": sum(len(pred.get(e, set()) - truth) for e, truth in gt.items()),
            "missed_links": sum(len(truth - pred.get(e, set())) for e, truth in gt.items()),
            "entities_with_empty_candidate_f05": P.macro_f05(pred, {e: gt[e] for e in empty_ids}),
            "empty_candidate_entities": len(empty_ids),
        }
    print("[audit] " + json.dumps(metrics), flush=True)
    return metrics


def project_base(source, destination):
    """Drop optional columns on disk for an exact original-schema comparison."""
    man = P.load_manifest(source)
    columns = [list(man["names"]).index(n) for n in F.FEATURE_NAMES]
    for i, path in enumerate(P.part_paths(source)):
        with np.load(path) as d:
            np.savez(f"{destination[:-4]}.part{i:03d}.npz", X=d["X"][:, columns],
                     y=d["y"], left=d["left"], right=d["right"])
    man["names"] = np.asarray(F.FEATURE_NAMES)
    np.savez(destination, **man)


def finalize(label):
    """Refit the selected classifier, preserving measured thresholds and filter."""
    from sklearn.base import clone
    stem = "filtered" if label == "larger_baseline" else "ambiguity"
    pairs = f"{ROOT}/pairs200k_{stem}.npz"
    model = f"{ROOT}/model200k_{label}.pkl"
    with open(f"{ROOT}/classifier_results.json", encoding="utf-8") as f:
        experiment = json.load(f)
    assert all(P.file_sha256(p) == sha for p, sha in experiment["frozen_sha256"].items())
    with open(model, "rb") as f:
        bundle = pickle.load(f)
    data = P.load_pairs(pairs)
    assert list(data["names"]) == list(bundle["features"])
    assert str(data["prefilter_sha256"]) == bundle["fixed_prefilter_sha256"]
    clf = clone(bundle["model"])
    print(f"[finalize] refitting {label} on {len(data['entities'])} entities / {len(data['y'])} pairs", flush=True)
    clf.fit(data["X"], data["y"])
    bundle.update(model=clf, refit=True, selected_experiment=label,
                  validation_metrics_before_refit=experiment[label],
                  training_entities=len(data["entities"]))
    out = f"{ROOT}/model_optimized.pkl"
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    selection = {"experiment": label, "model": out, "model_sha256": P.file_sha256(out),
                 "prefilter": BASELINE, "ambiguity_features": label == "ambiguity",
                 "validation_before_refit": experiment[label], "refit_entities": len(data["entities"])}
    with open(f"{ROOT}/classifier_selection.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    with open("work/runlog.txt", "a", encoding="utf-8") as f:
        f.write(f"\n2026-09-27 | all 200k entities | refit selected {label}; baseline prefilter frozen | 50+25 | "
                f"{experiment[label]['val_prefilter']:.6f} BEFORE refit | {experiment[label]['oracle']:.6f} | {out}\n")
    print("[finalize] " + json.dumps(selection), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--finalize", choices=["larger_baseline", "ambiguity"])
    args = ap.parse_args()
    if args.compare_only:
        compare_models()
        return
    if args.finalize:
        finalize(args.finalize)
        return
    start = time.time()
    frozen = {p: P.file_sha256(p) for p in FROZEN + [BASELINE]}
    results = {"frozen_sha256": frozen, "baseline": audit_baseline()}
    report = f"{ROOT}/classifier_results.json"

    def save():
        with open(report, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    save()
    if args.audit_only:
        return
    enhanced = f"{ROOT}/pairs200k_ambiguity.npz"
    base = f"{ROOT}/pairs200k_filtered.npz"
    if not args.skip_build:
        P.build("dataset/train", "train", f"{ROOT}/candidate_pairs_scores", enhanced,
                GT, aliases="work/aliases_full.tsv", workers=3,
                prefilter=BASELINE, ambiguity_features=True)
    project_base(enhanced, base)
    for label, pairs in [("larger_baseline", base), ("ambiguity", enhanced)]:
        results[label] = P.train(pairs, GT, f"{ROOT}/model200k_{label}.pkl",
                                 validation_from=f"{ROOT}/pairs40k.npz",
                                 fixed_prefilter=BASELINE)
        save()
        with open("work/runlog.txt", "a", encoding="utf-8") as f:
            f.write(f"\n2026-09-27 | 200k stored, original 10k holdout | {label}, frozen baseline prefilter | 50+25 | "
                    f"{results[label]['val_prefilter']:.6f} | {results[label]['oracle']:.6f} | classifier experiment\n")
    assert all(P.file_sha256(p) == sha for p, sha in frozen.items()), "frozen input changed"
    results["seconds"] = time.time() - start
    save()
    print("[experiment] complete " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
