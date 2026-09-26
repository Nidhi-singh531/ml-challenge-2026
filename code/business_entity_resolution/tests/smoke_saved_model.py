"""Exercise selected-model prediction and organizer validation on a small fixture.

This checks submission plumbing, not held-out accuracy or the real submission.
Run from the workspace root after the classifier experiment's final refit.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
sys.path.insert(0, str(ROOT / "utils"))
import pipeline as P
from blocking import decode_id
from validate_submission import validate


def main():
    source = "work/dev/pairs200k_filtered.npz"
    man = P.load_manifest(source)
    with np.load(P.part_paths(source)[0]) as data:
        selected = np.unique(data["left"])[:100]
        mask = np.isin(data["left"], selected)
        arrays = {k: data[k][mask] for k in ("X", "y", "left", "right")}
    # One additional real entity with no rows in this fixture tests empty output.
    no_pairs = next(e for e in man["entities"] if e not in selected)
    selected = np.sort(np.r_[selected, no_pairs])
    with tempfile.TemporaryDirectory(dir=ROOT / "work/dev") as tmp:
        folder = Path(tmp)
        pairs = folder / "pairs.npz"
        np.savez(folder / "pairs.part000.npz", **arrays)
        countries = P.pair_country(selected, man["entities"], man["entity_country"])
        np.savez(pairs, entities=selected, entity_country=countries,
                 names=man["names"], prefilter_sha256=man["prefilter_sha256"])
        s1 = folder / "test_source1.tsv"
        with s1.open("w", encoding="utf-8") as f:
            f.write("entity_id\tname\taddress\tcountry\n")
            for e, c in zip(selected, countries):
                f.write(f"{decode_id(e)}\tsmoke fixture\t\t{c}\n")
        results, candidates = folder / "matching_results.tsv", folder / "candidate_pairs.tsv"
        P.predict(str(pairs), "work/dev/model_optimized.pkl", str(s1), str(results),
                  candidates_out=str(candidates))
        errors, warnings = validate(str(results), str(candidates), str(folder), check_ids=False)
        if errors:
            raise AssertionError(errors)
        rows = results.read_text(encoding="utf-8").splitlines()[1:]
        assert len(rows) == len(selected)
        got = dict(row.split("\t", 1) for row in rows)
        assert got[decode_id(no_pairs)] == ""
        targets = [t for ids in got.values() for t in ids.split(",") if t]
        assert len(targets) == len(set(targets)), "target assigned more than once"
        print(f"PASS: {len(selected)} entities, {len(arrays['left'])} pairs; organizer validator; "
              "empty-entity output; global target uniqueness. This is a fixture, not the test submission.")
        for warning in warnings:
            print("validator warning:", warning)


if __name__ == "__main__":
    main()
