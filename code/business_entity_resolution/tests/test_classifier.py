"""Regression checks for frozen-schema compatibility and new evidence."""
import sys
import tempfile
import pickle
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ambiguity
import features as F
import pipeline as P


class ClassifierTests(unittest.TestCase):
    def test_missing_address_is_not_conflict(self):
        a = F.RecordView("Example", "12 Oak Street")
        empty = ambiguity.pair_features(a, F.RecordView("Example", "<NULL>"), 1, 1)
        conflict = ambiguity.pair_features(a, F.RecordView("Example", "99 Elm Road"), 1, 1)
        self.assertEqual(empty[-1], 0)
        self.assertEqual(empty[4], 0)
        self.assertEqual(conflict[-1], 1)
        self.assertGreater(conflict[-2], 0)

    def test_counts_partition_country_and_ignore_order_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s1.tsv"
            p.write_text("entity_id\tname\taddress\tcountry\n"
                         "S1-1\tBlue Oak Ltd\t\tA\n"
                         "S1-2\tOak Blue\t\tA\n"
                         "S1-3\tBlue Oak\t\tB\n", encoding="utf-8")
            counts = ambiguity.NameCounts(str(p), {})
            self.assertEqual(counts.get("Blue Oak", "A"), 2)
            self.assertEqual(counts.get("Blue Oak", "B"), 1)
            self.assertEqual(counts.get("", "A"), 0)

    def test_optional_features_preserve_base_text_columns(self):
        P._AMAP = {}
        base = P._text_chunk([(("Blue Oak", "12 Road"), [("Oak Blue", "12 Rd")])])
        enhanced = P._text_chunk([(("Blue Oak", "12 Road", 2), [("Oak Blue", "12 Rd", 2)])])
        np.testing.assert_array_equal(base, enhanced[:, :len(F.TEXT_FEATURE_NAMES)])
        self.assertEqual(enhanced.shape[1], len(F.TEXT_FEATURE_NAMES) + len(ambiguity.FEATURE_NAMES))

    def test_filter_before_assignment_allows_runner_up(self):
        left, right, prob = np.array([1, 2]), np.array([3, 3]), np.array([.9, .8])
        keep = np.array([False, True])
        self.assertEqual(P.assign(left[keep], right[keep], prob[keep], .5), {2: {3}})

    def test_predict_rejects_schema_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, pairs = Path(tmp) / "model.pkl", Path(tmp) / "pairs.npz"
            with model.open("wb") as f:
                pickle.dump({"features": ["wrong"]}, f)
            np.savez(pairs, names=np.asarray(F.FEATURE_NAMES))
            with self.assertRaisesRegex(ValueError, "schema differs"):
                P.predict(str(pairs), str(model), "unused", "unused")

    def test_predict_rejects_wrong_prefilter_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, pairs = Path(tmp) / "model.pkl", Path(tmp) / "pairs.npz"
            with model.open("wb") as f:
                pickle.dump({"features": F.FEATURE_NAMES, "fixed_prefilter_sha256": "expected"}, f)
            np.savez(pairs, names=np.asarray(F.FEATURE_NAMES), prefilter_sha256="different")
            with self.assertRaisesRegex(ValueError, "fixed prefilter"):
                P.predict(str(pairs), str(model), "unused", "unused")


if __name__ == "__main__":
    unittest.main()
