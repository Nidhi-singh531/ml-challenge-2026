"""Checks for stage-2 decoding and list features."""
import itertools
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import stage2 as S


def brute_ef(q, k, lam, umax=12):
    total = 0.0
    for bits in itertools.product([0, 1], repeat=len(q)):
        pr = math.prod(x if b else 1 - x for x, b in zip(q, bits))
        for m in range(umax + 1):
            pm = math.exp(-lam) * lam ** m / math.factorial(m) if lam > 0 else float(m == 0)
            t, u = sum(bits[:k]), sum(bits[k:]) + m
            f = (1 if t + u == 0 else 0) if k == 0 else (1.25 * t / (0.25 * (t + u) + k))
            total += pr * pm * f
    return total


class Stage2Tests(unittest.TestCase):
    def test_best_k_matches_brute_force(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = np.sort(rng.random(rng.integers(1, 6)))[::-1]
            lam = float(rng.choice([0.0, 0.1, 0.4]))
            vals = [brute_ef(q, k, lam) for k in range(len(q) + 1)]
            k, v = S.best_k(q, lam)
            self.assertAlmostEqual(v, max(vals), places=6)
            self.assertAlmostEqual(vals[k], max(vals), places=6)

    def test_confident_singleton_predicts_empty(self):
        self.assertEqual(S.best_k(np.array([0.1, 0.05]), 0.0)[0], 0)
        self.assertEqual(S.best_k(np.array([0.99, 0.97, 0.02]), 0.0)[0], 2)

    def test_decode_is_one_to_one(self):
        left = np.array([1, 2, 2]); right = np.array([9, 9, 8]); q = np.array([.9, .95, .9])
        pred = S.decode_ef(left, right, q, lambda e: 0.0)
        claims = [r for rs in pred.values() for r in rs]
        self.assertEqual(len(claims), len(set(claims)))
        self.assertIn(9, pred[2])

    def test_list_features_keep_input_order(self):
        left = np.array([5, 3, 5, 3]); right = np.array([2_000_000_001, 3_000_000_001,
                                                         3_000_000_002, 2_000_000_002])
        p = np.array([0.2, 0.9, 0.8, 0.4], np.float32)
        f = S.list_features(left, right, p)
        np.testing.assert_allclose(f[:, 0], p)
        self.assertEqual(f[2, 1], 0)  # 0.8 is entity 5's top
        self.assertEqual(f[0, 1], 1)
        self.assertAlmostEqual(f[1, 2], 0.5, places=6)  # 0.9 - 0.4
        self.assertTrue((f[:, 8] == 1).all())  # each is best in its own source

    def test_oof_never_scores_own_fold(self):
        rng = np.random.default_rng(0)
        left = np.repeat(np.arange(100), 3); X = rng.random((300, 4)).astype(np.float32)
        y = (rng.random(300) < 0.3).astype(int)
        p, pf = S.oof_stage1(X, y, left, np.arange(80), n_folds=4)
        self.assertTrue(np.isnan(p[left >= 80]).all())
        for e in range(80):
            self.assertEqual(len(set(pf[left == e].tolist())), 1)


class ExpandTests(unittest.TestCase):
    def test_generate_finds_rare_key_and_skips_common(self):
        import expand as E

        class Idx:
            def __init__(self, pairs):
                pairs = sorted(pairs)
                self.h = np.array([h for h, _ in pairs], np.int64)
                self.c = np.array([c for _, c in pairs], np.int64)
            lookup = E.KeyIndex.lookup

        rare = E.record_hashes("X", "Blue Oak Labs", {})
        common = E.record_hashes("X", "Star Traders", {})
        pairs = [(h, 2_000_000_007) for h in rare if h is not None]
        pairs += [(h, 3_000_000_000 + i) for h in common if h is not None
                  for i in range(E.FMAX + 5)]
        texts = {1_000_000_001: ("Blue Oak Labs", ""), 1_000_000_002: ("Star Traders", "")}
        gen = E.generate(np.array([1_000_000_001, 1_000_000_002]), np.array(["X", "X"]),
                         {}, {}, texts.get, Idx(pairs), {})
        self.assertEqual([(g[0], g[1]) for g in gen], [(1_000_000_001, 2_000_000_007)])
        self.assertEqual(gen[0][2][:3], [1, 1, 1])  # all three S1 keys hit


if __name__ == "__main__":
    unittest.main()
