"""Tests for the refusal-gate harness.

Two kinds of test, kept distinct:
- Hand-computed expectations on tiny fixed score lists. These were
  computed by hand from the vendored algorithms (see comments) and
  pin the transcription, including the grid conventions.
- Reproduction tests on tests/fixtures/*.json. Each fixture is the
  exact calibration score list produced by running the source repo's
  own demo code (brain: geometry seed 42, probe seed 11; heart:
  seed 42, probe seed 11; refusing: seed 1000, sigma 0.25; symmetry:
  closed-orbit demo, seed 42). No gate code was adjusted to fit:
  a mismatch here is a finding, not a cue to retune.

Run: python3 -m unittest discover -s tests -v   (from package root)
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from refusal_gate_harness import (  # noqa: E402
    brain_midpoint,
    compare,
    deltas,
    evaluate_fresh,
    heart_calibrate,
    refusing_first_best,
    reproduction_report,
    run_all,
    sym_calibrate,
    synthetic_scores,
    wp_bootstrap_tau_interval,
    wp_youden,
)

FIX = os.path.join(os.path.dirname(__file__), "fixtures")

# Hand set: in = [0.9, 0.8], ood = [0.2, 0.3]. J = 1 for every tau in
# (0.3, 0.8], and only there. Hence, by grid:
# - refusing (step 0.01): first J=1 tau = 0.31.
# - brain/wp (step 1/101): J=1 taus are k/101 for k = 31..80;
#   midpoint = (31 + 80) / 202 = 111/202.
# - heart first: 31/101; heart midpoint/midpoint_all/median: 111/202.
# - symmetry (step 0.01, 4dp): first 0.31, last 0.80, mid 0.555.
HAND_IN = [0.9, 0.8]
HAND_OOD = [0.2, 0.3]


def load_fixture(name):
    with open(os.path.join(FIX, name)) as f:
        data = json.load(f)
    return data["in"], data["ood"]


class TestHandComputed(unittest.TestCase):
    def test_refusing_first_best(self):
        tau, j, tpr, fpr = refusing_first_best(HAND_IN, HAND_OOD)
        self.assertAlmostEqual(tau, 0.31)
        self.assertAlmostEqual(j, 1.0)
        self.assertAlmostEqual(tpr, 1.0)
        self.assertAlmostEqual(fpr, 0.0)

    def test_brain_midpoint(self):
        tau, j = brain_midpoint(HAND_IN, HAND_OOD)
        self.assertAlmostEqual(tau, 111 / 202)
        self.assertAlmostEqual(j, 1.0)

    def test_wp_midpoint(self):
        tau, j = wp_youden(HAND_IN, HAND_OOD)
        self.assertAlmostEqual(tau, 111 / 202)
        self.assertAlmostEqual(j, 1.0)

    def test_heart_all_strategies(self):
        tau_m, j_m = heart_calibrate(HAND_IN, HAND_OOD,
                                     strategy="midpoint")
        tau_f, _ = heart_calibrate(HAND_IN, HAND_OOD, strategy="first")
        tau_a, _ = heart_calibrate(HAND_IN, HAND_OOD,
                                   strategy="midpoint_all")
        tau_med, _ = heart_calibrate(HAND_IN, HAND_OOD,
                                     strategy="median")
        self.assertAlmostEqual(tau_m, 111 / 202)
        self.assertAlmostEqual(tau_a, 111 / 202)
        self.assertAlmostEqual(tau_med, 111 / 202)
        self.assertAlmostEqual(tau_f, 31 / 101)
        self.assertAlmostEqual(j_m, 1.0)
        with self.assertRaises(ValueError):
            heart_calibrate(HAND_IN, HAND_OOD, strategy="bogus")

    def test_sym_midpoint(self):
        tau, j = sym_calibrate(HAND_IN, HAND_OOD)
        self.assertAlmostEqual(tau, 0.555)
        self.assertAlmostEqual(j, 1.0)

    def test_grid_conventions_differ(self):
        # refusing taus sit on multiples of 0.01; brain taus on
        # multiples of 1/101 (or midpoints thereof). The same
        # grid=101 parameter means different grids: 101 points vs
        # 102 points. This test pins that difference, it does not
        # normalise it away.
        tau_r, _, _, _ = refusing_first_best(HAND_IN, HAND_OOD)
        tau_b, _ = brain_midpoint(HAND_IN, HAND_OOD)
        self.assertAlmostEqual(tau_r * 100, round(tau_r * 100))
        self.assertAlmostEqual(tau_b * 101, 55.5)

    def test_empty_scores_raise(self):
        for fn in (refusing_first_best, brain_midpoint, wp_youden,
                   sym_calibrate):
            with self.assertRaises(ValueError):
                fn([], HAND_OOD)
            with self.assertRaises(ValueError):
                fn(HAND_IN, [])


class TestStrategyParameter(unittest.TestCase):
    def test_compare_requires_strategy(self):
        # strategy is keyword-only with no default: omitting it
        # must fail loudly, never fall back to a silent default.
        with self.assertRaises(TypeError):
            compare(HAND_IN, HAND_OOD)

    def test_compare_rejects_bad_strategy(self):
        with self.assertRaises(ValueError):
            compare(HAND_IN, HAND_OOD, strategy="median")

    def test_compare_first_vs_midpoint_on_plateau(self):
        # In-scores all >= 0.6, out-scores all <= 0.4: every tau in
        # (0.4, 0.6] gives J = 1, so the tie-break decides tau.
        in_p = [0.9, 0.8, 0.7, 0.6]
        ood_p = [0.1, 0.2, 0.3, 0.4]
        tau_f, j_f, _, _ = compare(in_p, ood_p, strategy="first")
        tau_m, j_m, _, _ = compare(in_p, ood_p, strategy="midpoint")
        self.assertAlmostEqual(j_f, 1.0)
        self.assertAlmostEqual(j_m, 1.0)
        self.assertAlmostEqual(tau_f, 0.41)
        self.assertAlmostEqual(tau_m, 50.5 / 101)
        self.assertGreater(tau_m - tau_f, 0.08)

    def test_heart_first_vs_midpoint_on_own_probes(self):
        # On the heart demo's own probes, first-best sits 0.0347
        # below the midpoint tau at the same J = 1.000: the
        # tie-break is material even when J is perfect.
        in_s, ood_s = load_fixture("heart_scores.json")
        tau_f, _ = heart_calibrate(in_s, ood_s, strategy="first")
        tau_m, _ = heart_calibrate(in_s, ood_s, strategy="midpoint")
        self.assertGreater(tau_m - tau_f, 0.03)


class TestReproduction(unittest.TestCase):
    def test_brain_fixture(self):
        in_s, ood_s = load_fixture("brain_scores.json")
        tau, j = brain_midpoint(in_s, ood_s)
        self.assertAlmostEqual(tau, 0.238, delta=0.001)
        self.assertAlmostEqual(j, 1.0)

    def test_heart_fixture(self):
        in_s, ood_s = load_fixture("heart_scores.json")
        tau, j = heart_calibrate(in_s, ood_s, strategy="midpoint")
        self.assertAlmostEqual(tau, 0.351, delta=0.001)
        self.assertAlmostEqual(j, 1.0)

    def test_refusing_fixture(self):
        in_s, ood_s = load_fixture("refusing_scores.json")
        tau, j, _, _ = refusing_first_best(in_s, ood_s)
        self.assertAlmostEqual(tau, 0.700)
        self.assertAlmostEqual(j, 0.735)

    def test_symmetry_fixture(self):
        in_s, ood_s = load_fixture("sym_scores.json")
        tau, j = sym_calibrate(in_s, ood_s)
        self.assertAlmostEqual(tau, 0.69)
        self.assertAlmostEqual(j, 0.875)

    def test_reproduction_report_statuses(self):
        rep = reproduction_report()
        self.assertEqual(rep["membrane-brain"]["status"], "reproduced")
        self.assertEqual(rep["membrane-heart"]["status"], "reproduced")
        self.assertEqual(rep["refusing-membrane"]["status"],
                         "reproduced_from_demo")
        self.assertEqual(rep["symmetry-quantum-flip-membrane"]["status"],
                         "reproduced_from_demo")
        self.assertEqual(rep["world-projection"]["status"],
                         "no_published_gate_number_in_repo")
        self.assertTrue(rep["membrane-brain"]["reproduced"])
        self.assertFalse(rep["world-projection"]["reproduced"])


class TestHarnessEndToEnd(unittest.TestCase):
    def test_brain_wp_identity(self):
        # Two repos, one algorithm: brain and world-projection must
        # agree exactly on identical inputs. A future divergence
        # here is a finding about the repos, not a harness bug.
        for seed in (1, 2, 3):
            in_s, ood_s = synthetic_scores(seed=seed)
            self.assertEqual(brain_midpoint(in_s, ood_s),
                             wp_youden(in_s, ood_s))

    def test_bootstrap_deterministic_and_valid(self):
        in_s, ood_s = synthetic_scores(seed=1)
        a = wp_bootstrap_tau_interval(in_s, ood_s)
        b = wp_bootstrap_tau_interval(in_s, ood_s)
        self.assertEqual(a, b)
        self.assertLessEqual(a["lower"], a["upper"])
        self.assertGreaterEqual(a["half_width"], 0.0)
        self.assertEqual(a["repeats"], 500)
        self.assertEqual(a["seed"], 909)
        with self.assertRaises(ValueError):
            wp_bootstrap_tau_interval(in_s, ood_s, repeats=50)

    def test_run_all_and_deltas(self):
        in_s, ood_s = synthetic_scores(seed=1)
        report = run_all(in_s, ood_s)
        for name in ("refusing-membrane", "membrane-brain",
                     "membrane-heart", "world-projection",
                     "symmetry-quantum-flip-membrane"):
            self.assertIn(name, report)
            self.assertIn("tau", report[name])
            self.assertIn("J", report[name])
        d = deltas(report)
        self.assertGreaterEqual(d["tau_spread_all"], 0.0)
        self.assertGreaterEqual(d["J_spread_all"], 0.0)
        self.assertIn("bootstrap", report["world-projection"])

    def test_fresh_evaluation_consistency(self):
        cal_in, cal_ood = synthetic_scores(seed=1)
        fresh_in, fresh_ood = synthetic_scores(seed=2)
        report = run_all(cal_in, cal_ood)
        ev = evaluate_fresh(report["membrane-brain"]["tau"],
                            fresh_in, fresh_ood)
        self.assertAlmostEqual(ev["J"], ev["TPR"] - ev["FPR"])
        self.assertGreaterEqual(ev["J"], -1.0)
        self.assertLessEqual(ev["J"], 1.0)


if __name__ == "__main__":
    unittest.main()
