"""The harness: same probes through every vendored gate, one report.

Design rule: the tie-break strategy is an explicit parameter of
:func:`compare`, never a silent default. ``strategy`` is keyword-only
with no default, so calling ``compare`` without it is a TypeError.
Valid strategies are "first" and "midpoint" (the two rules actually
shipped in the source repos); "midpoint" here means first/last
midpoint, the rule shared by membrane-brain, world-projection, and
symmetry-quantum-flip-membrane. membrane-heart's longest-run
midpoint is reported separately, because it is a third rule.

Published-number reproduction: four of the five source gates are
reproduced exactly. The method is stated in the README and encoded
in the test fixtures: run each source repo's own demo code, take
the exact calibration score lists it produced, and feed those lists
to the vendored gate. The fixtures in tests/fixtures/ are those
score lists, so the reproduction tests are self-contained and
re-runnable from the zip. world-projection publishes no gate
number, so its entry records that fact instead of a match.
:func:`reproduction_report` reports these statuses; a future
mismatch must be reported as a finding, never tuned away.
"""

import math
import random

from .gates import (
    brain_midpoint,
    heart_calibrate,
    refusing_first_best,
    sym_calibrate,
    tpr_fpr_at,
    wp_bootstrap_tau_interval,
    wp_youden,
)

__all__ = [
    "synthetic_scores",
    "run_all",
    "compare",
    "evaluate_fresh",
    "deltas",
    "reproduction_report",
    "STRATEGIES",
]

STRATEGIES = ("first", "midpoint")

# Published gate numbers and reproduction status, verified 2026-10-30
# by running the source repos' own demo code and feeding the exact
# calibration score lists it produced into the vendored gates.
# - membrane-brain: README demo tau 0.238, J 1.000. Original demo
#   (geometry seed 42, probe seed 11) gives tau 0.23762376, J 1.000;
#   vendored brain_midpoint returns the identical values.
# - membrane-heart: README demo tau 0.351, J 1.000. Original demo
#   (seed 42, probe seed 11) gives tau 0.35148515, J 1.000;
#   vendored heart_calibrate (default longest-run midpoint) returns
#   the identical values. On these same probes the "first" rule
#   gives 0.31683168 -- a 0.0347 tie-break delta at J = 1.000.
# - refusing-membrane: no gate tau/J in the README. Its demo
#   (200+200 probes, seed 1000, sigma 0.25) gives tau 0.700,
#   J 0.735; vendored refusing_first_best returns identical values.
# - symmetry-quantum-flip-membrane: no gate tau/J in the README.
#   Its closed-orbit demo (seed 42, 200+200 probes) gives tau 0.69,
#   J 0.875; vendored sym_calibrate returns identical values.
# - world-projection: README states the gate is built from 100+100
#   probes (seed 11) and that tau is seed-specific, but publishes no
#   number, so there is nothing to reproduce. wp_youden fidelity is
#   covered by hand-computed expectations and by its proven identity
#   with brain_midpoint on identical inputs (same algorithm).
PUBLISHED = {
    "membrane-brain": {"published_tau": 0.238, "published_J": 1.000,
                       "reproduced_tau": 0.2376237623762376,
                       "reproduced_J": 1.0,
                       "status": "reproduced",
                       "note": "README demo; tolerance 0.001 (README "
                               "rounds to 3 decimals)"},
    "membrane-heart": {"published_tau": 0.351, "published_J": 1.000,
                       "reproduced_tau": 0.35148514851485146,
                       "reproduced_J": 1.0,
                       "status": "reproduced",
                       "note": "README demo; tolerance 0.001"},
    "refusing-membrane": {"published_tau": None, "published_J": None,
                          "reproduced_tau": 0.7, "reproduced_J": 0.735,
                          "status": "reproduced_from_demo",
                          "note": "no README number; original demo run "
                                  "(seed 1000, sigma 0.25)"},
    "world-projection": {"published_tau": None, "published_J": None,
                         "reproduced_tau": None, "reproduced_J": None,
                         "status": "no_published_gate_number_in_repo",
                         "note": "README: gate from 100+100 probes, "
                                 "seed 11; tau seed-specific, no "
                                 "number published"},
    "symmetry-quantum-flip-membrane": {
        "published_tau": None, "published_J": None,
        "reproduced_tau": 0.69, "reproduced_J": 0.875,
        "status": "reproduced_from_demo",
        "note": "no README number; original closed-orbit demo "
                "run (seed 42)"},
}


def synthetic_scores(seed, n_in=100, n_ood=100,
                     in_mu=0.75, in_sd=0.12, ood_mu=0.25, ood_sd=0.12):
    """Deterministic synthetic score lists (overlaps in [0, 1]).

    In-scores cluster high, out-scores cluster low; the overlap of
    the two distributions decides J. This is score-level synthesis
    only: no membrane, wave, or pattern is simulated.
    """
    rng = random.Random(seed)
    def draw(mu, sd):
        return min(1.0, max(0.0, rng.gauss(mu, sd)))
    return ([draw(in_mu, in_sd) for _ in range(n_in)],
            [draw(ood_mu, ood_sd) for _ in range(n_ood)])


def run_all(in_scores, ood_scores, grid=101, bootstrap_repeats=500,
            bootstrap_seed=909):
    """Run every vendored gate on the same probes.

    Returns a dict per implementation: tau, J, and TPR/FPR at that
    implementation's own frozen tau (calibration-split values).
    world-projection additionally carries its bootstrap interval.
    """
    tau_r, j_r, tpr_r, fpr_r = refusing_first_best(
        in_scores, ood_scores, grid=grid)
    tau_b, j_b = brain_midpoint(in_scores, ood_scores, grid=grid)
    tpr_b, fpr_b = tpr_fpr_at(in_scores, ood_scores, tau_b)
    tau_h, j_h = heart_calibrate(in_scores, ood_scores, grid=grid,
                                 strategy="midpoint")
    tpr_h, fpr_h = tpr_fpr_at(in_scores, ood_scores, tau_h)
    tau_ha, j_ha = heart_calibrate(in_scores, ood_scores, grid=grid,
                                   strategy="midpoint_all")
    tau_w, j_w = wp_youden(in_scores, ood_scores, grid=grid)
    tpr_w, fpr_w = tpr_fpr_at(in_scores, ood_scores, tau_w)
    boot = wp_bootstrap_tau_interval(in_scores, ood_scores, grid=grid,
                                     repeats=bootstrap_repeats,
                                     seed=bootstrap_seed)
    tau_s, j_s = sym_calibrate(in_scores, ood_scores)
    tpr_s, fpr_s = tpr_fpr_at(in_scores, ood_scores, tau_s)
    return {
        "refusing-membrane": {"rule": "first-best", "tau": tau_r,
                              "J": j_r, "TPR": tpr_r, "FPR": fpr_r},
        "membrane-brain": {"rule": "midpoint first/last", "tau": tau_b,
                           "J": j_b, "TPR": tpr_b, "FPR": fpr_b},
        "membrane-heart": {"rule": "midpoint longest-run", "tau": tau_h,
                           "J": j_h, "TPR": tpr_h, "FPR": fpr_h},
        "membrane-heart-midpoint_all": {
            "rule": "midpoint first/last (v0.1.0 rule)", "tau": tau_ha,
            "J": j_ha, "TPR": tpr_fpr_at(in_scores, ood_scores, tau_ha)[0],
            "FPR": tpr_fpr_at(in_scores, ood_scores, tau_ha)[1]},
        "world-projection": {"rule": "midpoint first/last", "tau": tau_w,
                             "J": j_w, "TPR": tpr_w, "FPR": fpr_w,
                             "bootstrap": boot,
                             "bootstrap_midpoint":
                                 (boot["lower"] + boot["upper"]) / 2.0},
        "symmetry-quantum-flip-membrane": {
            "rule": "midpoint first/last, 0.01 steps, 4dp",
            "tau": tau_s, "J": j_s, "TPR": tpr_s, "FPR": fpr_s},
    }


def compare(in_scores, ood_scores, *, strategy, grid=101):
    """One named strategy, applied uniformly, on identical probes.

    ``strategy`` is keyword-only and has NO default: omitting it is
    a TypeError. This is the harness's anti-silent-default rule.
    "first" uses the refusing-membrane rule; "midpoint" uses the
    first/last midpoint rule shared by brain, world-projection, and
    symmetry (on the common grid). Returns (tau, J, TPR, FPR).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if strategy == "first":
        tau, j, tpr, fpr = refusing_first_best(
            in_scores, ood_scores, grid=grid)
    else:
        tau, j = brain_midpoint(in_scores, ood_scores, grid=grid)
        tpr, fpr = tpr_fpr_at(in_scores, ood_scores, tau)
    return tau, j, tpr, fpr


def evaluate_fresh(tau, fresh_in, fresh_ood):
    """TPR/FPR/J of a FROZEN tau on fresh probes (never calibrated on)."""
    tpr, fpr = tpr_fpr_at(fresh_in, fresh_ood, tau)
    return {"tau": tau, "TPR": tpr, "FPR": fpr, "J": tpr - fpr}


def deltas(report):
    """Numeric deltas between tie-break taus on identical probes."""
    first = report["refusing-membrane"]["tau"]
    mid = report["membrane-brain"]["tau"]
    mid_long = report["membrane-heart"]["tau"]
    boot_mid = report["world-projection"]["bootstrap_midpoint"]
    sym = report["symmetry-quantum-flip-membrane"]["tau"]
    return {
        "first_vs_midpoint": mid - first,
        "first_vs_midpoint_longest_run": mid_long - first,
        "midpoint_vs_midpoint_longest_run": mid_long - mid,
        "midpoint_vs_bootstrap_midpoint": boot_mid - mid,
        "midpoint_vs_symmetry": sym - mid,
        "tau_spread_all": max(first, mid, mid_long, boot_mid, sym)
                          - min(first, mid, mid_long, boot_mid, sym),
        "J_spread_all": max(v["J"] for v in report.values())
                         - min(v["J"] for v in report.values()),
    }


def reproduction_report():
    """Status of each source repo's published gate number.

    Four of five gates are reproduced exactly from the source repos'
    own demo score lists (see PUBLISHED for method and tolerance).
    world-projection publishes no number, so its entry records that
    fact rather than a fabricated match. Nothing here is adjusted to
    fit: a future mismatch must be reported as a finding, not tuned
    away.
    """
    out = {}
    for repo, pub in PUBLISHED.items():
        out[repo] = {"published_tau": pub["published_tau"],
                     "published_J": pub["published_J"],
                     "reproduced_tau": pub["reproduced_tau"],
                     "reproduced_J": pub["reproduced_J"],
                     "note": pub["note"], "status": pub["status"],
                     "reproduced": pub["status"] in (
                         "reproduced", "reproduced_from_demo")}
    return out
