"""Vendored refusal-gate implementations, with provenance.

Each function below is a faithful transcription of the gate logic from
one public sparkainlp-x repository. Only the gate code is vendored;
geometry, waves, membranes, and demo pipelines are NOT copied. Where
the originals differ in grid convention or tie-break, the difference
is preserved exactly and labelled -- the harness compares them, it
does not silently normalise them.

Grid conventions (all take a ``grid`` parameter, all scan tau in
[0, 1], all maximise Youden's J = TPR - FPR with accept rule
``score >= tau``):
- refusing-membrane: ``tau = g / (grid - 1)`` for ``g in range(grid)``
  -> grid points, step 1/(grid-1). At grid=101: 101 points, step 0.01.
- membrane-brain, membrane-heart, world-projection: ``tau = step /
  grid`` for ``step in range(grid + 1)`` -> grid+1 points, step
  1/grid. At grid=101: 102 points, step 1/101 ~= 0.0099.
- symmetry-quantum-flip-membrane: fixed 0.01 steps (no grid
  parameter), tau and J rounded to 4 decimals.

Tie-breaks (when several taus share maximal J, ties within 1e-12
unless noted):
- refusing-membrane: first (lowest) maximal-J tau.
- membrane-brain: midpoint of first and last maximal-J tau.
- membrane-heart: strategy parameter; default "midpoint" = midpoint
  of the LONGEST contiguous run of maximal-J taus (first run wins
  ties); also "first", "midpoint_all" (first/last, the v0.1.0 rule),
  and "median".
- world-projection: midpoint of first and last maximal-J tau, plus
  a deterministic percentile bootstrap interval for tau.
- symmetry-quantum-flip-membrane: midpoint of first and last
  maximal-J tau, rounded to 4 decimals.

All of this is classical, synthetic score arithmetic. No gate here
is a medical device, a quantum result, a brain model, or a safety
guarantee for any real system.
"""

import math
import random

__all__ = [
    "refusing_first_best",
    "brain_midpoint",
    "heart_calibrate",
    "wp_youden",
    "wp_bootstrap_tau_interval",
    "sym_calibrate",
    "tpr_fpr_at",
    "HEART_STRATEGIES",
]


def tpr_fpr_at(in_scores, ood_scores, tau):
    """TPR and FPR at a frozen tau under the accept rule score >= tau."""
    tpr = sum(1 for v in in_scores if v >= tau) / len(in_scores)
    fpr = sum(1 for v in ood_scores if v >= tau) / len(ood_scores)
    return tpr, fpr


# ------------------------------------------------------------------
# Provenance: sparkainlp-x/refusing-membrane, rmembrane.py,
# RefusingMembrane.calibrate (first-best tau).
# Original form: method on a membrane object, scoring probes with
# max_overlap(); the sweep itself is transcribed verbatim here and
# operates on score lists so all implementations share one input type.
# ------------------------------------------------------------------
def refusing_first_best(in_scores, ood_scores, grid=101):
    """First maximal-J tau. Returns (tau, J, tpr, fpr)."""
    in_scores = list(in_scores)
    ood_scores = list(ood_scores)
    if not in_scores or not ood_scores:
        raise ValueError("calibration needs in and out scores")
    best = None
    for g in range(grid):
        tau = g / (grid - 1)
        tpr, fpr = tpr_fpr_at(in_scores, ood_scores, tau)
        if best is None or (tpr - fpr) > best[0]:
            best = (tpr - fpr, tau, tpr, fpr)
    return best[1], best[0], best[2], best[3]


# ------------------------------------------------------------------
# Provenance: sparkainlp-x/membrane-brain, brain.py,
# RefusingMembrane.calibrate (midpoint of the taus maximising J).
# ------------------------------------------------------------------
def brain_midpoint(in_scores, ood_scores, grid=101):
    """Midpoint of first/last maximal-J tau. Returns (tau, J)."""
    in_scores = list(in_scores)
    ood_scores = list(ood_scores)
    if not in_scores or not ood_scores:
        raise ValueError("calibration needs in and out scores")
    best_j, best_taus = -2.0, []
    for step in range(grid + 1):
        tau = step / grid
        tpr, fpr = tpr_fpr_at(in_scores, ood_scores, tau)
        j = tpr - fpr
        if j > best_j + 1e-12:
            best_j, best_taus = j, [tau]
        elif abs(j - best_j) <= 1e-12:
            best_taus.append(tau)
    return (best_taus[0] + best_taus[-1]) / 2.0, best_j


# ------------------------------------------------------------------
# Provenance: sparkainlp-x/membrane-heart, membrane_heart/core.py,
# calibrate_tau (strategy parameter) and MembraneHeart.calibrate.
# The longest-run "midpoint" default is preserved exactly; it is NOT
# the same as first/last midpoint when maximal-J taus are split.
# ------------------------------------------------------------------
HEART_STRATEGIES = ("midpoint", "first", "midpoint_all", "median")


def heart_calibrate(in_scores, ood_scores, grid=101, strategy="midpoint"):
    """Strategy-parameterised tau. Returns (tau, J)."""
    if strategy not in HEART_STRATEGIES:
        raise ValueError(f"strategy must be one of {HEART_STRATEGIES}")
    in_scores = list(in_scores)
    ood_scores = list(ood_scores)
    if not in_scores or not ood_scores:
        raise ValueError("calibration needs in-library and "
                         "out-of-library probes")
    if grid < 1:
        raise ValueError("grid must be at least 1")
    if not all(math.isfinite(v) for v in in_scores + ood_scores):
        raise ValueError("calibration scores must be finite")
    best_j, best_idx = -2.0, []
    for step in range(grid + 1):
        tau = step / grid
        tpr, fpr = tpr_fpr_at(in_scores, ood_scores, tau)
        j = tpr - fpr
        if j > best_j + 1e-12:
            best_j, best_idx = j, [step]
        elif abs(j - best_j) <= 1e-12:
            best_idx.append(step)
    if strategy == "first":
        return best_idx[0] / grid, best_j
    if strategy == "midpoint_all":
        return (best_idx[0] + best_idx[-1]) / (2.0 * grid), best_j
    if strategy == "median":
        m = len(best_idx)
        mid = (best_idx[m // 2] + best_idx[(m - 1) // 2]) / 2.0
        return mid / grid, best_j
    runs, run = [], [best_idx[0]]
    for step in best_idx[1:]:
        if step == run[-1] + 1:
            run.append(step)
        else:
            runs.append(run)
            run = [step]
    runs.append(run)
    longest = max(runs, key=len)  # max keeps the first of equal lengths
    return (longest[0] + longest[-1]) / (2.0 * grid), best_j


# ------------------------------------------------------------------
# Provenance: sparkainlp-x/world-projection, calibrator.py,
# youden_tau and bootstrap_tau_interval. Note: this file's
# youden_tau is algorithmically identical to membrane-brain's
# midpoint rule; it is vendored separately, not merged, so the
# harness can verify that the two repos agree.
# ------------------------------------------------------------------
def _validate_scores(values, name):
    if not values:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must contain only finite values")


def wp_youden(in_vals, ood_vals, grid=101):
    """Midpoint of first/last maximal-J tau. Returns (tau, J)."""
    if not isinstance(grid, int) or grid < 1:
        raise ValueError("grid must be a positive integer")
    _validate_scores(in_vals, "in_vals")
    _validate_scores(ood_vals, "ood_vals")
    best_j, best_taus = -2.0, []
    for step in range(grid + 1):
        tau = step / grid
        tpr, fpr = tpr_fpr_at(in_vals, ood_vals, tau)
        j = tpr - fpr
        if j > best_j + 1e-12:
            best_j, best_taus = j, [tau]
        elif abs(j - best_j) <= 1e-12:
            best_taus.append(tau)
    return (best_taus[0] + best_taus[-1]) / 2.0, best_j


def _quantile(values, q):
    values = sorted(values)
    if not values:
        raise ValueError("quantile requires at least one value")
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def wp_bootstrap_tau_interval(in_vals, ood_vals, grid=101, repeats=500,
                              seed=909):
    """Deterministic percentile bootstrap interval for the tau."""
    if repeats < 100:
        raise ValueError("use at least 100 bootstrap replicates")
    _validate_scores(in_vals, "in_vals")
    _validate_scores(ood_vals, "ood_vals")
    rng = random.Random(seed)
    taus = []
    for _ in range(repeats):
        in_sample = [in_vals[rng.randrange(len(in_vals))]
                     for _ in range(len(in_vals))]
        ood_sample = [ood_vals[rng.randrange(len(ood_vals))]
                      for _ in range(len(ood_vals))]
        tau, _ = wp_youden(in_sample, ood_sample, grid=grid)
        taus.append(tau)
    lower, upper = _quantile(taus, 0.025), _quantile(taus, 0.975)
    return {"lower": lower, "upper": upper,
            "half_width": (upper - lower) / 2.0,
            "repeats": repeats, "seed": seed,
            "method": "percentile bootstrap"}


# ------------------------------------------------------------------
# Provenance: sparkainlp-x/symmetry-quantum-flip-membrane,
# src/symmetry_quantum_flip_membrane/gate.py, calibrate_tau.
# Fixed 0.01 steps; tau and J rounded to 4 decimals in the original.
# ------------------------------------------------------------------
def sym_calibrate(clean_overlaps, junk_overlaps):
    """Midpoint of first/last maximal-J tau. Returns (tau, J)."""
    clean = list(clean_overlaps)
    junk = list(junk_overlaps)
    if not clean or not junk:
        raise ValueError("calibration needs clean and junk overlaps")
    best_j = -2.0
    best_taus = []
    tau = 0.0
    while tau <= 1.0001:
        tp = sum(1 for x in clean if x >= tau)
        fn = len(clean) - tp
        tn = sum(1 for x in junk if x < tau)
        fp = len(junk) - tn
        sens = tp / len(clean) if clean else 0.0
        spec = tn / len(junk) if junk else 0.0
        j = sens + spec - 1.0
        if j > best_j + 1e-12:
            best_j = j
            best_taus = [tau]
        elif abs(j - best_j) <= 1e-12:
            best_taus.append(tau)
        tau = round(tau + 0.01, 10)
    tau_mid = (best_taus[0] + best_taus[-1]) / 2.0
    return round(tau_mid, 4), round(best_j, 4)
