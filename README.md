# refusal-gate-harness v0.1.0

One small, stdlib-only package that runs the SAME calibration probe
sets through every refusal-gate implementation found in the public
sparkainlp-x repos, on the same grid, and reports the differences.
It was built as the top-ranked item from the 2026-10-29 aggressive
code exploration, which found the Youden gate duplicated in four
repos with different tie-breaks.

SYNTHETIC, classical score arithmetic only. No medical, quantum-
hardware, brain, or consciousness claim is made anywhere in this
package, and no gate here is a safety guarantee for any real
system.

## What is vendored, with provenance

Only gate code is vendored; geometry, waves, membranes, and demo
pipelines are NOT copied. Each function in
`src/refusal_gate_harness/gates.py` names its source repo and file:

| Harness function | Source repo and file | Rule |
|---|---|---|
| `refusing_first_best` | refusing-membrane, `rmembrane.py`, `RefusingMembrane.calibrate` | first (lowest) maximal-J tau |
| `brain_midpoint` | membrane-brain, `brain.py`, `calibrate` | midpoint of first/last maximal-J tau |
| `heart_calibrate` | membrane-heart, `membrane_heart/core.py`, `calibrate_tau` | strategy parameter: `midpoint` (longest run, default), `first`, `midpoint_all`, `median` |
| `wp_youden`, `wp_bootstrap_tau_interval` | world-projection, `calibrator.py` | midpoint of first/last maximal-J tau + deterministic percentile bootstrap (500 repeats, seed 909) |
| `sym_calibrate` | symmetry-quantum-flip-membrane, `src/symmetry_quantum_flip_membrane/gate.py`, `calibrate_tau` | midpoint of first/last maximal-J tau, fixed 0.01 steps, 4 decimals |

All maximise Youden's J = TPR − FPR under the accept rule
`score >= tau`, then freeze tau. Calibration and evaluation probes
are always disjoint: calibrating and evaluating on the same probes
would be cheating.

## The grid difference nobody had written down

The same `grid=101` parameter means different grids:

| Implementation | Taus scanned at grid=101 | Step |
|---|---|---|
| refusing-membrane | `g / (grid - 1)`, g = 0..100 | 0.01 (101 points) |
| membrane-brain, membrane-heart, world-projection | `step / grid`, step = 0..101 | 1/101 ≈ 0.0099 (102 points) |
| symmetry-quantum-flip-membrane | fixed 0.01 steps, no grid parameter | 0.01, rounded to 4 decimals |

The harness preserves these conventions exactly and labels them.
It does not silently normalise them, because on identical probes
the grid alone moves tau by up to ~0.005.

## The falsifiable bar, and what it returned

**Bar 1 — reproduction.** For each source repo, run its OWN demo
code, take the exact calibration score lists it produced, and feed
those lists to the vendored gate. The score lists are stored as
test fixtures (`tests/fixtures/*.json`), so the reproduction tests
are self-contained and re-runnable. A mismatch is a finding, never
a cue to retune a gate. Tolerance for README numbers is 0.001,
because the READMEs round to 3 decimals.

| Repo | Published (README/demo) | Vendored gate on the demo's own scores | Verdict |
|---|---|---|---|
| membrane-brain | tau 0.238, J 1.000 (README) | tau 0.23762376, J 1.000 | REPRODUCED |
| membrane-heart | tau 0.351, J 1.000 (README) | tau 0.35148515, J 1.000 | REPRODUCED |
| refusing-membrane | no README number; demo run tau 0.700, J 0.735 (200+200 probes, seed 1000, sigma 0.25) | tau 0.700, J 0.735 | REPRODUCED from demo |
| symmetry-quantum-flip-membrane | no README number; closed-orbit demo tau 0.69, J 0.875 (seed 42) | tau 0.69, J 0.875 | REPRODUCED from demo |
| world-projection | no number published (README: gate from 100+100 probes, seed 11; tau seed-specific) | nothing to match | NO PUBLISHED NUMBER — recorded, not fabricated |

**Bar 2 — tie-break deltas on identical probes.** Measured, not
asserted:

| Probe set | first-best tau | midpoint tau | Delta | J |
|---|---|---|---|---|
| Plateau (in ≥ 0.6, ood ≤ 0.4) | 0.4100 | 0.5000 | **0.0900** | 1.000 |
| Heart demo's own probes | 0.31683 | 0.35149 | **0.0347** | 1.000 |
| Synthetic seed 1 (100+100, mu 0.75 vs 0.25) | 0.5200 | 0.5198 | −0.0002 | 0.970 |
| Overlapping seed 3 (mu 0.60 vs 0.45) | 0.4700 | 0.4703 | 0.0003 | 0.330 |

Conclusion, stated plainly: the tie-break is material when the
maximal-J interval is wide (plateau, heart's own probes) and
negligible when it is narrow (synthetic seeds 1 and 3). Any repo
that reports a frozen tau without naming its tie-break rule is
under-reporting its gate. The harness's `compare()` function
enforces this: its `strategy` argument is keyword-only with NO
default, so omitting it is a TypeError, never a silent choice.

Two further cross-repo findings, both tested:
- membrane-brain and world-projection implement the SAME midpoint
  rule: `brain_midpoint == wp_youden` exactly on every tested
  synthetic set. They are one algorithm in two repos.
- On the synthetic seed-1 probes, the world-projection bootstrap
  interval for tau is [0.5050, 0.5743] (half-width 0.0347) around
  the point tau 0.5198: the interval midpoint (0.5396) sits 0.0198
  above the point tau. Bootstrap uncertainty is the same order as
  the plateau tie-break delta, so both must be reported together.

## Run it

```sh
python3 demo.py
python3 -m unittest discover -s tests -v
```

`demo.py` calibrates on synthetic seed 1 and evaluates the frozen
tau on a fresh seed-2 split. Full shared-probe table (seed 1):

| Implementation | tau | J | TPR | FPR |
|---|---|---|---|---|
| refusing-membrane | 0.5200 | 0.970 | 0.980 | 0.010 |
| membrane-brain | 0.5198 | 0.970 | 0.980 | 0.010 |
| membrane-heart | 0.5198 | 0.970 | 0.980 | 0.010 |
| world-projection (+ bootstrap [0.5050, 0.5743]) | 0.5198 | 0.970 | 0.980 | 0.010 |
| symmetry-quantum-flip-membrane | 0.5250 | 0.970 | 0.980 | 0.010 |

Fresh-split evaluation of the brain midpoint tau (seed 2):
TPR 0.940, FPR 0.010, J 0.930 — the gate holds up on fresh probes,
it does not merely memorise the calibration split.

## Honest limits

- Score-level only: the harness compares threshold rules on
  overlap-like scores. It says nothing about any membrane,
  waveform, pattern, or physical system.
- Published demo taus are seed- and pipeline-specific constants,
  not universal values. Re-running a demo with a different seed
  gives a different tau; that is expected, not a failure.
- The bootstrap interval is a percentile interval over resampled
  calibration scores, not a confidence interval for any real-world
  quantity.
- This package modifies no published repo. The fixes it implies
  (name the tie-break rule in each gate's docs; fix the grid
  description; record the bootstrap interval) belong in those
  repos, under their owners' word, not here.

## Credits

Project author: Jean-François Brisson (Spark AI NLP). Mun Seok
Lee is cited as inspiration only for the Digital to Wave paradigm;
he wrote no line of this package, and no Lee code is vendored
here — the vendored code is Jean-François Brisson's own code from
the five repos listed above. Licence: AGPL-3.0-only.
