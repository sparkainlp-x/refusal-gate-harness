"""End-to-end demo: same synthetic probes through every gate."""

import json
import sys

sys.path.insert(0, "src")

from refusal_gate_harness import (
    compare,
    deltas,
    evaluate_fresh,
    reproduction_report,
    run_all,
    synthetic_scores,
)


def main():
    # Calibration split (seed 1) and a FRESH evaluation split (seed 2):
    # calibrating and evaluating on the same probes would be cheating.
    cal_in, cal_ood = synthetic_scores(seed=1)
    fresh_in, fresh_ood = synthetic_scores(seed=2)
    report = run_all(cal_in, cal_ood)
    print("== calibration split, seed 1 (100 in, 100 out) ==")
    for name, row in report.items():
        extra = ""
        if "bootstrap" in row:
            b = row["bootstrap"]
            extra = (" bootstrap [%.4f, %.4f] half-width %.4f"
                     % (b["lower"], b["upper"], b["half_width"]))
        print("%-32s tau=%.4f J=%.4f TPR=%.3f FPR=%.3f%s"
              % (name, row["tau"], row["J"], row["TPR"], row["FPR"],
                 extra))
    print("== tie-break deltas on identical probes ==")
    print(json.dumps(deltas(report), indent=2))
    print("== explicit-strategy comparison ==")
    for strategy in ("first", "midpoint"):
        tau, j, tpr, fpr = compare(cal_in, cal_ood, strategy=strategy)
        print("%-9s tau=%.4f J=%.4f TPR=%.3f FPR=%.3f"
              % (strategy, tau, j, tpr, fpr))
    print("== fresh-split evaluation of the brain midpoint tau ==")
    print(json.dumps(evaluate_fresh(report["membrane-brain"]["tau"],
                                     fresh_in, fresh_ood), indent=2))
    print("== reproduction status ==")
    print(json.dumps(reproduction_report(), indent=2))


if __name__ == "__main__":
    main()
