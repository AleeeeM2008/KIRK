"""Combine scheduler benchmark JSONs into one evaluation table.

Usage: python scripts/aggregate_scheduler.py out.json run1.json run2.json ...
"""
import json
import sys

import numpy as np


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out = sys.argv[1]
    runs = [json.load(open(p)) for p in sys.argv[2:]]

    rows = []
    for d in runs:
        r = d["result"]
        rows.append({
            "seed": d["config"]["seed"],
            "n": r["n"],
            "acc": r["acc"],
            "ci": r["acc_ci95"],
            "zoom": r["zoom_frac"],
            "time_short": r["time_short_s"],
            "time_zoom": r["time_zoom_s"],
            "time_truth": r["time_truth_all_s"],
            "tight_acc": r["by_kind"]["tight"]["acc"],
            "tight_zoom": r["by_kind"]["tight"]["zoom_frac"],
            "tight_n": r["by_kind"]["tight"]["n"],
            "tight_stable": r["by_kind"]["tight"]["truth_stable"],
            "wide_acc": r["by_kind"]["wide"]["acc"],
            "wide_zoom": r["by_kind"]["wide"]["zoom_frac"],
            "wide_n": r["by_kind"]["wide"]["n"],
            "wide_stable": r["by_kind"]["wide"]["truth_stable"],
        })

    n = sum(r["n"] for r in rows)
    correct = sum(round(r["acc"] * r["n"]) for r in rows)
    # Totals by kind, for the paper table.
    by_kind = {}
    for kind, pk in (("tight", "tight"), ("wide", "wide")):
        nk = sum(r[f"{pk}_n"] for r in rows)
        ck = sum(round(r[f"{pk}_acc"] * r[f"{pk}_n"]) for r in rows)
        zk = sum(round(r[f"{pk}_zoom"] * r[f"{pk}_n"]) for r in rows)
        by_kind[kind] = {
            "n": nk,
            "acc": ck / nk if nk else float("nan"),
            "zoom_frac": zk / nk if nk else float("nan"),
            "truth_stable": int(sum(r[f"{pk}_stable"] for r in rows)),
        }
    agg = {
        "runs": rows,
        "n_total": n,
        "acc_total": correct / n,
        "acc_mean": float(np.mean([r["acc"] for r in rows])),
        "acc_std": float(np.std([r["acc"] for r in rows])),
        "zoom_mean": float(np.mean([r["zoom"] for r in rows])),
        "by_kind": by_kind,
        "note": "aggregate of benchmark_scheduler.py runs, no speedup claim",
    }
    with open(out, "w") as f:
        json.dump(agg, f, indent=1)
    print(f"{n} systems, acc {agg['acc_total']:.4f}, "
          f"zoom {agg['zoom_mean']:.1%}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
