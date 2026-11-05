"""Validate predict_stable on 3, 4, 5 and 6 planet systems.

The distilled classifier was trained on 3-planet systems. For 4+ planets the
code takes the minimum over adjacent trios, which is SPOCK's rule, but that is
not evidence by itself. This script generates synthetic systems, runs the glass
predictor, and compares it with a long MERCURIUS truth run.

Upstream SPOCK is used for comparison when installed; it is optional.

Example:
    ~/kirk-env/bin/python scripts/validate_multiplanet.py
    ~/kirk-env/bin/python scripts/validate_multiplanet.py --quick
"""
import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rebound

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kirk import features  # noqa: E402


def make_system(n_planets, rng, compact):
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=1.0)
    period = 1.0
    for _ in range(n_planets):
        mass = 10 ** rng.uniform(-5.3, -4.7)  # Earth-mass range
        sim.add(m=mass, P=period, e=rng.uniform(0.01, 0.05),
                pomega=rng.uniform(0, 2 * np.pi),
                l=rng.uniform(0, 2 * np.pi))
        period *= rng.uniform(1.15, 1.35) if compact else rng.uniform(1.5, 2.0)
    sim.move_to_com()
    return sim


def chain_features(sim):
    """Minimum adjacent period ratio and mutual-Hill spacing."""
    ps = sim.particles
    a = [p.a for p in ps[1:]]
    m = [p.m for p in ps[1:]]
    m_star = ps[0].m
    ratios = [a[i + 1] / a[i] for i in range(len(a) - 1)]
    hills = []
    for i in range(len(a) - 1):
        rh = a[i] * ((m[i] + m[i + 1]) / 3.0 / m_star) ** (1 / 3.)
        hills.append((a[i + 1] - a[i]) / rh)
    return float(min(ratios)), float(min(hills))


def long_truth(sim0, orbits):
    sim = sim0.copy()
    try:
        sim.integrator = "mercurius"
        sim.dt = 0.05
    except Exception:
        sim.integrator = "ias15"

    min_p = min(abs(p.P) for p in sim.particles[1:])
    try:
        for t in np.linspace(0, orbits * min_p, 32):
            sim.integrate(t)
            for p in sim.particles[1:]:
                r = np.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2)
                if r > 100 or r < 0.02:
                    return False
            if any(o.e > 0.7 for o in sim.orbits()):
                return False
    except Exception:
        return False
    return True


def binary_auc(scores, labels):
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((((pos[:, None] > neg[None, :]).sum()
                   + 0.5 * (pos[:, None] == neg[None, :]).sum())
                  / (len(pos) * len(neg))))


def bootstrap_ci(pred, truth, n_boot=2000, seed=0):
    pred = np.asarray(pred, bool)
    truth = np.asarray(truth, bool)
    if len(pred) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pred), size=(n_boot, len(pred)))
    return [float(np.percentile((pred[idx] == truth[idx]).mean(1), 2.5)),
            float(np.percentile((pred[idx] == truth[idx]).mean(1), 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--n-per-count", type=int, default=50)
    ap.add_argument("--orbits", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--hill-threshold", type=float, default=6.75,
                    help="fixed Hill-spacing baseline threshold")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.counts, args.n_per_count, args.orbits = [3, 4], 5, 1000

    if args.out is None:
        args.out = (Path("/tmp/kirk_quick_multiplanet.json") if args.quick
                    else ROOT / "proofs" / "multiplanet.json")

    try:
        from spock import FeatureClassifier
        upstream = FeatureClassifier()
    except ImportError:
        upstream = None

    rng = np.random.default_rng(args.seed)
    rows = []
    t0 = time.time()
    for n in args.counts:
        for i in range(args.n_per_count):
            compact = i % 2 == 0
            sim = make_system(n, rng, compact)
            min_ratio, min_hill = chain_features(sim)
            p = float(features.predict_stable(sim, n_jobs=1))
            p_spock = float(upstream.predict_stable(sim)) if upstream else None
            truth = bool(long_truth(sim, args.orbits))
            rows.append({"n": n, "i": i, "compact": compact,
                         "min_period_ratio": min_ratio, "min_hill": min_hill,
                         "kirk": p, "spock": p_spock, "truth": truth})
            print(f"n={n} {i:2d} compact={compact} "
                  f"KIRK={p:.3f} SPOCK={p_spock if p_spock is None else round(p_spock, 3)} "
                  f"truth={int(truth)}  {time.time() - t0:.0f}s", flush=True)

    summary = {}
    for n in args.counts:
        sel = [r for r in rows if r["n"] == n]
        pred = np.array([r["kirk"] > 0.5 for r in sel])
        truth = np.array([r["truth"] for r in sel])
        out = {
            "n": len(sel),
            "stable_frac": float(truth.mean()),
            "acc": float((pred == truth).mean()),
            "acc_ci95": bootstrap_ci(pred, truth, seed=args.seed + n),
            "auc": binary_auc(np.array([r["kirk"] for r in sel]), truth),
            "zoom_frac": float(np.mean([0.01 < r["kirk"] < 0.70 for r in sel])),
        }
        if upstream:
            sp = np.array([r["spock"] for r in sel])
            out["spock_acc"] = float(((sp > 0.5) == truth).mean())
            out["spock_auc"] = binary_auc(sp, truth)
            out["kirk_spock_agree"] = float(np.mean(
                ((np.array([r["kirk"] for r in sel]) > 0.5) == (sp > 0.5))))
            out["kirk_spock_mean_abs_diff"] = float(np.mean(
                np.abs(np.array([r["kirk"] for r in sel]) - sp)))
        hills = np.array([r["min_hill"] for r in sel])
        out["hill_threshold"] = args.hill_threshold
        out["hill_acc"] = float(
            ((hills > args.hill_threshold) == truth).mean())
        if math.isnan(out["auc"]):
            out["auc"] = None
        if "spock_auc" in out and math.isnan(out["spock_auc"]):
            out["spock_auc"] = None
        summary[str(n)] = out
        auc_label = "—" if out["auc"] is None else f"{out['auc']:.3f}"
        print(f"n={n}: acc {out['acc']:.3f} AUC {auc_label} "
              f"stable {out['stable_frac']:.2f} "
              f"hill {out['hill_acc']:.2f} agree {out.get('kirk_spock_agree', 0):.2f}",
              flush=True)

    out_path = args.out
    config = {k: (str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items()}
    payload = {
        "config": config,
        "summary": summary,
        "rows": rows,
        "provenance": {
            "created": datetime.now(timezone.utc).isoformat(),
            "upstream_spock": upstream is not None,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
