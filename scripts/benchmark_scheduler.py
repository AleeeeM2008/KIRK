"""Measure the scheduler on Jupiter-mass systems.

This is the benchmark that used to live in a lab script. It does four things:

  1. builds a deterministic sample of tight and wide Jupiter systems,
  2. runs the cheap short-window features and the glass model,
  3. triages with the configured gates and integrates the zooms,
  4. integrates every system as truth so the ledger is complete.

Accuracy is against the truth run with a bootstrap interval. The defaults are
the fresh evaluation set (seed 10, 300 systems, 10^4-orbit truth); the gates
were tuned on seeds 5-9 at 2x10^3 orbits. Wall times are reported as
diagnostics only; no speedup claim is made.

Example:
    ~/kirk-env/bin/python scripts/benchmark_scheduler.py
    ~/kirk-env/bin/python scripts/benchmark_scheduler.py --quick
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rebound

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kirk import glass, scheduler  # noqa: E402
from kirk.short_features import short_features  # noqa: E402


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_jupiter(K12, K23, seed):
    """Three 1e-3 M_sun planets spaced by K mutual Hill radii."""
    rng = np.random.default_rng(seed)
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=1.0)
    m_p = 1e-3
    a1 = 1.0
    rh1 = a1 * (m_p / 3.0) ** (1 / 3.)
    a2 = a1 + K12 * rh1
    rh2 = a2 * (m_p / 3.0) ** (1 / 3.)
    a3 = a2 + K23 * rh2
    for a in (a1, a2, a3):
        sim.add(m=m_p, a=a, e=rng.uniform(0, 0.02),
                pomega=rng.uniform(0, 2 * np.pi),
                l=rng.uniform(0, 2 * np.pi))
    sim.move_to_com()
    return sim


def long_truth(sim0, orbits=2000):
    """MERCURIUS long run. Returns True if the system survives."""
    sim = sim0.copy()
    try:
        sim.integrator = "mercurius"
        sim.dt = 0.05
    except Exception:
        sim.integrator = "ias15"

    minP = min([p.P for p in sim.particles[1:]])
    T = orbits * abs(minP)
    try:
        for t in np.linspace(0, T, 16):
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


def build_sample(n, seed):
    """Half tight, half wide. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    systems = []
    for i in range(n // 2):
        K12 = rng.uniform(3.0, 4.5)
        K23 = rng.uniform(3.0, 4.5)
        systems.append((f"tight-{i}", "tight", make_jupiter(K12, K23, 300 + i)))
    for i in range(n - n // 2):
        K12 = rng.uniform(6.0, 9.0)
        K23 = rng.uniform(6.0, 9.0)
        systems.append((f"wide-{i}", "wide", make_jupiter(K12, K23, 700 + i)))
    return systems


def bootstrap_ci(pred, truth, n_boot=2000, seed=0):
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if len(pred) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pred), size=(n_boot, len(pred)))
    acc = (pred[idx] == truth[idx]).mean(1)
    return [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--orbits", type=int, default=10000)
    ap.add_argument("--short-orbits", type=int, default=200)
    ap.add_argument("--nout", type=int, default=60)
    ap.add_argument("--lo", type=float, default=0.01)
    ap.add_argument("--hi", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--model", type=Path,
                    default=ROOT / "models" / "nam_spock_distilled.pt")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.n, args.orbits, args.short_orbits, args.nout = 6, 500, 100, 40

    if args.out is None:
        args.out = (Path("/tmp/kirk_quick_scheduler.json") if args.quick
                    else ROOT / "proofs" / "scheduler.json")
    if args.n % 2:
        args.n += 1

    model, meta = glass.load_artifact(args.model)
    systems = build_sample(args.n, args.seed)
    print(f"{len(systems)} systems, short {args.short_orbits} orbits, truth {args.orbits} orbits",
          flush=True)

    rows = []
    t0 = time.time()
    for i, (name, kind, sim) in enumerate(systems):
        f, ok = short_features(sim, args.short_orbits, args.nout, seed=args.seed + i)
        if not ok:
            rows.append({"name": name, "kind": kind, "P": 0.0,
                         "action": "fast-unstable", "short_ok": False})
            continue
        x = np.array([[f[k] for k in meta["feats"]]], dtype=np.float32)
        p = float(glass.predict(model, meta, x)[0])
        rows.append({"name": name, "kind": kind, "P": p,
                     "action": scheduler.triage(p, args.lo, args.hi),
                     "short_ok": True})
    t_short = time.time() - t0

    t0 = time.time()
    for row, (_, _, sim) in zip(rows, systems):
        if row["action"] == "zoom":
            row["truth"] = bool(long_truth(sim, args.orbits))
        else:
            row["truth"] = None
    t_zoom = time.time() - t0

    t0 = time.time()
    truth_all = [bool(long_truth(sim, args.orbits)) for _, _, sim in systems]
    t_truth = time.time() - t0

    for row, truth in zip(rows, truth_all):
        row["truth_all"] = truth
        if row["action"] == "zoom":
            row["pred"] = row.pop("truth")
        else:
            row["pred"] = row["action"] == "fast-stable"

    pred = np.array([r["pred"] for r in rows], dtype=bool)
    truth = np.array(truth_all, dtype=bool)
    acc = float((pred == truth).mean())
    zoom_frac = float(np.mean([r["action"] == "zoom" for r in rows]))

    by_kind = {}
    for kind in ("tight", "wide"):
        sel = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {
            "n": len(sel),
            "acc": float(np.mean([r["pred"] == r["truth_all"] for r in sel])),
            "truth_stable": int(sum(r["truth_all"] for r in sel)),
            "zoom_frac": float(np.mean([r["action"] == "zoom" for r in sel])),
        }

    result = {
        "acc": acc,
        "acc_ci95": bootstrap_ci(pred, truth, seed=args.seed),
        "n": len(rows),
        "zoom_frac": zoom_frac,
        "time_short_s": t_short,
        "time_zoom_s": t_zoom,
        "time_truth_all_s": t_truth,
        "by_kind": by_kind,
        "gates": {"lo": args.lo, "hi": args.hi},
    }

    try:
        model_label = str(args.model.relative_to(ROOT))
    except ValueError:
        model_label = str(args.model)

    payload = {
        "model": model_label,
        "config": {
            "n": args.n,
            "orbits": args.orbits,
            "short_orbits": args.short_orbits,
            "nout": args.nout,
            "seed": args.seed,
            "lo": args.lo,
            "hi": args.hi,
        },
        "result": result,
        "rows": rows,
        "provenance": {
            "created": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "torch": __import__("torch").__version__,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))

    print(f"ACC {acc:.3f}  95% CI [{result['acc_ci95'][0]:.3f}, {result['acc_ci95'][1]:.3f}]",
          flush=True)
    print(f"zoom {zoom_frac:.0%}  "
          f"short {t_short:.1f}s zoom {t_zoom:.1f}s truth-all {t_truth:.1f}s",
          flush=True)
    for kind, v in by_kind.items():
        print(f"  {kind:5s} n={v['n']} acc={v['acc']:.3f} "
              f"zoom={v['zoom_frac']:.0%} truth-stable={v['truth_stable']}", flush=True)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
