"""Scheduler triage scored against real 1e9 labels — no long integration.

Runs the cheap 200-orbit screen + glass model on real-archive systems and
scores triage decisions against the dataset's 1e9 stability label.
fast-stable -> pred stable, fast-unstable -> pred unstable, zoom -> counted
as correct only if you would pay for truth; here we report zoom rate and
accuracy of non-zoomed calls plus overall accuracy if zooms are assumed
resolved correctly (upper bound) and if zooms are counted as abstentions.

This is the free-1e9 complement to benchmark_scheduler.py (which uses
synthetic Jupiters + 1e4 truth). No MERCURIUS run needed.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kirk import glass, scheduler  # noqa: E402
from kirk.short_features import short_features  # noqa: E402

DATA = Path(os.environ.get("KIRK_DATA", ROOT / "data"))
LABEL_COL = 19


def rebuild(row):
    import rebound
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=float(row[0]))
    for j in range(3):
        m, a, e, pomega, Omega, l = [float(row[1 + j * 6 + k]) for k in range(6)]
        sim.add(m=m, a=a, e=e, pomega=pomega, Omega=Omega, l=l)
    sim.move_to_com()
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-res", type=int, default=400)
    ap.add_argument("--n-rand", type=int, default=200)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--short-orbits", type=int, default=200)
    ap.add_argument("--nout", type=int, default=60)
    ap.add_argument("--lo", type=float, default=0.01)
    ap.add_argument("--hi", type=float, default=0.70)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "nam_spock_distilled.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "proofs" / "scheduler_real1e9.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_res, args.n_rand = 60, 30

    rng = np.random.default_rng(args.seed)
    resonant = np.load(DATA / "resonant_real.npz")["X"]
    random_ = np.load(DATA / "random_real.npz")["X"]
    rows = list(resonant[rng.choice(len(resonant), args.n_res, replace=False)])
    rows += list(random_[rng.choice(len(random_), args.n_rand, replace=False)])
    y = np.array([r[LABEL_COL] for r in rows], float) > 0.5
    print(f"sample {len(rows)} stable {(y).mean():.3f}", flush=True)

    model, meta = glass.load_artifact(args.model)
    sims = [rebuild(r) for r in rows]

    preds, actions, ok = [], [], []
    for i, sim in enumerate(sims):
        f, is_ok = short_features(sim, args.short_orbits, args.nout, seed=args.seed + i)
        if not is_ok:
            preds.append(0.0)
            actions.append("fast-unstable")
            ok.append(False)
            continue
        x = np.array([[f[k] for k in meta["feats"]]], dtype=np.float32)
        p = float(glass.predict(model, meta, x)[0])
        preds.append(p)
        actions.append(scheduler.triage(p, args.lo, args.hi))
        ok.append(True)
    preds = np.array(preds)
    actions = np.array(actions)

    # Non-zoom accuracy (what the paper can claim without paying).
    nz = actions != "zoom"
    acc_nz = float((((preds[nz] > 0.5) == y[nz]).mean())) if nz.sum() else float("nan")
    # Upper bound if every zoom resolved correctly.
    pred_all = (preds > 0.5)
    # zooms assumed correct:
    correct_ub = (nz & (pred_all == y)).sum() + (~nz).sum()
    acc_ub = float(correct_ub / len(y))
    # Lower bound if zooms all wrong (non-zoom correct only).
    acc_lb = float(((nz) & (pred_all == y)).sum() / len(y))

    out = {
        "config": {"n_res": args.n_res, "n_rand": args.n_rand, "seed": args.seed,
                   "short_orbits": args.short_orbits, "lo": args.lo, "hi": args.hi,
                   "model": str(args.model.relative_to(ROOT))},
        "n": len(y),
        "n_short_ok": int(np.sum(ok)),
        "zoom_frac": float((actions == "zoom").mean()),
        "fast_stable_frac": float((actions == "fast-stable").mean()),
        "fast_unstable_frac": float((actions == "fast-unstable").mean()),
        "acc_nonzoom": acc_nz,
        "n_nonzoom": int(nz.sum()),
        "acc_upper_bound_zoom_correct": acc_ub,
        "acc_lower_bound_zoom_wrong": acc_lb,
        "provenance": {"created": datetime.now(timezone.utc).isoformat()},
    }
    print(f"zoom {out['zoom_frac']:.1%} nonzoom acc {acc_nz:.3f} (n={out['n_nonzoom']}) "
          f"bounds [{acc_lb:.3f}, {acc_ub:.3f}]", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
