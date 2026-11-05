"""Glass-box ablation + calibration for the paper.

Loads the shipped distilled NAM, evaluates on real-archive systems with
real 1e9 labels, then:
  1. leave-one-feature-out (replace column with train mean) AUC drop,
  2. calibration table (predicted bin vs empirical stable rate),
  3. shape ranges from glass.shapes() for the supplement.

Uses exact Tsec features (the paper turf), not the 200-orbit screen.
Default 400 resonant + 200 random, seed 8, to match benchmark_table1.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kirk import features, glass  # noqa: E402

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


def binary_auc(scores, labels):
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((((pos[:, None] > neg[None, :]).sum()
                   + 0.5 * (pos[:, None] == neg[None, :]).sum())
                  / (len(pos) * len(neg))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-res", type=int, default=400)
    ap.add_argument("--n-rand", type=int, default=200)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "nam_spock_distilled.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "proofs" / "ablate.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_res, args.n_rand = 60, 30

    rng = np.random.default_rng(args.seed)
    resonant = np.load(DATA / "resonant_real.npz")["X"]
    random_ = np.load(DATA / "random_real.npz")["X"]
    rows = list(resonant[rng.choice(len(resonant), args.n_res, replace=False)])
    rows += list(random_[rng.choice(len(random_), args.n_rand, replace=False)])
    y = np.array([r[LABEL_COL] for r in rows], float)
    print(f"sample {len(rows)} systems, stable {y.mean():.3f}", flush=True)

    t0 = time.time()
    sims = [rebuild(r) for r in rows]
    out = features.generate_features(sims, n_jobs=args.jobs, singleWrap=True)
    print(f"Tsec features: {time.time()-t0:.0f}s", flush=True)

    model, meta = glass.load_artifact(args.model)
    feats = list(meta["feats"])
    X, stable = [], []
    for trios, st in out:
        f = trios[0]
        X.append([float(f[k]) if np.isfinite(float(f[k])) else 0.0 for k in feats])
        stable.append(bool(st))
    X = np.array(X, np.float32)
    stable = np.array(stable)
    keep = stable  # paper Sec.2: drop short-unstable
    Xk, yk = X[keep], y[keep]
    print(f"short-stable {keep.sum()}/{len(keep)}", flush=True)

    p = glass.predict(model, meta, Xk)
    base_auc = binary_auc(p, yk)
    base_acc = float(((p > 0.5) == (yk > 0.5)).mean())
    print(f"baseline AUC {base_auc:.4f} acc {base_acc:.4f} (n={len(yk)})", flush=True)

    # LOO: replace column with train mean (Xm), i.e. zero in normalized space.
    xm = np.asarray(meta["Xm"], dtype=np.float32)
    drops = {}
    for j, name in enumerate(feats):
        Xloo = Xk.copy()
        Xloo[:, j] = xm[j]
        ploo = glass.predict(model, meta, Xloo)
        auc = binary_auc(ploo, yk)
        drops[name] = {"auc": float(auc), "drop": float(base_auc - auc)}
        print(f"  LOO {name:14s} AUC {auc:.4f} drop {base_auc-auc:+.4f}", flush=True)

    # Calibration: 5 bins.
    bins = np.linspace(0, 1, 6)
    cal = []
    for b in range(5):
        m = (p >= bins[b]) & (p < bins[b + 1] if b < 4 else p <= bins[b + 1])
        if m.sum() == 0:
            cal.append({"bin": [float(bins[b]), float(bins[b + 1])], "n": 0,
                        "mean_p": None, "emp": None})
        else:
            cal.append({"bin": [float(bins[b]), float(bins[b + 1])], "n": int(m.sum()),
                        "mean_p": float(p[m].mean()), "emp": float(yk[m].mean())})
        print(f"  cal {bins[b]:.1f}-{bins[b+1]:.1f} n={cal[-1]['n']} "
              f"mean_p={cal[-1]['mean_p']} emp={cal[-1]['emp']}", flush=True)

    # Shape ranges for supplement.
    shapes = model.shapes(meta)
    ranges = {}
    for j, name in enumerate(feats):
        g, v = shapes[j]
        ranges[name] = {"lo": float(g[0]), "hi": float(g[-1]),
                        "f_lo": float(v[0]), "f_hi": float(v[-1]),
                        "span": float(v.max() - v.min())}

    payload = {
        "config": {"n_res": args.n_res, "n_rand": args.n_rand, "seed": args.seed,
                   "model": str(args.model.relative_to(ROOT))},
        "n": len(yk), "baseline_auc": base_auc, "baseline_acc": base_acc,
        "loo": drops, "calibration": cal, "shape_spans": ranges,
        "provenance": {"created": datetime.now(timezone.utc).isoformat()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
