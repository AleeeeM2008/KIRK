"""Check the ported feature pipeline against upstream SPOCK.

This is the test that matters if you are replacing spock: run the same systems
through both implementations and compare every feature. It needs spock
installed (`pip install -e ".[spock]"`), but only for the cross-check.

Example:
    make setup-spock
    make check-spock

`make check-spock` sets the loader path reboundx needs. If you call this by
hand, export LD_LIBRARY_PATH to the venv's site-packages first.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rebound

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kirk import features  # noqa: E402


def rebuild(row):
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
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        from spock import FeatureClassifier
    except ImportError as exc:
        raise SystemExit("spock is not installed; install the 'spock' extra") from exc

    path = args.data / "resonant_real.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found; run scripts/prepare_data.py")

    rows = np.load(path)["X"]
    idx = np.random.default_rng(args.seed).choice(len(rows), args.n, replace=False)
    sims = [rebuild(rows[i]) for i in idx]

    ours = features.generate_features(sims, singleWrap=True)
    ref = FeatureClassifier().generate_features(sims)

    worst = 0.0
    mismatches = []
    for i, ((ot, os), (rt, rs)) in enumerate(zip(ours, ref)):
        if os != rs:
            mismatches.append((i, "stable", rs, os))
        if len(ot) != len(rt):
            mismatches.append((i, "trios", len(rt), len(ot)))
            continue
        for j, (a, b) in enumerate(zip(ot, rt)):
            for key in features.FEATS:
                if np.isnan(a[key]) and np.isnan(b[key]):
                    continue
                diff = abs(float(a[key]) - float(b[key]))
                worst = max(worst, diff)
                if not np.isclose(a[key], b[key], rtol=1e-6, atol=1e-8):
                    mismatches.append((i, j, key, b[key], a[key]))

    print(f"compared {len(sims)} systems, {len(mismatches)} mismatches, "
          f"worst abs feature diff {worst:.3e}")
    if mismatches:
        for m in mismatches[:10]:
            print("  ", m)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
