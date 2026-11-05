"""Sanity check on real systems, not training data.

TRAPPIST-1 with three phase draws and Kepler-36. This script is for "does it
behave like a human would expect", not for benchmark numbers. The numbers are
in proofs/table1.json.

If upstream `spock` is installed, it also prints SPOCK's value next to KIRK's.
That is a cross-check, not a requirement.
"""
import math
import sys
from pathlib import Path

import numpy as np
import rebound

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ME = 3.003e-6
TRAPPIST = [(1.37, 1.51087), (1.31, 2.42182), (0.39, 4.04961), (0.69, 6.09962),
            (1.04, 9.20669), (1.32, 12.35294), (0.33, 18.767)]


def add_planet(sim, m_earth, P_days, e=0.01, l=0.0, pomega=0.0):
    """Add a planet by period in days. Units: star mass, AU, years."""
    a = ((P_days / 365.25) ** 2 * sim.particles[0].m) ** (1 / 3.)
    sim.add(m=m_earth * ME, a=a, e=e, pomega=pomega, l=l)


def build_trappist(seed=0):
    rng = np.random.default_rng(seed)
    sim = rebound.Simulation()
    sim.G = 4 * math.pi ** 2
    sim.add(m=0.0898)
    for m, P in TRAPPIST:
        add_planet(sim, m, P, e=0.01, l=rng.uniform(0, 2 * np.pi))
    sim.move_to_com()
    return sim


def build_k36():
    sim = rebound.Simulation()
    sim.G = 4 * math.pi ** 2
    sim.add(m=1.07)
    add_planet(sim, 4.45, 13.839, e=0.02, l=0.5)
    add_planet(sim, 8.08, 16.238, e=0.01, l=2.0)
    sim.move_to_com()
    return sim


def trio(sim, indices):
    """Copy one adjacent trio into its own sim. The star comes along."""
    sub = rebound.Simulation()
    sub.G = sim.G
    sub.add(m=sim.particles[0].m)
    for i in indices:
        p = sim.particles[i]
        sub.add(m=p.m, a=p.a, e=p.e, pomega=p.pomega, l=p.l)
    sub.move_to_com()
    return sub


def main():
    from kirk import features

    try:
        from spock import FeatureClassifier
        upstream = FeatureClassifier()
    except ImportError:
        upstream = None

    print("TRAPPIST-1: one adjacent trio at a time")
    for seed in (0, 1, 2):
        sim = build_trappist(seed)
        for indices in ([1, 2, 3], [2, 3, 4], [3, 4, 5]):
            sub = trio(sim, indices)
            p_kirk = float(features.predict_stable(sub))
            if upstream is None:
                print(f"  seed {seed} trio {indices}: KIRK {p_kirk:.3f}")
            else:
                p_spock = float(upstream.predict_stable(sub))
                print(f"  seed {seed} trio {indices}: SPOCK {p_spock:.3f}  "
                      f"KIRK {p_kirk:.3f}")

    print("Kepler-36: two planets, both fall back to the Hill rule.")
    k36 = build_k36()
    p_kirk = float(features.predict_stable(k36))
    if upstream is None:
        print(f"  KIRK {p_kirk:.3f}")
    else:
        p_spock = float(upstream.predict_stable(k36))
        print(f"  SPOCK {p_spock:.3f}  KIRK {p_kirk:.3f}")


if __name__ == "__main__":
    main()
