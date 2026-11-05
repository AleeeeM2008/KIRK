"""Generate initial conditions for training and tests.

Four families:

  resonant    sampled from SPOCK's inputresonantparams.csv, so the resonant
              manifold looks like their training distribution
  random      wide, non-resonant, the easy negative class
  jupiter     planets spaced in mutual Hill radii; fast, violent, good tests
  ttv         Kepler-307-ish inner pair with e- pinned by the TTV fit

This reproduces the distribution family, not their exact generator. If you
need bit-for-bit their data, use generate_training_data/generate_FC_data.py
upstream. Labels are not made here: use the 1e9 CSV for real archives, or
integrate for synthetic systems.
"""
from pathlib import Path

import numpy as np
import rebound

# Their resonant file: seed,pairindex,j,k,Zstar,libfac,Zcom
# We sample rows from it, which is why the initial conditions are not
# uniformly random. That matters; the unstable boundary lives near the MMRs.
DEFAULT_CSV = str(Path(__file__).resolve().parents[1] / "data" / "inputresonantparams.csv")


def resonant_from_csv(n, seed=0, csv_path=DEFAULT_CSV):
    """Sample n resonant chains. Falls back to a generic 3:2 chain if the CSV
    is missing, but then you are not testing the same distribution, so do not
    report those numbers as if you were."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    try:
        tab = pd.read_csv(csv_path)
    except Exception:
        tab = None

    sims = []
    for _ in range(n):
        sim = rebound.Simulation()
        sim.G = 1.0
        sim.add(m=1.0)

        if tab is not None:
            row = tab.iloc[rng.integers(len(tab))]
            j, k = int(row["j"]), int(row["k"])
            ratio = j / (j - k) * rng.uniform(0.99, 1.01)
            e0 = float(row["Zcom"]) * rng.uniform(0.5, 1.5)
        else:
            ratio = rng.uniform(1.15, 1.6)
            e0 = 0.03

        P1 = 1.0
        P2 = P1 * ratio
        P3 = P2 * rng.uniform(1.15, 1.6)
        for P in (P1, P2, P3):
            sim.add(m=1e-5, P=P, e=max(rng.normal(e0, e0 / 2), 0.002),
                    pomega=rng.uniform(0, 2 * np.pi),
                    l=rng.uniform(0, 2 * np.pi))
        sim.move_to_com()
        sims.append(sim)
    return sims


def random_wide(n, seed=1):
    """Wide spacings and low masses. Mostly stable, which is the point."""
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n):
        sim = rebound.Simulation()
        sim.G = 1.0
        sim.add(m=1.0)
        for P in (1.0, rng.uniform(1.1, 1.8), rng.uniform(1.5, 3.0)):
            sim.add(m=10 ** rng.uniform(-6, -4), P=P, e=rng.uniform(0, 0.1),
                    pomega=rng.uniform(0, 2 * np.pi),
                    l=rng.uniform(0, 2 * np.pi))
        sim.move_to_com()
        sims.append(sim)
    return sims


def jupiter_hill(n, tight=True, seed=2):
    """Jupiter-mass planets packed in mutual Hill radii.

    This is the hard case for a model trained on small planets: the masses are
    off-distribution and the encounters are fast. Do not tune on it and then
    claim it as a blind test.
    """
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n):
        K = rng.uniform(3.0, 4.5) if tight else rng.uniform(6.0, 9.0)
        m_p = 1e-3
        a1 = 1.0
        rh = a1 * (m_p / 3.0) ** (1 / 3.0)
        a2 = a1 + K * rh
        a3 = a2 + K * rh

        sim = rebound.Simulation()
        sim.G = 1.0
        sim.add(m=1.0)
        for a in (a1, a2, a3):
            sim.add(m=m_p, a=a, e=rng.uniform(0, 0.02),
                    pomega=rng.uniform(0, 2 * np.pi),
                    l=rng.uniform(0, 2 * np.pi))
        sim.move_to_com()
        sims.append(sim)
    return sims


def ttv_narrow(n, seed=3):
    """Kepler-307 style: inner pair near 5:4, e- pinned by the TTV fit.

    The point is that a single eccentricity combination is fixed while the
    other one is free. Models trained on random phases find this strange.
    """
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n):
        sim = rebound.Simulation()
        sim.G = 1.0
        sim.add(m=1.0)

        e_minus = rng.normal(0.0088, 0.0004)
        e_plus = rng.uniform(0.0, 0.08)
        e1 = max(abs(e_plus - e_minus / 2 + rng.normal(0, 0.002)), 0.001)
        e2 = max(abs(e_plus + e_minus / 2 + rng.normal(0, 0.002)), 0.001)

        sim.add(m=1e-5, P=1.0, e=e1,
                pomega=rng.uniform(0, 2 * np.pi), l=rng.uniform(0, 2 * np.pi))
        sim.add(m=1e-5, P=1.255, e=e2,
                pomega=rng.uniform(0, 2 * np.pi), l=rng.uniform(0, 2 * np.pi))
        sim.add(m=1e-5, P=rng.uniform(1.6, 2.2), e=rng.uniform(0.01, 0.06),
                pomega=rng.uniform(0, 2 * np.pi), l=rng.uniform(0, 2 * np.pi))
        sim.move_to_com()
        sims.append(sim)
    return sims


if __name__ == "__main__":
    for name, fn in [("resonant", lambda: resonant_from_csv(10)),
                     ("random", lambda: random_wide(10)),
                     ("jupiter", lambda: jupiter_hill(10)),
                     ("ttv", lambda: ttv_narrow(5))]:
        sims = fn()
        print(name, len(sims), "built, e.g. N =", len(sims[0].particles) - 1, "planets")
