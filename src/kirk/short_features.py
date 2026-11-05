"""Cheap features for the scheduler.

This is the short-window version of the SPOCK feature set. It integrates a
copy of the system for a few hundred orbits and measures how the eccentricity
vectors and the MEGNO move. It is not SPOCK's exact pipeline and it does not
claim to be. It is the cheap screen that decides whether you need the exact
one.

Only 3-planet systems. The scheduler is built around adjacent trios; do not
feed it a two-planet system and expect a classifier answer.
"""
import math

import numpy as np


def short_features(sim0, orbits=200, nout=60, seed=0):
    """Return (features, ok) for one 3-planet system.

    ok is False if the short integration blew up. You can still use the system;
    just treat it as unstable and skip the model.
    """
    sim = sim0.copy()
    try:
        sim.init_megno(seed=seed)
    except Exception:
        pass
    sim.integrator = "whfast"
    sim.dt = 0.05

    trio = [1, 2, 3]
    p0 = sim.particles
    srt = sorted(trio, key=lambda i: p0[i].a)
    gap_in = (p0[srt[1]].a - p0[srt[0]].a) / p0[srt[0]].a
    gap_out = (p0[srt[2]].a - p0[srt[1]].a) / p0[srt[1]].a
    if gap_in < gap_out:
        pairs = [("near", srt[0], srt[1]), ("far", srt[1], srt[2])]
    else:
        pairs = [("near", srt[1], srt[2]), ("far", srt[0], srt[1])]

    F = {}
    for lab, i1, i2 in pairs:
        F["EMcross" + lab] = (sim.particles[i2].a - sim.particles[i1].a) / sim.particles[i1].a

    # Secular timescale estimate. Same formula as the benchmark script.
    try:
        p1, p2, p3 = p0[trio[0]], p0[trio[1]], p0[trio[2]]
        minP0 = min([abs(p.P) for p in p0[1:4]])
        m_star = p0[0].m
        m_tot = p1.m + p2.m + p3.m
        mu1 = p1.m / m_tot
        mu3 = p3.m / m_tot
        a12 = p1.a / p2.a
        a23 = p2.a / p3.a
        ec12 = (a12 ** -0.25) * (a23 ** 0.625) * (1 - a12)
        ec23 = (a23 ** -0.5) * (a12 ** 0.125) * (1 - a23)
        w1 = abs((p3.n / (2 * math.pi)) * (m_tot / m_star) * (
            (mu1 / (mu1 + mu3)) / max(abs(ec12), 1e-9) ** 2
            + (mu3 / (mu1 + mu3)) / max(abs(ec23), 1e-9) ** 2))
        F["Tsec"] = float((2 * math.pi / w1) / minP0) if w1 > 0 else 0.0
    except Exception:
        F["Tsec"] = 0.0

    EMd = {lab: [] for lab, _, _ in pairs}
    EPd = {lab: [] for lab, _, _ in pairs}
    MMD = {lab: [] for lab, _, _ in pairs}
    MEG = []
    minP = min([p.P for p in sim.particles[1:]])

    try:
        for t in np.linspace(0, orbits * abs(minP), nout):
            sim.integrate(t)
            try:
                MEG.append(sim.megno())
            except Exception:
                MEG.append(2.0)  # neutral MEGNO when the integrator complains

            for lab, i1, i2 in pairs:
                q = sim.particles
                m1, m2 = q[i1].m, q[i2].m
                e1x = q[i1].e * math.cos(q[i1].pomega)
                e1y = q[i1].e * math.sin(q[i1].pomega)
                e2x = q[i2].e * math.cos(q[i2].pomega)
                e2y = q[i2].e * math.sin(q[i2].pomega)

                em = math.sqrt((e2x - e1x) ** 2 + (e2y - e1y) ** 2)
                EMd[lab].append(em)
                EPd[lab].append(math.sqrt(
                    (m1 * e1x + m2 * e2x) ** 2
                    + (m1 * e1y + m2 * e2y) ** 2) / (m1 + m2))

                # MMR strength. Zero unless the pair sits near a low-order
                # resonance. That is the intended behavior, not a bug.
                try:
                    n1, n2 = q[i1].n, q[i2].n
                    ratio = n2 / n1
                    best = 0.0
                    if 0 < ratio < 1:
                        for k in (1, 2):
                            for j in range(2, 12):
                                if abs(j / (j - k) - 1 / ratio) < 0.03 / (j - k + 1):
                                    emc = (q[i2].a - q[i1].a) / q[i1].a
                                    nres = (j * n2 - (j - k) * n1) / n1
                                    if nres != 0 and abs(emc) > 1e-12:
                                        s = abs(
                                            math.sqrt((m1 + m2) / q[0].m)
                                            * (em / emc) ** (k / 2.) / nres)
                                        if s > best and s != float("inf"):
                                            best = s
                    MMD[lab].append(best)
                except Exception:
                    MMD[lab].append(0.0)

        for lab in ("near", "far"):
            arr = np.array(MMD[lab][1:], float)
            finite = arr[np.isfinite(arr)]
            F["MMRstrength" + lab] = float(np.median(finite)) if finite.size else 0.0
            F["EMfracstd" + lab] = float(
                np.std(np.array(EMd[lab], float)) / max(F["EMcross" + lab], 1e-9))
            F["EPstd" + lab] = float(np.std(np.array(EPd[lab], float)))

        ma = np.array(MEG, float)
        F["MEGNO"] = float(np.median(ma[-nout // 10:]))
        F["MEGNOstd"] = float(ma[nout // 5:].std())

        for k in list(F.keys()):
            if not np.isfinite(F[k]):
                F[k] = 0.0
        return F, True
    except Exception as e:
        return {"err": str(e)[:80]}, False
