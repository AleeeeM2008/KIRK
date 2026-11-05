"""SPOCK 2.0 feature pipeline, without the spock dependency.

Adapted from dtamayo/spock (GPL-3.0): simsetup.py, features.py and
feature_functions.py. Same recipe, same MEGNO seed, same integrator settings,
same trio logic. If you change something, compare against the installed spock
package before trusting the numbers.

The only thing replaced is the classifier: this module feeds the glass NAM
instead of XGBoost. Feature extraction itself is SPOCK's.
"""
from collections import OrderedDict
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np
import rebound

# Same order as SPOCK's classifier. The glass models were trained on this
# order, so do not shuffle it without retraining.
FEATS = ['EMcrossnear', 'EMfracstdnear', 'EPstdnear', 'MMRstrengthnear',
         'EMcrossfar', 'EMfracstdfar', 'EPstdfar', 'MMRstrengthfar',
         'MEGNO', 'MEGNOstd', 'Tsec']


# ---------------------------------------------------------------------------
# Simulation setup
# ---------------------------------------------------------------------------

def _n_real(sim):
    """Number of real particles. REBOUND 5 dropped N_real."""
    return int(getattr(sim, "N_real", sim.N - getattr(sim, "N_var", 0)))

def check_hyperbolic(sim):
    orbits = sim.orbits()
    return bool(np.min([o.a for o in orbits]) < 0)


def check_valid_sim(sim):
    assert isinstance(sim, rebound.Simulation)
    masses = np.array([p.m for p in sim.particles[:_n_real(sim)]])
    if np.min(masses) < 0:
        raise AttributeError("negative particle mass")
    if np.max(masses) != masses[0]:
        raise AttributeError("particle 0 must be the primary (dominant mass)")


def set_integrator_and_timestep(sim):
    """Wisdom 2015 says 0.05 of the shortest periastron time. Keep it."""
    periods = np.array([p.P for p in sim.particles[1:_n_real(sim)]])
    eccs = np.array([p.e for p in sim.particles[1:_n_real(sim)]])
    if np.max(eccs) < 1:
        min_tperi = np.min(periods * (1 - eccs) ** 1.5 / np.sqrt(1 + eccs))
        sim.dt = 0.05 * min_tperi
    else:
        sim.dt = np.nan  # hyperbolic; the time series will be NaN

    if np.max(eccs) > 0.99:
        sim.integrator = "ias15"
    else:
        sim.integrator = "whfast"


def setup_sim(sim, megno=True, safe_mode=1):
    """Copy the system and put it in the form SPOCK's features expect.

    Rotating to the invariable plane is not cosmetic. It removes the observer
    frame, so an input with i=90 does not look like a different system.
    """
    sim = sim.copy()
    check_valid_sim(sim)

    rot = rebound.Rotation.to_new_axes(newz=sim.angular_momentum())
    sim.rotate(rot)

    if any(orb.inc > 0.2 for orb in sim.orbits()):
        import warnings
        warnings.warn("inclination > 0.2 relative to the invariable plane; "
                      "outside SPOCK's training range")

    try:
        sim.collision = "line"
    except Exception:
        sim.collision = "direct"

    max_d = np.array([p.d for p in sim.particles[1:_n_real(sim)]]).max()
    sim.exit_max_distance = 100 * max_d

    if not hasattr(sim, "ri_whfast"):
        raise RuntimeError(
            "kirk.features needs REBOUND 4.x (the WHFast configuration API). "
            "Install rebound==4.6.0, as the README does.")
    sim.ri_whfast.keep_unsynchronized = 0
    sim.ri_whfast.safe_mode = safe_mode

    if getattr(sim, "N_var", 0) == 0 and megno:
        sim.init_megno(seed=0)

    set_integrator_and_timestep(sim)

    # Collision radii are Hill radii. The exact value does not matter for the
    # classifier; this keeps the extreme mass-ratio cases sane.
    for p in sim.particles[1:_n_real(sim)]:
        p.r = p.a * (p.m / 3.0 / sim.particles[0].m) ** (1.0 / 3.0)

    sim.move_to_com()
    return sim


# ---------------------------------------------------------------------------
# Resonance and Hill helpers
# ---------------------------------------------------------------------------

def farey_sequence(n):
    """Farey sequence as (numerator, denominator) pairs. From celmech."""
    a, b, c, d = 0, 1, 1, n
    sequence = [(a, b)]
    while c <= n:
        k = int((n + b) / d)
        a, b, c, d = c, d, k * c - a, k * d - b
        sequence.append((a, b))
    return sequence


def resonant_period_ratios(min_per_ratio, max_per_ratio, order):
    """Period ratios of all resonances up to `order` in the given range."""
    if min_per_ratio < 0:
        raise AttributeError("min_per_ratio cannot be negative")
    if max_per_ratio >= 1:
        raise AttributeError("max_per_ratio must be < 1")

    min_j = int(np.floor(1.0 / (1.0 - min_per_ratio)))
    max_j = int(np.ceil(1.0 / (1.0 - max_per_ratio)))
    ratios = [(min_j - 1, min_j)]
    for j in range(min_j, max_j):
        ratios += [(x[1] * j - x[1] + x[0], x[1] * j + x[0])
                   for x in farey_sequence(order)[1:]]
    ratios = np.array(ratios)
    mask = np.array([min_per_ratio < x[0] / float(x[1]) < max_per_ratio
                     for x in ratios])
    return ratios[mask]


def get_pairs(sim, trio):
    """Label the near and far adjacent pairs in a trio."""
    ps = sim.particles
    sorted_indices = sorted(trio, key=lambda i: ps[i].a)
    em_in = ((ps[sorted_indices[1]].a - ps[sorted_indices[0]].a)
             / ps[sorted_indices[0]].a)
    em_out = ((ps[sorted_indices[2]].a - ps[sorted_indices[1]].a)
              / ps[sorted_indices[1]].a)

    if em_in < em_out:
        return [["near", sorted_indices[0], sorted_indices[1]],
                ["far", sorted_indices[1], sorted_indices[2]]]
    return [["near", sorted_indices[1], sorted_indices[2]],
            ["far", sorted_indices[0], sorted_indices[1]]]


def find_strongest_MMR(sim, i1, i2):
    """Strongest low-order MMR for one pair. Returns (j, k, strength)."""
    ps = sim.particles
    n1, n2 = ps[i1].n, ps[i2].n
    m1, m2 = ps[i1].m / ps[0].m, ps[i2].m / ps[0].m
    ratio = n2 / n1
    delta = 0.03

    if ratio < 0 or ratio > 1:
        return np.nan, np.nan, np.nan

    lo = max(ratio - delta, 0.0)
    hi = min(ratio + delta, 0.99)
    resonances = resonant_period_ratios(lo, hi, order=2)

    em = np.sqrt(
        (ps[i1].e * np.cos(ps[i1].pomega) - ps[i2].e * np.cos(ps[i2].pomega)) ** 2
        + (ps[i1].e * np.sin(ps[i1].pomega) - ps[i2].e * np.sin(ps[i2].pomega)) ** 2)
    em_cross = (ps[i2].a - ps[i1].a) / ps[i1].a

    j, k, best = np.nan, np.nan, 0.0
    for a, b in resonances:
        nres = (b * n2 - a * n1) / n1
        if nres == 0:
            s = np.inf
        else:
            s = np.abs(np.sqrt(m1 + m2) * (em / em_cross) ** ((b - a) / 2.0) / nres)
        if s > best:
            j, k, best = b, b - a, s
    if best == 0:
        best = np.nan
    return j, k, best


def hillfac(sim, i1=1, i2=2):
    """Hill stability ratio for two planets. >1 means Hill stable."""
    ps = sim.particles
    m0, m1, m2 = ps[0].m, ps[i1].m, ps[i2].m
    total = m1 + m2 + m0
    mass_prod = m1 * m2 + m1 * m0 + m2 * m0
    G = sim.G

    angular = np.linalg.norm(sim.angular_momentum())
    energy = sim.energy()
    p_over_a = -((2 * total) / (G ** 2 * mass_prod ** 3)) * angular ** 2 * energy
    crit_minus_one = (3 ** (4 / 3) * (m1 * m2)) / (m0 ** (2 / 3) * (m1 + m2) ** (4 / 3))

    sign = np.sign(p_over_a - 1)
    return sign * (abs(p_over_a - 1) / crit_minus_one) ** 0.5


# ---------------------------------------------------------------------------
# Secular timescale
# ---------------------------------------------------------------------------

def get_min_secT_trio(sim, trio):
    """Fastest secular timescale of one trio, in inner-planet orbits."""
    ps = sim.particles
    p1, p2, p3 = ps[trio[0]], ps[trio[1]], ps[trio[2]]
    min_p = np.min([np.abs(p.P) for p in sim.particles[1:_n_real(sim)]])
    m_star = ps[0].m
    m_tot = p1.m + p2.m + p3.m
    mu1 = p1.m / m_tot
    mu3 = p3.m / m_tot
    a12 = p1.a / p2.a
    a23 = p2.a / p3.a

    ec12 = a12 ** (-1 / 4) * a23 ** (3 / 4) * a23 ** (-1 / 8) * (1 - a12)
    ec23 = a23 ** (-1 / 2) * a12 ** (1 / 8) * (1 - a23)
    w1 = np.abs((p3.n / (2 * np.pi)) * (m_tot / m_star)
                * ((mu1 / (mu1 + mu3)) / ec12 ** 2
                   + (mu3 / (mu1 + mu3)) / ec23 ** 2))
    return (2 * np.pi / w1) / min_p


def get_min_secT(sim):
    secs = []
    for trio in [[j, j + 1, j + 2] for j in range(1, _n_real(sim) - 2)]:
        secs.append(get_min_secT_trio(sim, trio))
    return min(secs)


# ---------------------------------------------------------------------------
# Trio features
# ---------------------------------------------------------------------------

class Trio:
    """Feature container for one adjacent trio.

    It keeps the time series during the short integration and reduces it to
    the 11 numbers the classifier sees.
    """

    def __init__(self, trio_indices, sim, nout):
        self.trio = trio_indices
        self.pairs = get_pairs(sim, trio_indices)
        self.running = OrderedDict()
        self.running["time"] = [np.nan] * nout
        self.running["MEGNO"] = [np.nan] * nout
        for label in ("near", "far"):
            self.running["EM" + label] = [np.nan] * nout
            self.running["EP" + label] = [np.nan] * nout
            self.running["MMRstrength" + label] = [np.nan] * nout

        self.features = OrderedDict()
        for label in ("near", "far"):
            self.features["EMcross" + label] = np.nan
            self.features["EMfracstd" + label] = np.nan
            self.features["EPstd" + label] = np.nan
            self.features["MMRstrength" + label] = np.nan
        self.features["MEGNO"] = np.nan
        self.features["MEGNOstd"] = np.nan

    def fill_starting_features(self, sim):
        ps = sim.particles
        for label, i1, i2 in self.pairs:
            self.features["EMcross" + label] = (ps[i2].a - ps[i1].a) / ps[i1].a
        self.features["Tsec"] = get_min_secT_trio(sim, self.trio)

    def fill_tseries_entry(self, sim, min_p, i):
        ps = sim.particles
        for label, i1, i2 in self.pairs:
            m1, m2 = ps[i1].m, ps[i2].m
            e1x = ps[i1].e * np.cos(ps[i1].pomega)
            e1y = ps[i1].e * np.sin(ps[i1].pomega)
            e2x = ps[i2].e * np.cos(ps[i2].pomega)
            e2y = ps[i2].e * np.sin(ps[i2].pomega)

            self.running["time"][i] = sim.t / min_p
            self.running["EM" + label][i] = np.sqrt(
                (e2x - e1x) ** 2 + (e2y - e1y) ** 2)
            self.running["EP" + label][i] = np.sqrt(
                (m1 * e1x + m2 * e2x) ** 2
                + (m1 * e1y + m2 * e2y) ** 2) / (m1 + m2)
            self.running["MMRstrength" + label][i] = find_strongest_MMR(sim, i1, i2)[2]

        self.running["MEGNO"][i] = sim.megno()

    def fill_final_features(self, sim):
        nout = len(self.running["MEGNO"])
        megno = np.array(self.running["MEGNO"], dtype=float)
        if not np.isnan(megno).any():
            self.features["MEGNO"] = np.median(megno[-(nout // 10):])
            self.features["MEGNOstd"] = np.std(megno[nout // 5:])

        for label in ("near", "far"):
            mmr = np.array(self.running["MMRstrength" + label], dtype=float)
            self.features["MMRstrength" + label] = np.median(mmr[1:])
            em = np.array(self.running["EM" + label], dtype=float)
            ep = np.array(self.running["EP" + label], dtype=float)
            self.features["EMfracstd" + label] = (
                np.std(em) / self.features["EMcross" + label])
            self.features["EPstd" + label] = np.std(ep)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _generate_features(sim, nbody_tmax=1e6):
    """One system -> ([trio feature dicts], stable-after-short-run)."""
    sim = setup_sim(sim)

    n_orbits = get_min_secT(sim)
    if n_orbits > nbody_tmax:
        import warnings
        warnings.warn(f"min secular timescale > {nbody_tmax} orbits; clamping")
        n_orbits = nbody_tmax

    nout = int((n_orbits / 1e4) * 80)
    trios = [Trio([j, j + 1, j + 2], sim, nout)
             for j in range(1, _n_real(sim) - 2)]
    for trio in trios:
        trio.fill_starting_features(sim)

    min_p = np.min([np.abs(p.P) for p in sim.particles[1:_n_real(sim)]])
    times = np.linspace(0, n_orbits * min_p, nout)
    stable = True

    for i, time in enumerate(times):
        try:
            sim.integrate(time, exact_finish_time=0)
        except (rebound.Collision, rebound.Escape):
            stable = False
            break
        if sim._status == 4 or sim._status == 7:
            stable = False
            break
        for trio in trios:
            trio.fill_tseries_entry(sim, min_p, i)

    if stable:
        for trio in trios:
            trio.fill_final_features(sim)

    return [trio.features for trio in trios], stable


def generate_features(sims, n_jobs=-1, nbody_tmax=1e6, singleWrap=False,
                      Nbodytmax=None):
    """SPOCK-style generate_features.

    One sim returns ([trio dicts], stable). A list returns a list of those.
    `Nbodytmax` is the upstream spelling; both names work.
    """
    if Nbodytmax is not None:
        nbody_tmax = Nbodytmax
    if isinstance(sims, rebound.Simulation):
        sims = [sims]
    if n_jobs == -1:
        n_jobs = cpu_count()

    with ThreadPool(n_jobs) as pool:
        res = list(pool.map(lambda s: _generate_features(s, nbody_tmax), sims))

    if len(sims) == 1 and not singleWrap:
        return res[0]
    return res


def _feature_vector(feat):
    return np.array([float(feat[k]) if np.isfinite(float(feat[k])) else 0.0
                     for k in FEATS], dtype=np.float32)


def feature_rows(sims, n_jobs=-1, nbody_tmax=1e6, fill=0.0,
                 Nbodytmax=None):
    """Flatten every trio of every system.

    Returns (X, stable_per_system, owner). X has one row per trio in FEATS
    order. owner[i] is the system index for row i. Use this when you need the
    min-over-trios rule.
    """
    if Nbodytmax is not None:
        nbody_tmax = Nbodytmax
    res = generate_features(sims, n_jobs=n_jobs, nbody_tmax=nbody_tmax,
                            singleWrap=True)
    X, stable, owner = [], [], []
    for system_index, (trios, st) in enumerate(res):
        stable.append(bool(st))
        for feat in trios:
            X.append([float(feat[k]) if np.isfinite(float(feat[k])) else fill
                      for k in FEATS])
            owner.append(system_index)
    return (np.array(X, np.float32), np.array(stable, bool),
            np.array(owner, int))


def feature_matrix(sims, n_jobs=-1, nbody_tmax=1e6, fill=0.0,
                   Nbodytmax=None):
    """First trio per system, for 3-planet/quick use.

    Kept for the quickstart. For 4+ planet systems use feature_rows() or
    predict_stable(), which take the minimum over trios like SPOCK.
    """
    if Nbodytmax is not None:
        nbody_tmax = Nbodytmax
    res = generate_features(sims, n_jobs=n_jobs, nbody_tmax=nbody_tmax,
                            singleWrap=True)
    X, stable = [], []
    for trios, st in res:
        if not trios:
            raise ValueError("no trios: this helper is for 3+ planet systems; "
                             "use predict_stable() for 0/1/2 planets")
        X.append(_feature_vector(trios[0]))
        stable.append(bool(st))
    return np.array(X, np.float32), np.array(stable, bool)


_MODEL = None
_MODEL_META = None
_MODEL_PATH = None
_REGRESSOR = None
_REGRESSOR_META = None
_REGRESSOR_PATH = None


def _classifier(model_path=None):
    global _MODEL, _MODEL_META, _MODEL_PATH
    from pathlib import Path
    from kirk import glass
    path = Path(model_path) if model_path is not None else (
        Path(__file__).resolve().parents[2] / "models" / "nam_spock_distilled.pt")
    if _MODEL is None or str(path) != _MODEL_PATH:
        _MODEL, _MODEL_META = glass.load_artifact(path)
        _MODEL_PATH = str(path)
    return _MODEL, _MODEL_META


def _regressor(model_path=None):
    global _REGRESSOR, _REGRESSOR_META, _REGRESSOR_PATH
    from pathlib import Path
    from kirk import glass
    path = Path(model_path) if model_path is not None else (
        Path(__file__).resolve().parents[2] / "models" / "nam_spock_deepreg.pt")
    if _REGRESSOR is None or str(path) != _REGRESSOR_PATH:
        _REGRESSOR, _REGRESSOR_META = glass.load_artifact(path)
        _REGRESSOR_PATH = str(path)
    return _REGRESSOR, _REGRESSOR_META


def predict_stable(sims, n_jobs=-1, nbody_tmax=1e6, singleWrap=False,
                   model_path=None, Nbodytmax=None):
    """Probability of stability, same call shape as SPOCK's predict_stable.

    0/1 planets: 1.0. Two planets: Hill rule. Three or more: classifier on
    every adjacent trio, then the minimum probability. A system that goes
    unstable during the short integration gets 0.
    """
    from kirk import glass

    if Nbodytmax is not None:
        nbody_tmax = Nbodytmax
    if isinstance(sims, rebound.Simulation):
        sim_list = [sims]
        single = True
    else:
        sim_list = list(sims)
        single = False
    if not sim_list:
        return np.array([])

    n_planets = [_n_real(s) - 1 for s in sim_list]
    probs = np.ones(len(sim_list), dtype=float)

    for i, n in enumerate(n_planets):
        if n <= 1:
            probs[i] = 1.0
        elif n == 2:
            probs[i] = 1.0 if hillfac(sim_list[i]) > 1 else 0.0

    three_plus = [i for i, n in enumerate(n_planets) if n >= 3]
    if three_plus:
        X, stable, owner = feature_rows(
            [sim_list[i] for i in three_plus], n_jobs=n_jobs,
            nbody_tmax=nbody_tmax)
        model, meta = _classifier(model_path)
        if len(X):
            trio_probs = glass.predict(model, meta, X)
        else:
            trio_probs = np.array([])

        for local, system_index in enumerate(three_plus):
            rows = np.where(owner == local)[0]
            if not stable[local] or rows.size == 0:
                probs[system_index] = 0.0
            else:
                probs[system_index] = float(np.min(trio_probs[rows]))

    if single and not singleWrap:
        return probs[0]
    return probs


def predict_instability_time(sims, n_jobs=-1, nbody_tmax=1e6, singleWrap=False,
                             model_path=None, Nbodytmax=None):
    """Estimated instability time in orbits of the inner planet.

    This uses the distilled DeepRegressor NAM. It returns a point estimate,
    not SPOCK's posterior percentiles. For 0/1 planets: inf. Two planets:
    inf if Hill stable, 0 otherwise. Three or more: the shortest trio time,
    and 0 if the system died during the short integration.
    """
    from kirk import glass

    if Nbodytmax is not None:
        nbody_tmax = Nbodytmax
    if isinstance(sims, rebound.Simulation):
        sim_list = [sims]
        single = True
    else:
        sim_list = list(sims)
        single = False
    if not sim_list:
        return np.array([])

    n_planets = [_n_real(s) - 1 for s in sim_list]
    times = np.full(len(sim_list), np.inf, dtype=float)

    for i, n in enumerate(n_planets):
        if n == 2:
            times[i] = np.inf if hillfac(sim_list[i]) > 1 else 0.0

    three_plus = [i for i, n in enumerate(n_planets) if n >= 3]
    if three_plus:
        X, stable, owner = feature_rows(
            [sim_list[i] for i in three_plus], n_jobs=n_jobs,
            nbody_tmax=nbody_tmax)
        model, meta = _regressor(model_path)
        log_times = glass.predict_logit(model, meta, X) if len(X) else np.array([])
        # The regressor was trained on log10(t) with t in orbits. Keep the
        # output in a sane range; 1e12 orbits is already a non-answer.
        values = 10.0 ** np.clip(log_times, 0.0, 12.0)

        for local, system_index in enumerate(three_plus):
            rows = np.where(owner == local)[0]
            if not stable[local] or rows.size == 0:
                times[system_index] = 0.0
            else:
                times[system_index] = float(np.min(values[rows]))

    if single and not singleWrap:
        return float(times[0])
    return times
