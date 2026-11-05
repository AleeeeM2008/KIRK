"""Reproduce SPOCK 2.0 Table 1 and add our glass rows.

Four cells: whether the features come from a fixed 1e4-orbit run or the
Tsec-length run, and whether Tsec is included. AUC and FPR at 0.5 against the
real 1e9 CSV labels, with the paper's Sec. 2 exclusions.

This is a script. Run it, do not import it.

Needs the SPOCK archives (run scripts/prepare_data.py first):
  data/{resonant_real,random_real}.npz
Override the directory with KIRK_DATA if yours lives elsewhere.
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rebound
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kirk import features  # noqa: E402
DATA = Path(os.environ.get("KIRK_DATA", ROOT / "data"))

# Column layout of the real archives:
#   0                      star mass
#   1..18                  3 planets x (m, a, e, pomega, Omega, l)
#   19                     1e9-orbit stability label
#   20                     Tsec-like value; not used here
LABEL_COL = 19


def load_teacher():
    """Read the XGBoost JSON once. The file is downloaded by download_assets.py."""
    path = Path(os.environ.get("KIRK_TEACHER_JSON",
                               ROOT / "models" / "featureclassifier.json"))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/download_assets.py.")
    with open(path) as f:
        doc = json.load(f)
    learner = doc["learner"]
    feats = learner["feature_names"]
    trees = learner["gradient_booster"]["model"]["trees"]
    base = float(learner["learner_model_param"]["base_score"])
    base_m = math.log(base / (1 - base))
    return feats, trees, base_m


def teacher_logit(feat, feats, trees, base_m):
    """Teacher log-odds. Same walk as src/kirk/teacher.py, kept local so this
    script does not need the package installed."""
    x = [float(feat.get(k, float("nan"))) for k in feats]
    m = base_m
    for t in trees:
        left = t["left_children"]
        right = t["right_children"]
        split_index = t["split_indices"]
        split_value = t["split_conditions"]
        weight = t["base_weights"]
        default_left = t["default_left"]

        i = 0
        while True:
            if left[i] == -1:
                m += weight[i]
                break
            v = x[split_index[i]]
            if v != v:
                i = left[i] if default_left[i] == 1 else right[i]
            else:
                i = left[i] if v < split_value[i] else right[i]
    return m


def feats_fixed(sim0, nout=80):
    """SPOCK-1 style features over a fixed 1e4 orbits.

    This is our own approximation, not their code. It exists to fill the
    "1e4" rows of the table. If you want the real old features, run SPOCK 1.
    """
    sim = sim0.copy()
    try:
        sim.init_megno(seed=0)
    except Exception:
        pass
    sim.integrator = "whfast"
    sim.dt = 0.05

    trio = [1, 2, 3]
    p0 = sim.particles
    srt = sorted(trio, key=lambda i: p0[i].a)
    Ein = (p0[srt[1]].a - p0[srt[0]].a) / p0[srt[0]].a
    Eout = (p0[srt[2]].a - p0[srt[1]].a) / p0[srt[1]].a
    if Ein < Eout:
        pairs = [("near", srt[0], srt[1]), ("far", srt[1], srt[2])]
    else:
        pairs = [("near", srt[1], srt[2]), ("far", srt[0], srt[1])]

    F = {}
    for lab, i1, i2 in pairs:
        F["EMcross" + lab] = (sim.particles[i2].a - sim.particles[i1].a) / sim.particles[i1].a

    # Secular timescale. Cheap version of their get_min_secT.
    try:
        p1, p2, p3 = p0[trio[0]], p0[trio[1]], p0[trio[2]]
        minP0 = min([abs(p.P) for p in p0[1:4]])
        mS = p0[0].m
        mt = p1.m + p2.m + p3.m
        mu1 = p1.m / mt
        mu3 = p3.m / mt
        a12 = p1.a / p2.a
        a23 = p2.a / p3.a
        ec12 = (a12 ** -0.25) * (a23 ** 0.625) * (1 - a12)
        ec23 = (a23 ** -0.5) * (a12 ** 0.125) * (1 - a23)
        w1 = abs((p3.n / (2 * math.pi)) * (mt / mS) * (
            (mu1 / (mu1 + mu3)) / max(abs(ec12), 1e-9) ** 2 +
            (mu3 / (mu1 + mu3)) / max(abs(ec23), 1e-9) ** 2))
        F["Tsec"] = float((2 * math.pi / w1) / minP0) if w1 > 0 else 0.0
    except Exception:
        F["Tsec"] = 0.0

    EMd = {lab: [] for lab, _, _ in pairs}
    EPd = {lab: [] for lab, _, _ in pairs}
    MMD = {lab: [] for lab, _, _ in pairs}
    MEG = []
    minP = min([p.P for p in sim.particles[1:]])

    try:
        for t in np.linspace(0, 10000 * abs(minP), nout):
            sim.integrate(t)
            try:
                MEG.append(sim.megno())
            except Exception:
                # megno can fail on the first step. 2.0 is the neutral value.
                MEG.append(2.0)

            for lab, i1, i2 in pairs:
                q = sim.particles
                m1, m2 = q[i1].m, q[i2].m
                e1x = q[i1].e * math.cos(q[i1].pomega)
                e1y = q[i1].e * math.sin(q[i1].pomega)
                e2x = q[i2].e * math.cos(q[i2].pomega)
                e2y = q[i2].e * math.sin(q[i2].pomega)

                EMv = math.sqrt((e2x - e1x) ** 2 + (e2y - e1y) ** 2)
                EMd[lab].append(EMv)
                EPd[lab].append(math.sqrt(
                    (m1 * e1x + m2 * e2x) ** 2 +
                    (m1 * e1y + m2 * e2y) ** 2) / (m1 + m2))

                # MMR strength. Only pairs sitting near a low-order resonance
                # get a nonzero value; everything else stays at zero.
                try:
                    n1, n2 = q[i1].n, q[i2].n
                    ratio = n2 / n1
                    best = 0.0
                    if 0 < ratio < 1:
                        for k in (1, 2):
                            for j in range(2, 12):
                                if abs(j / (j - k) - 1 / ratio) < 0.03 / (j - k + 1):
                                    EMc = (q[i2].a - q[i1].a) / q[i1].a
                                    nres = (j * n2 - (j - k) * n1) / n1
                                    if nres != 0 and abs(EMc) > 1e-12:
                                        s = abs(
                                            math.sqrt((m1 + m2) / q[0].m) *
                                            (EMv / EMc) ** (k / 2.) / nres)
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
        return {"err": str(e)[:60]}, False


def rebuild(row):
    """Rebuild a simulation from one archive row. G=1, as in their data."""
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=float(row[0]))
    for j in range(3):
        m, a, e, pomega, Omega, l = [float(row[1 + j * 6 + k]) for k in range(6)]
        sim.add(m=m, a=a, e=e, pomega=pomega, Omega=Omega, l=l)
    sim.move_to_com()
    return sim


def metrics(scores, labels):
    """AUC and FPR at 0.5. Small arrays, so the O(n^2) AUC is fine."""
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("metrics() needs both classes; your split is broken")
    auc = float((((pos[:, None] > neg[None, :]).sum()
                  + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg))))
    pred = scores > 0.5
    fp = ((pred == 1) & (labels < 0.5)).sum()
    tn = ((pred == 0) & (labels < 0.5)).sum()
    fpr = float(fp / max(fp + tn, 1))
    return auc, fpr


def bootstrap_metrics(scores, labels, n_boot=1000, seed=0):
    """Percentile CI for AUC and FPR. n is small and the split is fixed."""
    rng = np.random.default_rng(seed)
    aucs, fprs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(scores), len(scores))
        a, f = metrics(scores[idx], labels[idx])
        aucs.append(a)
        fprs.append(f)
    return ([float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            [float(np.percentile(fprs, 2.5)), float(np.percentile(fprs, 97.5))])


class NAM(nn.Module):
    """Local copy so the benchmark does not depend on src/ being importable."""

    def __init__(self, d, h=32):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(1, h), nn.Tanh(),
                          nn.Linear(h, h), nn.Tanh(),
                          nn.Linear(h, 1))
            for _ in range(d)
        ])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.bias + sum([n(x[:, i:i + 1]) for i, n in enumerate(self.nets)])


def train_glass(X, y, name):
    """Train a NAM on 70% of X and report train and holdout.

    Normalization uses the train split only. Doing it on all rows is the
    classic quiet leak, and it makes the holdout look better than it is.
    """
    perm = np.random.default_rng(0).permutation(len(y))
    cut = int(len(y) * 0.7)
    tr, ho = perm[:cut], perm[cut:]

    torch.manual_seed(0)
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-8
    Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    Y = torch.from_numpy((y > 0.5).astype(np.float32)).unsqueeze(1)

    model = NAM(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for epoch in range(500):
        batch = torch.from_numpy(
            np.random.default_rng(epoch).permutation(tr)[:min(512, len(tr))])
        opt.zero_grad()
        loss = nn.BCEWithLogitsLoss()(model(Xn[batch]), Y[batch])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xn)).numpy().ravel()

    a_tr, f_tr = metrics(p[tr], y[tr])
    a_ho, f_ho = metrics(p[ho], y[ho])
    a_ci, f_ci = bootstrap_metrics(p[ho], y[ho], seed=0)
    print(f"GLASS {name}: train AUC {a_tr:.4f} FPR {f_tr:.4f} | "
          f"holdout AUC {a_ho:.4f} [{a_ci[0]:.4f}, {a_ci[1]:.4f}] "
          f"FPR {f_ho:.4f} [{f_ci[0]:.4f}, {f_ci[1]:.4f}] (n={len(ho)})",
          flush=True)
    return a_ho, f_ho, a_ci, f_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-res", type=int, default=400)
    ap.add_argument("--n-rand", type=int, default=200)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.n_res, args.n_rand = 100, 50

    feats, trees, base_m = load_teacher()
    old = [k for k in feats if k != "Tsec"]

    rng = np.random.default_rng(args.seed)
    resonant = np.load(DATA / "resonant_real.npz")["X"]
    random_ = np.load(DATA / "random_real.npz")["X"]
    ir = rng.choice(len(resonant), args.n_res, replace=False)
    iq = rng.choice(len(random_), args.n_rand, replace=False)
    rows = list(resonant[ir]) + list(random_[iq])
    y_all = np.array([r[LABEL_COL] for r in rows])
    print(f"table sample {len(rows)} systems, truth-stable {y_all.mean():.3f}", flush=True)

    # Our fixed-1e4 features. Keep both layouts: 11 columns for the +Tsec
    # rows, 10 for the old-feature rows.
    t0 = time.time()
    sims = [rebuild(r) for r in rows]
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(feats_fixed, sims))
    print(f"1e4-fixed features: {time.time() - t0:.0f}s", flush=True)

    F10 = np.array(
        [[float(f.get(k, 0.0)) if ok else 0.0 for k in feats] for f, ok in res],
        np.float32)
    old_cols = [feats.index(k) for k in old]
    F10_old = F10[:, old_cols]
    ok10 = np.array([ok for _, ok in res])

    # Their exact Tsec features.
    t0 = time.time()
    out = features.generate_features(sims, n_jobs=8, singleWrap=True)
    print(f"Tsec features: {time.time() - t0:.0f}s", flush=True)

    F11, stable = [], []
    for trio_feats, st in out:
        f = trio_feats[0]
        F11.append([float(f[k]) if np.isfinite(float(f[k])) else 0.0 for k in feats])
        stable.append(bool(st))
    F11 = np.array(F11, np.float32)
    stable = np.array(stable)

    # Sec. 2: drop systems that went unstable during the short integration.
    both_ok = ok10 & stable
    idx = np.where(both_ok)[0]
    y = y_all[idx]
    print(f"short-stable in both configs: {both_ok.sum()}/{len(rows)}", flush=True)

    # The teacher JSON we have is SPOCK 2.x, which always uses Tsec. So the
    # teacher row below is a Tsec row. Do not call it a SPOCK-1 rerun. The
    # paper values are the reference for the other three cells.
    Yt = np.array([
        teacher_logit({k: F11[i, j] for j, k in enumerate(feats)},
                      feats, trees, base_m)
        for i in idx
    ], np.float32)
    pt = 1 / (1 + np.exp(-Yt))
    a, f = metrics(pt, y)
    a_ci, f_ci = bootstrap_metrics(pt, y, seed=args.seed)
    print(f"TEACHER (Tsec features) AUC {a:.4f} [{a_ci[0]:.4f}, {a_ci[1]:.4f}] "
          f"FPR {f:.4f} [{f_ci[0]:.4f}, {f_ci[1]:.4f}]  "
          f"[paper Tsec+Tsec: 0.9502/0.1431]", flush=True)

    X10 = F10[idx]
    X10_old = F10_old[idx]
    X11 = F11[idx]
    print("--- glass rows (same sample) ---", flush=True)
    train_glass(X10_old, y, "1e4-old (10 feats, approximate MMR)")
    train_glass(X11[:, :10], y, "Tsec-old (10 feats, exact MMR)")
    train_glass(X10, y, "1e4+Tsec (11 feats, approximate MMR)")
    train_glass(X11, y, "Tsec+Tsec (11 feats, exact MMR)")
    print("paper Table 1: 1e4-old 0.9430/0.1605 | Tsec-old 0.9417/0.1660 | "
          "1e4+Tsec 0.9442/0.1595 | Tsec+Tsec 0.9502/0.1431", flush=True)


if __name__ == "__main__":
    main()
