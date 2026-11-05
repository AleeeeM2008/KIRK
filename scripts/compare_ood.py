"""OOD 2.2k vs 12k on identical systems. Reuses saved truths, no long reruns."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import rebound

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kirk import glass, features
from kirk.short_features import short_features

DATA = Path(os.environ.get("KIRK_DATA", ROOT / "data"))
OLD = ROOT / "models" / "nam_spock_distilled.pt"
NEW = ROOT / "models" / "nam_spock_distilled_12k.pt"

def auc(scores, labels):
    pos = scores[labels > 0.5]; neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((((pos[:, None] > neg[None, :]).sum()
                   + 0.5 * (pos[:, None] == neg[None, :]).sum())
                  / (len(pos) * len(neg))))

def acc(scores, labels):
    return float(((np.asarray(scores) > 0.5) == (np.asarray(labels) > 0.5)).mean())

def rebuild_real(row):
    sim = rebound.Simulation(); sim.G = 1.0; sim.add(m=float(row[0]))
    for j in range(3):
        m, a, e, pomega, Omega, l = [float(row[1 + j*6 + k]) for k in range(6)]
        sim.add(m=m, a=a, e=e, pomega=pomega, Omega=Omega, l=l)
    sim.move_to_com(); return sim

def make_jupiter(K12, K23, seed):
    rng = np.random.default_rng(seed)
    sim = rebound.Simulation(); sim.G = 1.0; sim.add(m=1.0)
    m_p = 1e-3; a1 = 1.0
    rh1 = a1 * (m_p/3.0)**(1/3.); a2 = a1 + K12*rh1
    rh2 = a2 * (m_p/3.0)**(1/3.); a3 = a2 + K23*rh2
    for a in (a1, a2, a3):
        sim.add(m=m_p, a=a, e=rng.uniform(0,0.02),
                pomega=rng.uniform(0,2*np.pi), l=rng.uniform(0,2*np.pi))
    sim.move_to_com(); return sim

m_old, meta_old = glass.load_artifact(OLD)
m_new, meta_new = glass.load_artifact(NEW)
out = {"models": {"old": "nam_spock_distilled.pt", "new": "nam_spock_distilled_12k.pt"}}

# A. Real archives seed 8 (same as ablate)
rng = np.random.default_rng(8)
res = np.load(DATA/"resonant_real.npz")["X"]; rnd = np.load(DATA/"random_real.npz")["X"]
rows = list(res[rng.choice(len(res),400,replace=False)]) + list(rnd[rng.choice(len(rnd),200,replace=False)])
y = np.array([r[19] for r in rows], float)
sims = [rebuild_real(r) for r in rows]
gen = features.generate_features(sims, n_jobs=12, singleWrap=True)
X, st = [], []
for trios, s in gen:
    f = trios[0]
    X.append([float(f[k]) if np.isfinite(float(f[k])) else 0.0 for k in meta_old["feats"]])
    st.append(bool(s))
X = np.array(X, np.float32); st = np.array(st)
keep = st; Xk, yk = X[keep], y[keep]
po = glass.predict(m_old, meta_old, Xk); pn = glass.predict(m_new, meta_new, Xk)
out["real_seed8"] = {"n": int(len(yk)),
    "old": {"auc": auc(po, yk), "acc": acc(po, yk)},
    "new": {"auc": auc(pn, yk), "acc": acc(pn, yk)}}
print("real", out["real_seed8"], flush=True)

# B. Jupiter scheduler systems, short-200 screen, truth from saved runs
for seed in (10, 11, 12):
    d = json.load(open(f"/tmp/kirk_sched_{seed}.json"))
    truth = np.array([r["truth_all"] for r in d["rows"]], bool)
    rng = np.random.default_rng(seed)
    systems = []
    for i in range(150):
        K12 = rng.uniform(3.0,4.5); K23 = rng.uniform(3.0,4.5)
        systems.append(make_jupiter(K12, K23, 300+i))
    for i in range(150):
        K12 = rng.uniform(6.0,9.0); K23 = rng.uniform(6.0,9.0)
        systems.append(make_jupiter(K12, K23, 700+i))
    # note build_sample in benchmark uses n//2 tight then rest wide with one rng;
    # order above matches build_sample for n=300
    Po, Pn = [], []
    for i, sim in enumerate(systems):
        f, ok = short_features(sim, 200, 60, seed=seed+i)
        if not ok:
            Po.append(0.0); Pn.append(0.0); continue
        xo = np.array([[f[k] for k in meta_old["feats"]]], dtype=np.float32)
        xn = np.array([[f[k] for k in meta_new["feats"]]], dtype=np.float32)
        Po.append(float(glass.predict(m_old, meta_old, xo)[0]))
        Pn.append(float(glass.predict(m_new, meta_new, xn)[0]))
    Po = np.array(Po); Pn = np.array(Pn)
    out[f"jupiter_{seed}"] = {"n": len(truth),
        "old": {"auc": auc(Po, truth.astype(float)), "acc": acc(Po, truth.astype(float))},
        "new": {"auc": auc(Pn, truth.astype(float)), "acc": acc(Pn, truth.astype(float))}}
    print(seed, out[f"jupiter_{seed}"], flush=True)

Path(ROOT/"proofs"/"ood_12k.json").write_text(json.dumps(out, indent=1))
print("saved proofs/ood_12k.json", flush=True)
