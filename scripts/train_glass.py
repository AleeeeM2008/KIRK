"""Train the glass NAM on SPOCK's teacher.

This is the script that was missing. It samples real initial conditions from
SPOCK's archives, runs their exact feature pipeline, labels the systems with
the teacher JSON, and fits an additive NAM. The output .pt contains the
weights, the normalization, the feature order and enough provenance to tell
where it came from.

Short run (a few minutes):
    ~/kirk-env/bin/python scripts/prepare_data.py
    ~/kirk-env/bin/python scripts/train_glass.py --quick

Full run:
    ~/kirk-env/bin/python scripts/train_glass.py

The feature cache in data/ is keyed by sample size and seed. Reruns with the
same arguments skip the expensive part.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rebound
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kirk import teacher  # noqa: E402
from kirk.features import FEATS  # noqa: E402  (imports spock)
from kirk.glass import NAM  # noqa: E402

LABEL_COL = 19


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def rebuild(row):
    """One archive row -> one REBOUND simulation, G=1 as in their data."""
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=float(row[0]))
    for j in range(3):
        m, a, e, pomega, Omega, l = [float(row[1 + j * 6 + k]) for k in range(6)]
        sim.add(m=m, a=a, e=e, pomega=pomega, Omega=Omega, l=l)
    sim.move_to_com()
    return sim


def load_sample(data_dir, n_res, n_rand, seed):
    """Pick rows from the real archives. No synthetic data here."""
    res_path = data_dir / "resonant_real.npz"
    rnd_path = data_dir / "random_real.npz"
    for p in (res_path, rnd_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Pass --data with the SPOCK archives.")

    resonant = np.load(res_path)["X"]
    random_ = np.load(rnd_path)["X"]
    rng = np.random.default_rng(seed)

    n_res = min(n_res, len(resonant))
    n_rand = min(n_rand, len(random_))
    rows = np.concatenate([
        resonant[rng.choice(len(resonant), n_res, replace=False)],
        random_[rng.choice(len(random_), n_rand, replace=False)],
    ])
    labels = rows[:, LABEL_COL].astype(np.float32)
    sims = [rebuild(r) for r in rows]
    return sims, labels, {"resonant": res_path.name, "random": rnd_path.name,
                          "n_resonant": n_res, "n_random": n_rand,
                          "sha256_resonant": sha256(res_path),
                          "sha256_random": sha256(rnd_path)}


def extract_features(sims, jobs, chunk=200):
    """Run SPOCK's exact feature pipeline in chunks so progress is visible."""
    from kirk import features

    X, stable = [], []
    t0 = time.time()
    for start in range(0, len(sims), chunk):
        out = features.generate_features(sims[start:start + chunk], n_jobs=jobs,
                                         singleWrap=True)
        for trios, st in out:
            f = trios[0]
            X.append([float(f[k]) if np.isfinite(float(f[k])) else 0.0 for k in FEATS])
            stable.append(bool(st))
        print(f"  features {len(X)}/{len(sims)}  {time.time() - t0:.0f}s", flush=True)
    return np.array(X, np.float32), np.array(stable)


def make_teacher_logits(X):
    return np.array([
        teacher.logit({k: float(X[i, j]) for j, k in enumerate(FEATS)})
        for i in range(len(X))
    ], np.float32)


def binary_auc(scores, labels):
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((((pos[:, None] > neg[None, :]).sum()
                   + 0.5 * (pos[:, None] == neg[None, :]).sum())
                  / (len(pos) * len(neg))))


def fidelity(z, y):
    """Compare student logits z with teacher logits y."""
    p = 1.0 / (1.0 + np.exp(-z))
    pt = 1.0 / (1.0 + np.exp(-y))
    agree = float(((p > 0.5) == (pt > 0.5)).mean())
    rmse = float(np.sqrt(np.mean((z - y) ** 2)))
    return agree, rmse, binary_auc(p, (pt > 0.5).astype(np.float32))


def bootstrap_ci(scores, labels, fn, n_boot=2000, seed=0):
    """Percentile bootstrap over systems. n is small; do not over-read the tails."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(scores), len(scores))
        vals.append(fn(scores[idx], labels[idx]))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def train_nam(X, Y, seed, epochs, holdout, lr=5e-3):
    """Fit NAM on teacher logits. Normalization comes from the train split only."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    perm = rng.permutation(len(X))
    cut = int(len(X) * (1.0 - holdout))
    tr, ho = perm[:cut], perm[cut:]

    xm = X[tr].mean(0)
    xs_raw = X[tr].std(0)
    constant = xs_raw < 1e-6
    xs = np.where(constant, 1.0, xs_raw)

    ym = float(Y[tr].mean())
    ys = float(Y[tr].std()) + 1e-8

    Xt = torch.from_numpy(((X - xm) / xs).astype(np.float32))
    Xt[:, constant] = 0.0
    Yt = torch.from_numpy(((Y - ym) / ys).astype(np.float32)).unsqueeze(1)

    model = NAM(len(FEATS))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    tr_t = torch.from_numpy(tr)

    for epoch in range(epochs):
        batch = tr_t[torch.from_numpy(
            np.random.default_rng(seed + epoch).permutation(len(tr))[:512])]
        opt.zero_grad()
        loss = loss_fn(model(Xt[batch]), Yt[batch])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xt).numpy().ravel() * ys + ym

    out = {"n_train": len(tr), "n_holdout": len(ho), "constant_features": int(constant.sum())}
    out["agree_train"], out["rmse_train"], out["auc_teacher_train"] = fidelity(pred[tr], Y[tr])
    out["agree_holdout"], out["rmse_holdout"], out["auc_teacher_holdout"] = fidelity(pred[ho], Y[ho])

    meta = {"xm": xm, "xs_raw": xs_raw, "ym": ym, "ys": ys}
    return model, meta, pred, tr, ho, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--teacher-json", type=Path)
    ap.add_argument("--n-resonant", type=int, default=1500)
    ap.add_argument("--n-random", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--seed-sweep", type=int, default=5,
                    help="train this many seeds and report the spread")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--stats", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--quick", action="store_true", help="tiny sample, for smoke tests")
    args = ap.parse_args()

    if args.teacher_json:
        os.environ["KIRK_TEACHER_JSON"] = str(args.teacher_json)
    if args.quick:
        args.n_resonant, args.n_random, args.epochs = 120, 60, 300

    if args.out is None:
        args.out = (Path("/tmp/kirk_quick.pt") if args.quick
                    else ROOT / "models" / "nam_spock_distilled.pt")
    if args.stats is None:
        args.stats = (Path("/tmp/kirk_quick_stats.json") if args.quick
                      else ROOT / "models" / "nam_spock_stats.json")
    if args.report is None:
        args.report = (Path("/tmp/kirk_quick_distill.json") if args.quick
                       else ROOT / "proofs" / "distill.json")

    cache = args.cache or (args.data / f"features_{args.n_resonant}_{args.n_random}_{args.seed}.npz")

    if cache.exists() and not args.no_cache:
        d = np.load(cache)
        X, stable, labels, Y = d["X"], d["stable"], d["labels"], d["Y"]
        data_info = json.loads(str(d["data_info"]))
        print(f"loaded cache {cache} ({len(X)} systems)", flush=True)
    else:
        print(f"sampling {args.n_resonant} resonant + {args.n_random} random", flush=True)
        sims, labels, data_info = load_sample(
            args.data, args.n_resonant, args.n_random, args.seed)
        X, stable = extract_features(sims, args.jobs)
        Y = make_teacher_logits(X)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, stable=stable, labels=labels, Y=Y,
                            feats=np.array(FEATS), data_info=json.dumps(data_info))
        print(f"wrote cache {cache}", flush=True)

    # Paper Sec. 2: short-unstable systems are not classifier training data.
    keep = stable
    Xk, Yk, labels_k = X[keep], Y[keep], labels[keep]
    print(f"short-stable {keep.sum()}/{len(keep)}", flush=True)

    model, meta, pred, tr, ho, metrics = train_nam(
        Xk, Yk, args.seed, args.epochs, args.holdout)

    p = 1.0 / (1.0 + np.exp(-pred))
    metrics["auc_truth_holdout"] = binary_auc(p[ho], labels_k[ho])
    metrics["acc_truth_holdout"] = float(
        ((p[ho] > 0.5) == (labels_k[ho] > 0.5)).mean())
    metrics["n_systems"] = int(len(Xk))
    metrics["acc_truth_holdout_ci95"] = bootstrap_ci(
        p[ho], labels_k[ho],
        lambda s, y: float(((s > 0.5) == (y > 0.5)).mean()), seed=args.seed)
    metrics["auc_truth_holdout_ci95"] = bootstrap_ci(
        p[ho], labels_k[ho], binary_auc, seed=args.seed)
    teacher_label = (1.0 / (1.0 + np.exp(-Yk[ho])) > 0.5).astype(np.float32)
    metrics["auc_teacher_holdout_ci95"] = bootstrap_ci(
        pred[ho], teacher_label, binary_auc, seed=args.seed)
    def agreement(student_logits, labels):
        return float((((1 / (1 + np.exp(-student_logits))) > 0.5)
                      == (labels > 0.5)).mean())

    metrics["agree_holdout_ci95"] = bootstrap_ci(
        pred[ho], teacher_label, agreement, seed=args.seed)

    sweep = []
    for seed_i in range(args.seed, args.seed + max(args.seed_sweep, 1)):
        if seed_i == args.seed:
            m_i = metrics
        else:
            _, _, pred_i, _, ho_i, m_i = train_nam(
                Xk, Yk, seed_i, args.epochs, args.holdout)
            p_i = 1.0 / (1.0 + np.exp(-pred_i))
            m_i = dict(m_i)
            m_i["auc_truth_holdout"] = binary_auc(p_i[ho_i], labels_k[ho_i])
            m_i["acc_truth_holdout"] = float(
                ((p_i[ho_i] > 0.5) == (labels_k[ho_i] > 0.5)).mean())
        sweep.append({
            "seed": seed_i,
            "agree_holdout": m_i["agree_holdout"],
            "auc_teacher_holdout": m_i["auc_teacher_holdout"],
            "auc_truth_holdout": m_i["auc_truth_holdout"],
            "acc_truth_holdout": m_i["acc_truth_holdout"],
        })
    metrics["seed_sweep"] = sweep
    for key in ("agree_holdout", "auc_teacher_holdout", "auc_truth_holdout",
                "acc_truth_holdout"):
        vals = np.array([row[key] for row in sweep], dtype=float)
        metrics[key + "_seed_mean"] = float(vals.mean())
        metrics[key + "_seed_std"] = float(vals.std())

    provenance = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "seed": args.seed,
        "epochs": args.epochs,
        "holdout": args.holdout,
        "data": data_info,
        "teacher_json": (args.teacher_json.name if args.teacher_json
                         else "models/featureclassifier.json"),
        "teacher_sha256": sha256(args.teacher_json) if args.teacher_json else None,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }

    artifact = {
        "w": model.state_dict(),
        "Xm": meta["xm"],
        "Xs": meta["xs_raw"],  # zero means constant in training; _normalize handles it
        "Ym": meta["ym"],
        "Ys": meta["ys"],
        "feats": FEATS,
        "metrics": metrics,
        "provenance": provenance,
        "agree": metrics["agree_train"],
        "rmse": metrics["rmse_train"],
        "auc": metrics["auc_teacher_train"],
        "n": metrics["n_systems"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.out)

    try:
        artifact_label = str(args.out.relative_to(ROOT))
    except ValueError:
        artifact_label = str(args.out)
    report = {"artifact": artifact_label, "metrics": metrics,
              "provenance": provenance}
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(report, indent=1))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1))

    print("holdout: agree %.4f  rmse %.4f  auc(teacher) %.4f  auc(truth) %.4f"
          % (metrics["agree_holdout"], metrics["rmse_holdout"],
             metrics["auc_teacher_holdout"], metrics["auc_truth_holdout"]))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
