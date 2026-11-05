"""Make the glass-box figure used in the README.

The example is TRAPPIST-1 seed 0. The inner trio [1,2,3] and outer trio
[3,4,5] get different KIRK probabilities. Most of that difference is one
shape: EMfracstdnear. This script plots that curve and marks both systems,
then shows the per-feature contributions side by side.

Run it from the repo root in the kirk environment. Needs matplotlib.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rebound
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from kirk import features, glass  # noqa: E402

ME = 3.003e-6
TRAPPIST = [(1.37, 1.51087), (1.31, 2.42182), (0.39, 4.04961), (0.69, 6.09962),
            (1.04, 9.20669), (1.32, 12.35294), (0.33, 18.767)]
TARGET = "EMfracstdnear"


def build_trappist(seed=0):
    rng = np.random.default_rng(seed)
    sim = rebound.Simulation()
    sim.G = 4 * np.pi ** 2
    sim.add(m=0.0898)
    for m, P in TRAPPIST:
        a = ((P / 365.25) ** 2 * sim.particles[0].m) ** (1 / 3.)
        sim.add(m=m * ME, a=a, e=0.01, pomega=0.0,
                l=rng.uniform(0, 2 * np.pi))
    sim.move_to_com()
    return sim


def trio(sim, indices):
    sub = rebound.Simulation()
    sub.G = sim.G
    sub.add(m=sim.particles[0].m)
    for i in indices:
        p = sim.particles[i]
        sub.add(m=p.m, a=p.a, e=p.e, pomega=p.pomega, l=p.l)
    sub.move_to_com()
    return sub


def contributions(model, meta, x):
    z = (x - meta["Xm"]) / meta["Xs"]
    out = []
    with torch.no_grad():
        for i, net in enumerate(model.nets):
            out.append(float(net(torch.tensor([[z[i]]], dtype=torch.float32))))
    return np.array(out)


def main():
    model, meta = glass.load_artifact(ROOT / "models" / "nam_spock_distilled.pt")
    feats = list(meta["feats"])
    sim = build_trappist(0)

    cases = [("inner trio [1,2,3]", [1, 2, 3]),
             ("outer trio [3,4,5]", [3, 4, 5])]
    ys, ym = float(meta["Ys"]), float(meta["Ym"])
    values, contribs, probs, logits, norm_logits = [], [], [], [], []
    for _, indices in cases:
        f, _ = features.generate_features(trio(sim, indices))
        x = np.array([f[0][k] for k in feats], np.float32)
        c_norm = contributions(model, meta, x)
        c = c_norm * ys
        norm_logit = float(model.bias.detach()) + c_norm.sum()
        logit = norm_logit * ys + ym
        p = 1.0 / (1.0 + np.exp(-logit))
        values.append(x)
        contribs.append(c)
        probs.append(p)
        logits.append(float(logit))
        norm_logits.append(float(norm_logit))

    j = feats.index(TARGET)
    grid, curve = model.shapes(meta)[j]
    curve = curve * ys

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.1, 1.4]})
    ax1.plot(grid, curve, color="#1f4e79", lw=2)
    ax1.axhline(0, color="0.7", lw=0.8)
    colors = ["#2a9d8f", "#c0392b"]
    for k, ((label, _), x, p, col) in enumerate(zip(cases, values, probs, colors)):
        ax1.scatter([x[j]], [contribs[k][j]], s=55, color=col, zorder=3)
        ax1.annotate(f"{label}\n{x[j]:.4f}, P={p:.2f}",
                     (x[j], contribs[k][j]),
                     textcoords="offset points", xytext=(10, 10),
                     fontsize=8, color=col)
    ax1.set_xlabel(TARGET)
    ax1.set_ylabel("contribution to logit")
    ax1.set_title("One shape function, two real trios")
    ax1.grid(alpha=0.25)

    top = np.argsort(np.abs(contribs[1] - contribs[0]))[::-1][:7]
    y = np.arange(len(top))
    ax2.barh(y - 0.18, contribs[0][top], height=0.35,
             color=colors[0], label="inner trio")
    ax2.barh(y + 0.18, contribs[1][top], height=0.35,
             color=colors[1], label="outer trio")
    ax2.axvline(0, color="0.7", lw=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels([feats[i] for i in top], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("contribution to logit")
    ax2.set_title("Where the difference comes from")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle("KIRK is additive: a prediction is a bias plus these curves",
                 fontsize=11)
    fig.tight_layout()
    out = ROOT / "assets" / "shape_case.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved {out}")

    gap = logits[0] - logits[1]
    target_gap = contribs[0][j] - contribs[1][j]
    share = abs(target_gap) / abs(gap)

    comparison = {}
    try:
        from spock import FeatureClassifier
        upstream = FeatureClassifier()
        for seed in (0, 1, 2):
            seed_sim = build_trappist(seed)
            row = {}
            for name, indices in (("inner", [1, 2, 3]),
                                  ("outer", [3, 4, 5])):
                sub = trio(seed_sim, indices)
                row[name] = {"kirk": float(features.predict_stable(sub)),
                             "spock": float(upstream.predict_stable(sub))}
            comparison[str(seed)] = row
    except ImportError:
        old_path = ROOT / "proofs" / "shape_case.json"
        if old_path.exists():
            comparison = json.loads(old_path.read_text()).get("spock_comparison", {})

    record = {
        "target_feature": TARGET,
        "Ys": ys,
        "Ym": ym,
        "logit_gap": float(gap),
        "target_contribution_gap": float(target_gap),
        "target_share_of_gap": float(share),
        "cases": [
            {"label": label, "trio": indices, "p": p, "logit": logit,
             "normalized_logit": norm_logit,
             "features": {k: float(v) for k, v in zip(feats, x)},
             "contributions": {k: float(v) for k, v in zip(feats, c)},
             "contributions_normalized": {
                 k: float(v) for k, v in zip(feats, c / ys)}}
            for (label, indices), x, c, p, logit, norm_logit in zip(
                cases, values, contribs, probs, logits, norm_logits)
        ],
        "spock_comparison": comparison,
        "note": ("TRAPPIST-1 seed 0. Logit units are the model's output before "
                 "the sigmoid; contributions are already multiplied by Ys. "
                 f"The target feature accounts for {share:.1%} of the "
                 f"{gap:.2f} logit gap; other features make up the rest."),
    }
    (ROOT / "proofs" / "shape_case.json").write_text(json.dumps(record, indent=1))
    print("wrote proofs/shape_case.json")


if __name__ == "__main__":
    main()
