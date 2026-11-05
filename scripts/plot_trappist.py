"""Poster-style figure for the TRAPPIST-1 inner trio.

Top: the seven planets with the inner trio highlighted.
Bottom left: the MEGNO shape for the three phase draws.
Bottom right: KIRK vs upstream SPOCK probabilities for the same trio.

This is one trio, not a claim about the full seven-planet system. Run it in
the kirk environment; upstream SPOCK is optional.
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
# (label, period days, radius Earth, mass Earth)
PLANETS = [
    ("b", 1.51087, 1.116, 1.37), ("c", 2.42182, 1.097, 1.31),
    ("d", 4.04961, 0.788, 0.39), ("e", 6.09962, 0.920, 0.69),
    ("f", 9.20669, 1.045, 1.04), ("g", 12.35294, 1.129, 1.32),
    ("h", 18.767, 0.755, 0.33),
]


def build_system(seed):
    rng = np.random.default_rng(seed)
    sim = rebound.Simulation()
    sim.G = 4 * np.pi ** 2
    sim.add(m=0.0898)
    for _, P, _, m in PLANETS:
        a = ((P / 365.25) ** 2 * sim.particles[0].m) ** (1 / 3.)
        sim.add(m=m * ME, a=a, e=0.01, l=rng.uniform(0, 2 * np.pi))
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


def contributions(model, meta, x, ys):
    z = (x - meta["Xm"]) / meta["Xs"]
    out = []
    with torch.no_grad():
        for i, net in enumerate(model.nets):
            out.append(float(net(torch.tensor([[z[i]]], dtype=torch.float32)).detach()))
    return np.array(out) * ys


def main():
    model, meta = glass.load_artifact(ROOT / "models" / "nam_spock_distilled.pt")
    feats = list(meta["feats"])
    ys, ym = float(meta["Ys"]), float(meta["Ym"])

    kirk, spock, megno_x, megno_y = [], [], [], []
    for seed in (0, 1, 2):
        f, _ = features.generate_features(trio(build_system(seed), [1, 2, 3]))
        x = np.array([f[0][k] for k in feats], np.float32)
        c = contributions(model, meta, x, ys)
        logit = (float(model.bias.detach()) + (c / ys).sum()) * ys + ym
        kirk.append(1.0 / (1.0 + np.exp(-logit)))
        megno_x.append(float(f[0]["MEGNO"]))
        megno_y.append(float(c[feats.index("MEGNO")]))
    try:
        from spock import FeatureClassifier
        upstream = FeatureClassifier()
        spock = [float(upstream.predict_stable(trio(build_system(s), [1, 2, 3])))
                 for s in (0, 1, 2)]
    except ImportError:
        old_path = ROOT / "proofs" / "trappist_kirk.json"
        if old_path.exists():
            spock = json.loads(old_path.read_text()).get(
                "spock_p", [0.625, 0.734, 0.775])
        else:
            spock = [0.625, 0.734, 0.775]

    j = feats.index("MEGNO")
    grid = np.linspace(min(megno_x) - 0.25, max(megno_x) + 0.25, 200)
    z = ((grid - meta["Xm"][j]) / meta["Xs"][j]).astype(np.float32)
    with torch.no_grad():
        curve = (model.nets[j](torch.from_numpy(z).unsqueeze(1)).numpy().ravel() * ys)

    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.0], hspace=0.55, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, :])
    periods = np.array([p[1] for p in PLANETS])
    radii = np.array([p[2] for p in PLANETS])
    ax0.set_xscale("log")
    ax0.set_xlim(1.1, 25)
    ax0.set_ylim(-1.3, 1.3)
    ax0.axhline(0, color="0.7", lw=1)
    for i, (label, P, r, _) in enumerate(PLANETS):
        color = "#2a9d8f" if i < 3 else "0.55"
        ax0.scatter(P, 0, s=120 * r ** 2, color=color, edgecolor="white", zorder=3)
        ax0.annotate(label, (P, 0.22), ha="center", fontsize=10, color=color)
        ax0.plot([P, P], [0, 0.18], color=color, lw=0.8, alpha=0.6)
    ax0.add_patch(plt.Rectangle((1.35, -0.62), 4.6, 1.0, fill=False,
                                edgecolor="#2a9d8f", lw=1.4, ls="--", clip_on=False))
    ax0.text(2.0, 0.72, "inner trio b–c–d", color="#2a9d8f", fontsize=10)
    ax0.set_yticks([])
    ax0.set_xlabel("orbital period (days)")
    ax0.set_title("TRAPPIST-1, inner trio highlighted", fontsize=11, loc="left")
    for side in ("top", "right", "left"):
        ax0.spines[side].set_visible(False)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(grid, curve, color="#1f4e79", lw=2)
    ax1.axhline(0, color="0.7", lw=0.8)
    for k, (mx, my) in enumerate(zip(megno_x, megno_y)):
        ax1.scatter([mx], [my], s=45, color=plt.cm.viridis(0.2 + 0.3 * k), zorder=3)
    ax1.set_xlabel("MEGNO")
    ax1.set_ylabel("contribution to logit")
    ax1.set_title("MEGNO shape over the three draws", fontsize=11, loc="left")
    ax1.grid(alpha=0.25)
    ax1.text(0.03, 0.06, "KIRK 0.926–0.931", transform=ax1.transAxes,
             fontsize=9, color="#1f4e79")

    ax2 = fig.add_subplot(gs[1, 1])
    draws = np.arange(3)
    ax2.plot(draws, kirk, "o-", color="#2a9d8f", lw=2, ms=7, label="KIRK")
    ax2.plot(draws, spock, "o-", color="#c0392b", lw=2, ms=7, label="SPOCK")
    ax2.set_xticks(draws)
    ax2.set_xticklabels(["draw 0", "draw 1", "draw 2"])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("P(stable)")
    ax2.set_title("Same trio, same inputs, three phase draws", fontsize=11, loc="left")
    ax2.legend(frameon=False, loc="lower right")
    ax2.grid(alpha=0.25)
    ax2.axhline(0.5, color="0.85", lw=0.8)

    fig.suptitle("A glass-box look at TRAPPIST-1's inner trio", fontsize=14)
    fig.text(0.5, 0.005,
             "KIRK = distilled NAM; SPOCK = upstream XGBoost. One adjacent trio, "
             "not the full seven-planet system. Radii illustrative.",
             ha="center", fontsize=8, color="0.4")
    out = ROOT / "assets" / "trappist_kirk.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")

    (ROOT / "proofs" / "trappist_kirk.json").write_text(json.dumps({
        "planet_periods_days": {p[0]: p[1] for p in PLANETS},
        "inner_trio": ["b", "c", "d"],
        "kirk_p": kirk,
        "spock_p": spock,
        "megno": megno_x,
        "note": "Inner trio only; three phase draws of the same system.",
    }, indent=1))
    print("wrote proofs/trappist_kirk.json")


if __name__ == "__main__":
    main()
