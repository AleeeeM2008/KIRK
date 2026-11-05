"""SPOCK 1 figures 3 and 4, with KIRK added.

Fig. 3 analog: ROC on the resonant holdout.
Fig. 4 analog: random-archive holdout, Hill-spacing projection plus ROC.

The KIRK curve is the shipped distilled model. MEGNO and EMcross are raw
single-feature baselines, not the trained XGBoost baselines in SPOCK's paper.
Random holdout n is small (66), so do not over-read that panel.

Run in the kirk environment. Needs matplotlib and data/resonant_real.npz +
data/random_real.npz (scripts/prepare_data.py).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kirk import glass  # noqa: E402

BG = "#0b0f14"
PANEL = "#0f1620"
FG = "#e6edf3"
MUTED = "#8b949e"
GRID = "#263241"
TEAL = "#2ec4b6"
BLUE = "#4ea8de"
ORANGE = "#f4a261"
GRAY = "#adb5bd"
RED = "#e76f51"


def style_dark(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    for title in (ax.title, getattr(ax, "_left_title", None),
                  getattr(ax, "_right_title", None)):
        if title is not None:
            title.set_color(FG)
    ax.grid(color=GRID, alpha=0.55)


def roc(scores, labels):
    order = np.argsort(-scores)
    scores, labels = scores[order], labels[order]
    tpr = np.concatenate(([0.0], np.cumsum(labels) / labels.sum()))
    fpr = np.concatenate(([0.0], np.cumsum(1 - labels) / (1 - labels).sum()))
    return fpr, tpr


def auc(scores, labels):
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((((pos[:, None] > neg[None, :]).sum()
                   + 0.5 * (pos[:, None] == neg[None, :]).sum())
                  / (len(pos) * len(neg))))


def hill_spacings(rows):
    """Inner and outer pair Hill spacings for 3-planet rows."""
    out = []
    for row in rows:
        mstar = float(row[0])
        vals = []
        for j in range(2):
            m1, a1 = float(row[1 + j * 6]), float(row[2 + j * 6])
            m2, a2 = float(row[1 + (j + 1) * 6]), float(row[2 + (j + 1) * 6])
            rh = a1 * ((m1 + m2) / 3.0 / mstar) ** (1 / 3.)
            vals.append((a2 - a1) / rh)
        out.append(vals)
    return np.array(out)


def plot_roc(ax, curves):
    for label, scores, y, color in curves:
        fpr, tpr = roc(scores, y)
        ax.plot(fpr, tpr, color=color, lw=2.2,
                label=f"{label}  AUC {auc(scores, y):.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--", alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.legend(frameon=False, loc="lower right", fontsize=8, labelcolor=FG)
    style_dark(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path,
                    default=ROOT / "data" / "features_1500_700_0.npz")
    args = ap.parse_args()

    data = np.load(args.features)
    X, Y, stable, labels = data["X"], data["Y"], data["stable"], data["labels"]
    model, meta = glass.load_artifact(ROOT / "models" / "nam_spock_distilled.pt")
    feats = list(meta["feats"])

    # Rebuild the exact sample order used by scripts/train_glass.py.
    res = np.load(ROOT / "data" / "resonant_real.npz")["X"]
    rnd = np.load(ROOT / "data" / "random_real.npz")["X"]
    rng = np.random.default_rng(0)
    rows = np.concatenate([
        res[rng.choice(len(res), 1500, replace=False)],
        rnd[rng.choice(len(rnd), 700, replace=False)],
    ])
    hills = hill_spacings(rows)
    kind = np.array(["res"] * 1500 + ["rand"] * 700)
    keep = stable
    Xk, Yk, yk, kindk, hillk = X[keep], Y[keep], labels[keep], kind[keep], hills[keep]
    perm = np.random.default_rng(0).permutation(len(Xk))
    ho = perm[int(len(Xk) * 0.85):]

    p_kirk = glass.predict(model, meta, Xk)
    p_teach = 1.0 / (1.0 + np.exp(-Yk))
    megno = -Xk[:, feats.index("MEGNO")]
    emcross = np.minimum(Xk[:, feats.index("EMcrossnear")],
                         Xk[:, feats.index("EMcrossfar")])

    res_sel = ho[kindk[ho] == "res"]
    rand_sel = ho[kindk[ho] == "rand"]
    curves_res = [
        ("KIRK", p_kirk[res_sel], yk[res_sel], TEAL),
        ("SPOCK teacher", p_teach[res_sel], yk[res_sel], BLUE),
        ("MEGNO raw", megno[res_sel], yk[res_sel], ORANGE),
        ("EMcross proxy", emcross[res_sel], yk[res_sel], GRAY),
    ]
    curves_rand = [
        ("KIRK", p_kirk[rand_sel], yk[rand_sel], TEAL),
        ("SPOCK teacher", p_teach[rand_sel], yk[rand_sel], BLUE),
        ("MEGNO raw", megno[rand_sel], yk[rand_sel], ORANGE),
        ("EMcross proxy", emcross[rand_sel], yk[rand_sel], GRAY),
    ]

    # Fig. 3 analog
    fig, ax = plt.subplots(figsize=(6.4, 5.6), facecolor=BG)
    plot_roc(ax, curves_res)
    ax.set_title(f"Resonant holdout (n={len(res_sel)})", fontsize=11, loc="left")
    fig.suptitle("SPOCK Fig. 3 analogue, with KIRK", color=FG, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out3 = ROOT / "assets" / "spock_fig3_kirk.png"
    fig.savefig(out3, dpi=200, facecolor=BG)
    print("saved", out3)

    # Fig. 4 analog
    fig = plt.figure(figsize=(7.2, 7.4), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.38)
    ax0 = fig.add_subplot(gs[0])
    stable_mask = yk[rand_sel] > 0.5
    ax0.scatter(hillk[rand_sel][stable_mask, 0], hillk[rand_sel][stable_mask, 1],
                s=20, color=BLUE, alpha=0.8, label="stable")
    ax0.scatter(hillk[rand_sel][~stable_mask, 0], hillk[rand_sel][~stable_mask, 1],
                s=20, color=RED, alpha=0.8, label="unstable")
    ax0.set_xlabel("inner pair separation (mutual Hill radii)")
    ax0.set_ylabel("outer pair separation (mutual Hill radii)")
    ax0.set_title(f"Random archive holdout (n={len(rand_sel)})", fontsize=11, loc="left")
    ax0.legend(frameon=False, fontsize=8, labelcolor=FG)
    style_dark(ax0)
    ax1 = fig.add_subplot(gs[1])
    plot_roc(ax1, curves_rand)
    ax1.set_title("ROC on the same random holdout", fontsize=11, loc="left")
    fig.suptitle("SPOCK Fig. 4 analogue, with KIRK", color=FG, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out4 = ROOT / "assets" / "spock_fig4_kirk.png"
    fig.savefig(out4, dpi=200, facecolor=BG)
    print("saved", out4)

    def summary(curves):
        return {label: {"auc": auc(scores, y), "n": int(len(y)),
                        "stable": int(y.sum())}
                for label, scores, y, _ in curves}
    (ROOT / "proofs" / "spock_figs_kirk.json").write_text(json.dumps({
        "fig3_resonant_holdout": summary(curves_res),
        "fig4_random_holdout": summary(curves_rand),
        "note": ("KIRK = shipped distilled model. MEGNO and EMcross are raw "
                 "single-feature baselines, not trained XGBoost baselines. "
                 "Random holdout is small."),
    }, indent=1))
    print("wrote proofs/spock_figs_kirk.json")


if __name__ == "__main__":
    main()
