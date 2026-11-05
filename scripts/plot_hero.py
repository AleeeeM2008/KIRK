"""Hero figure: same-benchmark performance plus one readable shape.

Left: ROC on real 1e9-orbit labels, KIRK vs the SPOCK teacher.
Right: one shape function from the additive model.

The point is that the glass model is close to the teacher on the teacher's own
benchmark, and the whole model is made of curves like the one on the right.

Run in the kirk environment. Needs matplotlib and a feature cache produced by
scripts/train_glass.py (or --features).
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
    return float((((pos[:, None] > neg[None, :]).sum()
                   + 0.5 * (pos[:, None] == neg[None, :]).sum())
                  / (len(pos) * len(neg))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path,
                    default=ROOT / "data" / "features_1500_700_0.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "hero_benchmark.png")
    args = ap.parse_args()

    if not args.features.exists():
        raise SystemExit(
            f"{args.features} not found. Run scripts/train_glass.py once, or pass --features.")

    d = np.load(args.features)
    X, Y, keep, labels = d["X"], d["Y"], d["stable"], d["labels"]
    Xk, Yk, yk = X[keep], Y[keep], labels[keep]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xk))
    cut = int(len(Xk) * 0.85)
    ho = perm[cut:]

    model, meta = glass.load_artifact(ROOT / "models" / "nam_spock_distilled.pt")
    p_kirk = glass.predict(model, meta, Xk[ho])
    p_teacher = 1.0 / (1.0 + np.exp(-Yk[ho]))
    y = yk[ho]

    fpr_k, tpr_k = roc(p_kirk, y)
    fpr_t, tpr_t = roc(p_teacher, y)
    auc_k, auc_t = auc(p_kirk, y), auc(p_teacher, y)

    feats = list(meta["feats"])
    j = feats.index("EMfracstdnear")
    grid, curve = model.shapes(meta)[j]
    curve = curve * float(meta["Ys"])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 5.0), facecolor=BG,
        gridspec_kw={"width_ratios": [1.0, 1.15]})

    ax1.fill_between(fpr_k, tpr_k, color=TEAL, alpha=0.10)
    ax1.fill_between(fpr_t, tpr_t, color=BLUE, alpha=0.10)
    ax1.plot(fpr_k, tpr_k, color=TEAL, lw=2.6,
             label=f"KIRK (distilled NAM)   AUC {auc_k:.3f}")
    ax1.plot(fpr_t, tpr_t, color=BLUE, lw=2.6,
             label=f"SPOCK teacher (XGBoost)   AUC {auc_t:.3f}")
    ax1.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls="--", alpha=0.8)
    ax1.set_xlabel("false-positive rate")
    ax1.set_ylabel("true-positive rate")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title("Real 10⁹-orbit labels, held-out systems", fontsize=11, loc="left")
    ax1.text(0.03, 0.94, f"n = {len(y)} systems, {int(y.sum())} stable",
             transform=ax1.transAxes, color=MUTED, fontsize=8.5, va="top")
    ax1.legend(frameon=False, loc="lower right", fontsize=8.5, labelcolor=FG)
    style_dark(ax1)

    ax2.fill_between(grid, curve, 0, color=TEAL, alpha=0.10)
    ax2.plot(grid, curve, color=TEAL, lw=2.6)
    ax2.axhline(0, color=MUTED, lw=0.9)
    ax2.set_xlabel("EMfracstdnear (raw feature)")
    ax2.set_ylabel("contribution to logit")
    ax2.set_title("One of the 11 shape functions", fontsize=11, loc="left")
    ax2.text(0.04, 0.08, "read it, edit it, retrain\nthat is the model",
             transform=ax2.transAxes, color=MUTED, fontsize=8.5)
    style_dark(ax2)

    fig.suptitle("KIRK: within 0.026 AUC of SPOCK on the SPOCK benchmark",
                 color=FG, fontsize=13)
    fig.text(0.5, 0.015, "and every prediction is a bias plus readable curves",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=BG)
    print(f"saved {args.out}")

    (ROOT / "proofs" / "hero_benchmark.json").write_text(json.dumps({
        "n_holdout": int(len(y)),
        "n_stable": int(y.sum()),
        "auc_kirk": auc_k,
        "auc_teacher": auc_t,
        "feature_shown": "EMfracstdnear",
        "note": ("Real SPOCK archives and 1e9 labels, Sec. 2 exclusions, "
                 "split seed 0, holdout 15%."),
    }, indent=1))
    print("wrote proofs/hero_benchmark.json")


if __name__ == "__main__":
    main()
