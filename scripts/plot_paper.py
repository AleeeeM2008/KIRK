"""Paper figures from proofs/*.json. No N-body here."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    p = OUT / name
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    print(f"saved {p}")

# 1. LOO ablation
d = json.load(open(ROOT / "proofs" / "ablate.json"))
items = sorted(d["loo"].items(), key=lambda kv: -kv[1]["drop"])
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar([k for k, _ in items], [v["drop"] for _, v in items])
ax.set_xlabel("feature", fontsize=12)
ax.set_ylabel("AUC drop", fontsize=12)
ax.set_title(f"Leave-one-out ablation (baseline AUC {d['baseline_auc']:.3f}, n={d['n']})", fontsize=12)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=11)
plt.xticks(rotation=30, ha="right")
save(fig, "fig_ablation.png")
plt.close(fig)
plt.rcParams.update({"font.size": 10})

# 2. Calibration
fig, ax = plt.subplots(figsize=(4, 4))
for b in d["calibration"]:
    if b["n"]:
        ax.scatter([b["mean_p"]], [b["emp"]], s=[max(20, b["n"] / 2)])
        ax.text(b["mean_p"], b["emp"], f" n={b['n']}", fontsize=8)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("mean predicted p")
ax.set_ylabel("empirical stable rate")
ax.set_title("Calibration (Tsec features, 1e9 labels)")
ax.set_aspect("equal")
save(fig, "fig_calibration.png")
plt.close(fig)

# 3. Scheduler by-kind
s = json.load(open(ROOT / "proofs" / "scheduler_eval.json"))
fig, ax = plt.subplots(figsize=(5, 3.5))
kinds = ["tight", "wide"]
acc = [s["by_kind"][k]["acc"] for k in kinds]
zoom = [s["by_kind"][k]["zoom_frac"] for k in kinds]
x = range(len(kinds))
ax.bar([i - 0.2 for i in x], acc, width=0.4, label="acc")
ax.bar([i + 0.2 for i in x], zoom, width=0.4, label="zoom frac")
ax.set_xticks(list(x), kinds)
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title(f"Scheduler 900 systems, acc {s['acc_total']:.3f}")
save(fig, "fig_scheduler.png")
plt.close(fig)

# 4. Multiplanet
m = json.load(open(ROOT / "proofs" / "multiplanet.json"))
fig, ax = plt.subplots(figsize=(5, 3.5))
ns = sorted(m["summary"], key=int)
ka = [m["summary"][k]["acc"] for k in ns]
ha = [m["summary"][k]["hill_acc"] for k in ns]
x = range(len(ns))
ax.plot(list(x), ka, "o-", label="KIRK acc")
ax.plot(list(x), ha, "s--", label="Hill 6.75")
ax.set_xticks(list(x), [f"{k}" for k in ns])
ax.set_xlabel("planets")
ax.set_ylabel("acc at 1e5 truth")
ax.legend()
ax.set_title("Min-over-trios (50/count)")
save(fig, "fig_multiplanet.png")
plt.close(fig)

# 5. Distill old vs 12k
old = json.load(open(ROOT / "proofs" / "distill.json"))
new = json.load(open(ROOT / "proofs" / "distill_12k.json"))
fig, ax = plt.subplots(figsize=(5, 3.5))
labels = ["auc_truth", "acc_truth", "agree"]
ov = [old["metrics"]["auc_truth_holdout"], old["metrics"]["acc_truth_holdout"],
      old["metrics"]["agree_holdout"]]
nv = [new["metrics"]["auc_truth_holdout"], new["metrics"]["acc_truth_holdout"],
      new["metrics"]["agree_holdout"]]
x = range(len(labels))
ax.bar([i - 0.2 for i in x], ov, width=0.4, label="2.2k seed0")
ax.bar([i + 0.2 for i in x], nv, width=0.4, label="12k seed0")
ax.set_xticks(list(x), labels)
ax.set_ylim(0.7, 1.0)
ax.legend()
ax.set_title("Data scaling: 5x data, no gain (seed0)")
save(fig, "fig_scaling.png")
plt.close(fig)
print("done")
