"""Plot a synthetic spread of TRAPPIST-1 orbits split by KIRK's triage lane.

This is NOT a posterior. TRAPPIST-1's real posterior is tight, so a faithful
"before/after stability" picture would show almost no change. Instead this
draws N systems around the published solution with a deliberately broader
spread (eccentricity, period jitter, mass scatter, random phases), scores each
with KIRK, and shows which draws the scheduler would trust, which it would
send to a real integration (zoom), and which it would reject.

The default is the inner trio b-c-d. The seven-planet system is available with
--planets 7, but the min-over-trios rule caps its probability and usually
leaves no trust-stable lane; that is a model limitation, not a plotting bug.

Left panel : every draw, colored by triage lane.
Right panel: only the draws in the trust-stable lane, colored by planet.

Radius is drawn on a log scale so the planets are visible.

Run it from the repo root in the kirk environment. Needs matplotlib.
    python scripts/plot_orbit_lanes.py --n 60
    python scripts/plot_orbit_lanes.py --dry-run     # layout check only
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ME = 3.003e-6
MSTAR = 0.0898
TRAPPIST = [(1.37, 1.51087), (1.31, 2.42182), (0.39, 4.04961), (0.69, 6.09962),
            (1.04, 9.20669), (1.32, 12.35294), (0.33, 18.767)]
RMIN = 0.008  # AU, inside the innermost orbit; sets the log-radius origin


def kepler_a(P_days):
    return ((P_days / 365.25) ** 2 * MSTAR) ** (1 / 3.)


def draw_system(rng, e_scale, p_jitter, m_jitter, planets):
    """One synthetic system: a list of per-planet orbital parameter dicts."""
    out = []
    for m, P in planets:
        P_i = P * (1 + p_jitter * rng.standard_normal())
        out.append({
            "P": P_i,
            "a": kepler_a(P_i),
            "m": m * ME * float(np.exp(m_jitter * rng.standard_normal())),
            "e": float(min(1e-3 + abs(e_scale * rng.standard_normal()), 0.3)),
            "pomega": float(rng.uniform(0, 2 * np.pi)),
            "l": float(rng.uniform(0, 2 * np.pi)),
        })
    return out


def to_sim(params):
    import rebound
    sim = rebound.Simulation()
    sim.G = 4 * np.pi ** 2
    sim.add(m=MSTAR)
    for q in params:
        sim.add(m=q["m"], a=q["a"], e=q["e"], pomega=q["pomega"], l=q["l"])
    sim.move_to_com()
    return sim


def lane_color(lane):
    s = str(lane).lower()
    if "unst" in s:
        return "#d1495b"
    if "stab" in s:
        return "#2a9d8f"
    return "#edae49"


def lane_rank(lane):
    s = str(lane).lower()
    return 0 if ("stab" in s and "unst" not in s) else (2 if "unst" in s else 1)


def orbit_xy(a, e, pomega, n=360):
    f = np.linspace(0, 2 * np.pi, n)
    r = a * (1 - e ** 2) / (1 + e * np.cos(f))
    R = 1.0 + np.log10(r / RMIN)
    th = f + pomega
    return R * np.cos(th), R * np.sin(th)


def draw_orbit(ax, q, color, alpha):
    x, y = orbit_xy(q["a"], q["e"], q["pomega"])
    ax.plot(x, y, color=color, lw=2.6, alpha=alpha * 0.18, solid_capstyle="round")
    ax.plot(x, y, color=color, lw=0.7, alpha=alpha)


def style_axis(ax, lim):
    ax.set_facecolor("#05070d")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.scatter([0], [0], s=70, color="#ffd60a", zorder=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="number of synthetic draws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--e-scale", type=float, default=0.04,
                    help="scale of the half-normal eccentricity draw")
    ap.add_argument("--p-jitter", type=float, default=5e-3,
                    help="relative period jitter")
    ap.add_argument("--m-jitter", type=float, default=0.15,
                    help="log-normal mass scatter")
    ap.add_argument("--planets", type=int, default=3, choices=[3, 7],
                    help="3 = inner trio b-c-d; 7 = full chain")
    ap.add_argument("--dry-run", action="store_true",
                    help="fake probabilities, layout check only; does not import kirk")
    ap.add_argument("--out", default=str(ROOT / "assets" / "orbit_lanes.png"))
    args = ap.parse_args()

    planets = TRAPPIST[:3] if args.planets == 3 else TRAPPIST
    label = "b–c–d" if args.planets == 3 else "b–h"
    rng = np.random.default_rng(args.seed)
    draws = [draw_system(rng, args.e_scale, args.p_jitter, args.m_jitter, planets)
             for _ in range(args.n)]

    if args.dry_run:
        ps = rng.beta(0.6, 0.6, size=args.n)
        triage = lambda p: "stable" if p > 0.70 else ("unstable" if p < 0.01 else "zoom")  # noqa: E731
    else:
        from kirk import features, scheduler
        ps = np.atleast_1d(np.asarray(
            features.predict_stable([to_sim(d) for d in draws]), dtype=float))
        if len(ps) != args.n:
            raise RuntimeError(f"predict_stable returned {len(ps)} values for {args.n} sims")
        triage = scheduler.triage

    lanes = [triage(float(p)) for p in ps]
    counts = Counter(lanes)
    stable_lanes = {l for l in counts if lane_rank(l) == 0}

    lim = 1.0 + np.log10(max(d[-1]["a"] * (1 + d[-1]["e"]) for d in draws) / RMIN) + 0.15
    planet_colors = plt.cm.cool(np.linspace(0.0, 1.0, len(planets)))

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 5.6), facecolor="#05070d")
    style_axis(axl, lim)
    style_axis(axr, lim)

    # stable first, unstable last, so the rare lanes are not hidden underneath
    for d, lane in sorted(zip(draws, lanes), key=lambda t: -lane_rank(t[1])):
        for q in d:
            draw_orbit(axl, q, lane_color(lane), 0.30 if lane_rank(lane) == 0 else 0.45)

    kept = [d for d, l in zip(draws, lanes) if l in stable_lanes]
    for d in kept:
        for q, c in zip(d, planet_colors):
            draw_orbit(axr, q, c, 0.5)

    legend_order = sorted(counts, key=lane_rank)
    y0 = -lim * 0.97
    for k, lane in enumerate(legend_order):
        axl.text(-lim * 0.95, y0 + k * lim * 0.075,
                 f"{lane}: {counts[lane]}/{args.n}", color=lane_color(lane),
                 fontsize=9, ha="left", va="bottom")
    axl.set_title(f"Synthetic TRAPPIST-1 {label} draws, colored by KIRK lane",
                  color="white", fontsize=11)
    for label, color in zip("bcdefgh"[:len(planets)], planet_colors):
        axr.plot([], [], color=color, lw=2, label=label)
    axr.legend(frameon=False, labelcolor="white", fontsize=8,
               loc="lower right")
    axr.set_title(f"Trust-stable lane only ({len(kept)}/{args.n})",
                  color="white", fontsize=11)

    note = (f"Synthetic spread, not a posterior: e~|N(0,{args.e_scale})|, "
            f"period jitter {args.p_jitter:g}, mass scatter {args.m_jitter:g}, "
            f"random phases. Seed {args.seed}. {label} only. Radius is log scale.")
    if args.dry_run:
        note = "DRY RUN, fake probabilities, not a result. " + note
    fig.text(0.5, 0.02, note, color="0.7", fontsize=7.5, ha="center")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.07, wspace=0.03)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print(f"saved {out}")

    if not args.dry_run:
        proofs = ROOT / "proofs"
        proofs.mkdir(exist_ok=True)
        record = {
            "n": args.n,
            "planets": args.planets,
            "labels": label,
            "seed": args.seed,
            "spread": {"e_scale": args.e_scale, "p_jitter": args.p_jitter,
                       "m_jitter": args.m_jitter},
            "lane_counts": {str(k): int(v) for k, v in counts.items()},
            "p": [float(p) for p in ps],
            "lanes": [str(l) for l in lanes],
            "note": ("Synthetic spread around the published TRAPPIST-1 solution, "
                     "not an observational posterior. The default is the inner trio; "
                     "7-planet predictions use min-over-trios and are provisional."),
        }
        (proofs / "orbit_lanes.json").write_text(json.dumps(record, indent=1))
        print("wrote proofs/orbit_lanes.json")


if __name__ == "__main__":
    main()
