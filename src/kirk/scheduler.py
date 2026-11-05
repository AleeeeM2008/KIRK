"""Scheduler for N-body work.

The idea is simple and old: do not run the expensive integration unless you
have to. This release ships one short stage (features plus the glass model) and
the long run for the systems the gates cannot call. The analytic prefilter from
the lab is not wired in here and is not claimed as part of this package.

The gates are hand-edited, not learned. Read the misses, move a threshold,
rerun. No gradients and no tuning loop.
"""


def triage(prob, lo=0.01, hi=0.70):
    """Pick a lane from the glass probability.

    Above hi: trust stable. Below lo: trust unstable. In between: zoom, because
    that is exactly where a cheap model is not worth trusting.
    """
    if prob > hi:
        return "fast-stable"
    if prob < lo:
        return "fast-unstable"
    return "zoom"


def reward(pred, truth, cost, zoomed=False, detail_bonus=0.2, cost_w=0.1):
    """One number for the Hand-RL score.

    Correct call is +1, wrong is -1, then subtract the cost. A correct zoom
    also pays a detail bonus because a zoom buys more than a label: you get
    the encounter time and the pair. That bonus is the only reason the policy
    ever spends money.
    """
    r = (1.0 if pred == truth else -1.0) - cost_w * cost
    if zoomed and pred == truth:
        r += detail_bonus
    return r


def run_batch(items, decide, truth_fn, zoom_cost=1.0, fast_cost=0.05,
              detail_bonus=0.2, cost_w=0.1):
    """Run a policy over (name, sim) pairs.

    decide(sim) -> (prediction, action). Only zooms call truth_fn, so a
    non-zoom row has truth=None. Pass truth_all to summarize() if you want the
    full picture; that is an evaluation run, not something the policy sees.

    Returns one dict per system.
    """
    rows = []
    for name, sim in items:
        pred, action = decide(sim)
        zoomed = action == "zoom"
        cost = zoom_cost if zoomed else fast_cost
        truth = bool(truth_fn(sim)) if zoomed else None
        final = truth if zoomed else pred
        r = None if truth is None else reward(
            pred, truth, cost, zoomed, detail_bonus, cost_w)
        rows.append({
            "name": name,
            "pred": bool(final),
            "action": action,
            "cost": cost,
            "truth": truth,
            "reward": r,
        })
    return rows


def summarize(rows, truth_all):
    """Accuracy, cost and zoom fraction.

    truth_all maps name -> bool for every system, including the ones the
    policy skipped. Without it you can only score the zooms, and that number
    is useless.
    """
    acc = sum(1 for r in rows if r["pred"] == truth_all[r["name"]]) / max(len(rows), 1)
    cost = sum(r["cost"] for r in rows)
    zooms = sum(1 for r in rows if r["action"] == "zoom")
    return {"acc": acc, "cost": cost, "zoom_frac": zooms / max(len(rows), 1)}
