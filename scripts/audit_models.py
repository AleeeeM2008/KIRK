"""Check every .pt in models/ and write a machine-readable index.

The folder is a lab drawer. Some files are classifiers, some are regression
heads, some were saved by code that no longer exists. This script says which
is which instead of making you load them one by one and guess.

Output: models/index.json. Exit code is 0 even when a file is unsupported;
unsupported is information, not a crash.
"""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

EXACT_FEATS = ['EMcrossnear', 'EMfracstdnear', 'EPstdnear', 'MMRstrengthnear',
               'EMcrossfar', 'EMfracstdfar', 'EPstdfar', 'MMRstrengthfar',
               'MEGNO', 'MEGNOstd', 'Tsec']


def hidden_layers(state):
    """Count the Linear layers in a NAM state dict, if it looks like one."""
    layers = []
    for key in state:
        if key.startswith("nets.0.") and key.endswith(".weight"):
            layers.append(int(key.split(".")[2]))
    return len(layers)


def classify(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    keys = set(d)
    state = d.get("w", {})
    feats = d.get("feats")
    prefixes = sorted({".".join(k.split(".")[:2]) for k in state})

    info = {
        "file": str(path.relative_to(MODELS)),
        "bytes": path.stat().st_size,
        "keys": sorted(keys),
        "features": feats,
        "state_prefixes": prefixes,
        "task": "unknown",
        "loads_as": None,
        "predict_ready": False,
        "api": None,
        "notes": "",
    }

    if "rules" in keys and "coef" in keys:
        info.update(task="rule overlay", notes="RuleFit overlay, not a standalone model.")
        return info
    if "d" in keys:
        info.update(task="collision pair head", loads_as="collision",
                    predict_ready=True, api="kirk.heads.predict_collision",
                    notes="Pair probabilities for (1,2), (2,3), (1,3).")
        return info
    if "rho" in keys:
        info.update(task="instability-time regression", loads_as="deepreg",
                    predict_ready=True, api="kirk.features.predict_instability_time",
                    notes="Point estimate in orbits, not a posterior.")
        return info
    if "slot_q" in state:
        info.update(task="slot attention / EBM variant",
                    notes="Custom head, no loader in kirk.glass.")
        return info

    if any(k.startswith("experts.") for k in state):
        info.update(task="mixture of experts",
                    notes="Router plus experts. The saved key layout differs from "
                          "the MoE class in kirk.glass, so it does not load as-is.")
        return info

    if any(k.startswith("nets.") for k in state):
        depth = hidden_layers(state)
        if depth == 3:
            info["loads_as"] = "nam"
        elif depth == 2:
            info["notes"] = "One hidden layer; kirk.glass.NAM expects two."
        else:
            info["notes"] = f"Unexpected NAM depth ({depth})."

        if feats is None:
            info["task"] = "outcome regression"
            info["loads_as"] = "outcome"
            info["predict_ready"] = True
            info["api"] = "kirk.heads.predict_outcome"
            info["notes"] = "Post-merger orbits for pair (1,2) only."
            return info
        if "Ym" in keys and "Ys" in keys:
            info["task"] = "classifier (logit head)"
        elif "platt_a" in keys:
            info["task"] = "classifier (raw logit + Platt)"
            info["notes"] = "Platt scaling is not applied by predict_logit."
        else:
            info["task"] = "raw logit head"
            info["notes"] = "No Ym/Ys, so predict_logit refuses it."

        exact_input = list(feats) == EXACT_FEATS
        if info["loads_as"] and "Ym" in keys and "Ys" in keys and exact_input:
            try:
                from kirk import glass
                model, meta = glass.load_artifact(path, arch="nam")
                glass.predict(model, meta, meta["Xm"][None, :])
                info["predict_ready"] = True
                info["api"] = "kirk.features.predict_stable"
            except Exception as exc:
                info["notes"] = f"load failed: {type(exc).__name__}: {exc}"
        elif not exact_input and feats is not None:
            info["notes"] = "Feature list is not the exact SPOCK feature set."
        return info

    info["notes"] = "No known model layout."
    return info


def main():
    rows = [classify(p) for p in sorted(MODELS.rglob("*.pt"))]
    (MODELS / "index.json").write_text(json.dumps(rows, indent=1))

    print(f"{'file':28s} {'task':34s} ready")
    for r in rows:
        print(f"{r['file']:28s} {r['task']:34s} {'yes' if r['predict_ready'] else 'no'}")
    print(f"\nwrote {MODELS / 'index.json'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    main()
