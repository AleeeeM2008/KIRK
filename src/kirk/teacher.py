"""Run SPOCK's XGBoost classifier without installing xgboost.

The JSON export is just trees and a base score, so we walk it in plain Python.
This is what produced the distillation labels and the offline checks. It was
checked against the real predict_stable to four decimals; if you change it,
check it again.

N-body truth (REBOUND) is still the judge. This is only the teacher.
"""
import json
import math
import os

# First hit wins. KIRK_TEACHER_JSON is for local runs against a copy that is
# not in the repo (we do not redistribute the teacher). Read it at call time,
# not import time, or setting the variable in a script is useless.
def _candidates():
    return [p for p in [
        os.environ.get("KIRK_TEACHER_JSON"),
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "featureclassifier.json"),
        "/tmp/fc.json",
    ] if p]

_doc = None
FEATS = None
TREES = None
BASE_M = None


def _load():
    global _doc, FEATS, TREES, BASE_M
    if _doc is not None:
        return
    for p in _candidates():
        if os.path.exists(p):
            with open(p) as f:
                _doc = json.load(f)
            break
    if _doc is None:
        raise FileNotFoundError(
            "featureclassifier.json not found. Set KIRK_TEACHER_JSON or run "
            "scripts/download_assets.py.")

    learner = _doc["learner"]
    FEATS = learner["feature_names"]
    TREES = learner["gradient_booster"]["model"]["trees"]
    base = float(learner["learner_model_param"]["base_score"])
    # xgboost stores base_score as a probability; the trees add margins.
    BASE_M = math.log(base / (1 - base))


def logit(feat):
    """Teacher log-odds for one feature dict.

    Missing features become NaN. XGBoost sends NaN down the default branch,
    and so do we. Do not quietly replace them with zero here; that is a
    different model.
    """
    _load()
    x = [float(feat.get(k, float("nan"))) for k in FEATS]
    m = BASE_M
    for t in TREES:
        left = t["left_children"]
        right = t["right_children"]
        split_index = t["split_indices"]
        split_value = t["split_conditions"]
        weight = t["base_weights"]
        default_left = t["default_left"]

        i = 0
        while True:
            if left[i] == -1:
                m += weight[i]
                break
            v = x[split_index[i]]
            if v != v:  # NaN
                i = left[i] if default_left[i] == 1 else right[i]
            else:
                i = left[i] if v < split_value[i] else right[i]
    return m


def prob(feat):
    """Teacher probability of stability for one feature dict."""
    return 1.0 / (1.0 + math.exp(-logit(feat)))
