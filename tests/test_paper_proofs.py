import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_scheduler_eval_has_no_speedup_claim():
    d = json.load(open(ROOT / "proofs" / "scheduler_eval.json"))
    assert d["n_total"] == 900
    assert abs(d["acc_total"] - 0.9944) < 1e-3
    assert "speedup_cost_mean" not in d
    assert "speedup_wall_mean" not in d
    for r in d["runs"]:
        assert "speedup_cost" not in r
        assert "speedup_wall" not in r
    assert d["by_kind"]["tight"]["n"] == 450
    assert d["by_kind"]["wide"]["n"] == 450
    assert d["by_kind"]["tight"]["zoom_frac"] < 0.1
    assert d["by_kind"]["wide"]["zoom_frac"] > 0.9


def test_multiplanet_n50():
    d = json.load(open(ROOT / "proofs" / "multiplanet.json"))
    for k in ("3", "4", "5", "6"):
        assert d["summary"][k]["n"] == 50
        assert 0.0 <= d["summary"][k]["auc"] <= 1.0


def test_ablate_has_loo_and_calibration():
    d = json.load(open(ROOT / "proofs" / "ablate.json"))
    assert "loo" in d and "Tsec" in d["loo"]
    assert len(d["calibration"]) == 5
    # top drops are Tsec / EMfracstdnear on this seed
    drops = {k: v["drop"] for k, v in d["loo"].items()}
    assert drops["Tsec"] > 0.01


def test_real1e9_is_jupiter_gates_negative_control():
    d = json.load(open(ROOT / "proofs" / "scheduler_real1e9.json"))
    # documents non-transfer; just check bounds are consistent
    assert d["zoom_frac"] > 0.3
    assert d["acc_lower_bound_zoom_wrong"] <= d["acc_upper_bound_zoom_correct"]
