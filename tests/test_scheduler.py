from kirk import scheduler


def test_triage_gates():
    assert scheduler.triage(0.95) == "fast-stable"
    assert scheduler.triage(0.001) == "fast-unstable"
    assert scheduler.triage(0.42) == "zoom"


def test_reward_pays_for_a_correct_zoom():
    correct = scheduler.reward(True, True, cost=1.0, zoomed=True)
    wrong = scheduler.reward(False, True, cost=1.0, zoomed=True)
    assert correct > wrong
    assert correct == 1.0 - 0.1 + 0.2


def test_run_batch_only_zooms_call_truth():
    calls = []

    def decide(sim):
        return sim["pred"], sim["action"]

    def truth(sim):
        calls.append(sim["name"])
        return sim["truth"]

    items = [
        ("a", {"name": "a", "pred": True, "action": "fast-stable", "truth": True}),
        ("b", {"name": "b", "pred": True, "action": "zoom", "truth": False}),
    ]
    rows = scheduler.run_batch(items, decide, truth)
    assert calls == ["b"]
    assert rows[0]["truth"] is None
    assert rows[1]["truth"] is False
    assert rows[1]["pred"] is False
    assert rows[0]["reward"] is None
    assert rows[1]["reward"] is not None


def test_summarize_scores_skipped_systems_too():
    rows = [
        {"name": "a", "pred": True, "action": "fast-stable", "cost": 0.05, "truth": None},
        {"name": "b", "pred": False, "action": "zoom", "cost": 1.0, "truth": True},
    ]
    out = scheduler.summarize(rows, {"a": True, "b": True})
    assert out["acc"] == 0.5
    assert out["cost"] == 1.05
    assert out["zoom_frac"] == 0.5
