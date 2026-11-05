import numpy as np
import pytest
import rebound

from kirk import features


def three_planet_sim(periods=(1.0, 1.5, 2.25), masses=(1e-5, 1e-5, 1e-5)):
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=1.0)
    for P, m in zip(periods, masses):
        sim.add(m=m, P=P, e=0.01, pomega=0.3, l=0.7)
    sim.move_to_com()
    return sim


def fake_trios():
    base = {k: 1.0 for k in features.FEATS}
    return [dict(base), dict(base)]


def test_get_pairs_labels_near_and_far():
    sim = three_planet_sim()
    pairs = features.get_pairs(sim, [1, 2, 3])
    assert [p[0] for p in pairs] == ["near", "far"]
    assert pairs[0][1] < pairs[0][2]
    assert pairs[1][1] < pairs[1][2]


def test_farey_sequence_is_ordered():
    seq = features.farey_sequence(4)
    assert seq[0] == (0, 1)
    assert seq[-1] == (1, 1)


@pytest.mark.skipif(not hasattr(rebound.Simulation(), "ri_whfast"),
                    reason="feature extraction needs REBOUND 4.x")
def test_three_planet_features_run():
    X, stable = features.feature_matrix(three_planet_sim())
    assert X.shape == (1, len(features.FEATS))
    assert stable.shape == (1,)


def test_few_planet_predictions():
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=1.0)
    assert features.predict_stable(sim) == 1.0

    sim.add(m=1e-5, a=1.0, e=0.01)
    sim.move_to_com()
    assert features.predict_stable(sim) == 1.0

    sim.add(m=1e-5, a=3.0, e=0.01)
    sim.move_to_com()
    got = features.predict_stable(sim)
    expected = 1.0 if features.hillfac(sim) > 1 else 0.0
    assert got == expected


def test_few_planet_instability_times():
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=1.0)
    assert features.predict_instability_time(sim) == float("inf")

    sim.add(m=1e-5, a=1.0, e=0.01)
    sim.move_to_com()
    assert features.predict_instability_time(sim) == float("inf")

    sim.add(m=1e-5, a=3.0, e=0.01)
    sim.move_to_com()
    got = features.predict_instability_time(sim)
    expected = float("inf") if features.hillfac(sim) > 1 else 0.0
    assert got == expected


def test_feature_rows_keeps_every_trio(monkeypatch):
    def fake(sim, nbody_tmax=1e6):
        return fake_trios(), True

    monkeypatch.setattr(features, "_generate_features", fake)
    X, stable, owner = features.feature_rows([object(), object()])
    assert X.shape == (4, len(features.FEATS))
    assert owner.tolist() == [0, 0, 1, 1]
    assert stable.tolist() == [True, True]


def test_feature_matrix_wraps_single_sim(monkeypatch):
    monkeypatch.setattr(features, "generate_features",
                        lambda sims, **kw: [(fake_trios(), True)])
    X, stable = features.feature_matrix(object())
    assert X.shape == (1, len(features.FEATS))
    assert stable.tolist() == [True]


def test_feature_matrix_rejects_two_planets(monkeypatch):
    monkeypatch.setattr(features, "generate_features",
                        lambda sims, **kw: [([], True)])
    with pytest.raises(ValueError, match="3\\+ planet"):
        features.feature_matrix(object())
