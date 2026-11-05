import rebound

from kirk import heads


def sim_three():
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.add(m=1.0)
    for P in (1.0, 1.4, 2.0):
        sim.add(m=1e-5, P=P, e=0.01, pomega=0.3, l=0.7)
    sim.move_to_com()
    return sim


def test_collision_features_shape():
    assert heads.collision_features(sim_three()).shape == (18,)


def test_collision_probabilities_sum_to_one():
    p = heads.predict_collision(sim_three())
    assert set(p) == {"(1,2)", "(2,3)", "(1,3)"}
    assert abs(sum(p.values()) - 1.0) < 1e-6


def test_outcome_is_positive():
    a1, e1, a2, e2 = heads.predict_outcome(sim_three())
    assert a1 > 0 and a2 > 0
    assert 0 <= e1 < 1 and 0 <= e2 < 1
