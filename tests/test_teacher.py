import json

import pytest

from kirk import teacher

TREE = {
    "left_children": [1, -1, -1],
    "right_children": [2, -1, -1],
    "split_indices": [0, 0, 0],
    "split_conditions": [0.5, 0.0, 0.0],
    "base_weights": [0.0, -1.0, 1.0],
    "default_left": [0, 0, 0],
}


@pytest.fixture()
def tiny_teacher(tmp_path, monkeypatch):
    doc = {
        "learner": {
            "feature_names": ["x"],
            "learner_model_param": {"base_score": "5E-1"},
            "gradient_booster": {"model": {"trees": [TREE]}},
        }
    }
    path = tmp_path / "teacher.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setenv("KIRK_TEACHER_JSON", str(path))

    teacher._doc = None
    teacher.FEATS = None
    teacher.TREES = None
    teacher.BASE_M = None
    yield
    teacher._doc = None
    teacher.FEATS = None
    teacher.TREES = None
    teacher.BASE_M = None


def test_tree_walk_goes_left_and_right(tiny_teacher):
    left = teacher.logit({"x": 0.0})
    right = teacher.logit({"x": 1.0})
    assert right > left


def test_missing_feature_uses_default_branch(tiny_teacher):
    # NaN goes right here because default_left is 0.
    assert teacher.logit({}) == pytest.approx(teacher.logit({"x": 1.0}))


def test_prob_is_sigmoid(tiny_teacher):
    p = teacher.prob({"x": 1.0})
    assert 0.0 < p < 1.0
