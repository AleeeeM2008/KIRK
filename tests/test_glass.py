import numpy as np
import torch

from pathlib import Path

from kirk import glass

ROOT = Path(__file__).resolve().parents[1]


def test_predict_is_probability():
    model, meta = glass.load_artifact(ROOT / "models" / "nam_spock_distilled.pt")
    x = np.asarray(meta["Xm"], dtype=np.float32)[None, :]
    p = glass.predict(model, meta, x)
    assert p.shape == (1,)
    assert np.isfinite(p[0])
    assert 0.0 <= p[0] <= 1.0


def test_constant_feature_is_zeroed():
    class Echo(torch.nn.Module):
        def forward(self, x):
            return x[:, 1:2]

    meta = {
        "Xm": np.zeros(2, dtype=np.float32),
        "Xs": np.array([1.0, 0.0], dtype=np.float32),
        "Ym": 0.0,
        "Ys": 1.0,
    }
    z = glass.predict_logit(Echo(), meta, np.array([[5.0, 7.0]], dtype=np.float32))
    assert z[0] == 0.0  # the constant column is fed its training value


def test_shapes_are_one_per_feature():
    model, meta = glass.load_artifact(ROOT / "models" / "nam_spock_exact.pt")
    curves = model.shapes(meta)
    assert len(curves) == len(meta["feats"])
    for grid, values in curves.values():
        assert grid.shape == values.shape == (50,)
        assert np.isfinite(values).all()


def test_artifact_without_output_scaling_raises():
    class Echo(torch.nn.Module):
        def forward(self, x):
            return x[:, :1]

    meta = {"Xm": np.zeros(1, dtype=np.float32), "Xs": np.ones(1, dtype=np.float32)}
    try:
        glass.predict_logit(Echo(), meta, np.zeros((1, 1), dtype=np.float32))
    except KeyError as exc:
        assert "Ym" in str(exc)
    else:
        raise AssertionError("predict_logit accepted an artifact without Ym/Ys")
