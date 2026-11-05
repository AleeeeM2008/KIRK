"""Extra heads that came with the distillation: collisions and outcomes.

These are small additive models, same idea as the stability NAM. They are
instant: the inputs are orbital elements, no short integration.

The collision head predicts which of the three pairs hits first. The outcome
head predicts the post-merger orbits, but it was trained only on the (1,2)
collision, so do not treat it as a general collision solver.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _model_path(name):
    return Path(__file__).resolve().parents[2] / "models" / name


def collision_features(sim):
    """18 values: three pairs of (a ratio, Hill gap, e1, e2, log m1, log m2)."""
    ps = sim.particles
    n_real = getattr(sim, "N_real", sim.N - getattr(sim, "N_var", 0))
    if n_real < 4:
        raise ValueError("collision features need a 3-planet system")
    out = []
    for i1, i2 in ((1, 2), (2, 3), (1, 3)):
        a1, a2 = ps[i1].a, ps[i2].a
        e1, e2 = ps[i1].e, ps[i2].e
        m1, m2 = ps[i1].m, ps[i2].m
        rh = a1 * ((m1 + m2) / 3.0 / ps[0].m) ** (1 / 3.)
        out += [a2 / a1, (a2 - a1) / rh, e1, e2,
                np.log10(m1), np.log10(m2)]
    return np.array(out, np.float32)


class CollisionHead(nn.Module):
    """Three independent additive scores, softmaxed into pair probabilities."""

    def __init__(self, d=18, h=24, k=3):
        super().__init__()
        self.d = d
        self.k = k
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(1, h), nn.Tanh(), nn.Linear(h, 1))
            for _ in range(d * k)
        ])
        self.bias = nn.Parameter(torch.zeros(k))

    def forward(self, x):
        outs = []
        for j in range(self.k):
            terms = [self.nets[j * self.d + i](x[:, i:i + 1])
                     for i in range(self.d)]
            outs.append((self.bias[j] + sum(terms)).squeeze(-1))
        return torch.stack(outs, dim=1)


class OutcomeHead(nn.Module):
    """Four additive regressions for log10(a1, e1, a2, e2) after a merger."""

    def __init__(self, d=18, h=24, k=4):
        super().__init__()
        self.d = d
        self.k = k
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(1, h), nn.Tanh(), nn.Linear(h, 1))
            for _ in range(d * k)
        ])
        self.bias = nn.Parameter(torch.zeros(k))

    def forward(self, x):
        outs = []
        for j in range(self.k):
            terms = [self.nets[j * self.d + i](x[:, i:i + 1])
                     for i in range(self.d)]
            outs.append((self.bias[j] + sum(terms)).squeeze(-1))
        return torch.stack(outs, dim=1)


def _normalize(x, meta):
    return (x - meta["Xm"]) / np.where(np.asarray(meta["Xs"]) < 1e-6, 1.0,
                                       meta["Xs"])


def predict_collision(sim, model_path=None):
    """P(pair collides first) for (1,2), (2,3), (1,3). Sums to 1."""
    path = model_path or _model_path("nam_spock_collision.pt")
    d = torch.load(path, map_location="cpu", weights_only=False)
    model = CollisionHead(d=d["d"])
    model.load_state_dict(d["w"])
    model.eval()

    x = _normalize(collision_features(sim), d).astype(np.float32)
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(x[None, :])), 1).numpy()[0]
    return {"(1,2)": float(p[0]), "(2,3)": float(p[1]), "(1,3)": float(p[2])}


def predict_outcome(sim, model_path=None):
    """Post-merger orbits for the (1,2) collision, as (a1, e1, a2, e2).

    The training run only covered pair (1,2). Do not use this for the other
    pairs without retraining.
    """
    path = model_path or _model_path("nam_spock_outcome.pt")
    d = torch.load(path, map_location="cpu", weights_only=False)
    model = OutcomeHead()
    model.load_state_dict(d["w"])
    model.eval()

    x = _normalize(collision_features(sim), d).astype(np.float32)
    with torch.no_grad():
        raw = model(torch.from_numpy(x[None, :])).numpy()[0]
    values = raw * np.asarray(d["Ys"]) + np.asarray(d["Ym"])
    return tuple(float(v) for v in 10.0 ** values)
