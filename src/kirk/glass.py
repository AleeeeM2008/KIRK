"""Glass-box models: NAM, mixture of EBM experts, analytic prefilter.

The point is that a prediction is bias plus a sum of one-dimensional shapes,
so you can read it. That is the whole trick. Do not add a wide hidden layer
here; if a feature needs to interact with another, put it in EBMExpert.

Inference is torch only. No xgboost, no sklearn. The .pt files in ../models
carry the weights, the feature order and the normalization, so one file is
enough to reproduce a prediction.
"""
import numpy as np
import torch
import torch.nn as nn


class NAM(nn.Module):
    """logit = bias + sum_i f_i(x_i).

    Each f_i is a small net with one input. Plot it, edit it, argue with it.
    """

    def __init__(self, d=11, h=32):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(1, h), nn.Tanh(),
                          nn.Linear(h, h), nn.Tanh(),
                          nn.Linear(h, 1))
            for _ in range(d)
        ])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.bias + sum([n(x[:, i:i + 1]) for i, n in enumerate(self.nets)])

    def shapes(self, meta, X=None, xgrid=50, nstd=2.0):
        """Sample each f_i on a grid so you can plot it.

        The nets see normalized inputs, so the grid is normalized before the
        call and returned in raw units. Pass X (raw feature matrix) to use its
        min/max per feature. Leave X out and you get mean +/- nstd*std from the
        artifact, which is usually the sane plot.

        Returns {feature_index: (raw_grid, f_values)}.
        """
        xm = torch.as_tensor(meta["Xm"], dtype=torch.float32)
        xs = torch.as_tensor(meta["Xs"], dtype=torch.float32)
        if X is None:
            lo, hi = xm - nstd * xs, xm + nstd * xs
        else:
            Xt = torch.as_tensor(np.atleast_2d(np.asarray(X)), dtype=torch.float32)
            lo, hi = Xt.min(0).values, Xt.max(0).values

        self.eval()
        out = {}
        with torch.no_grad():
            for i, net in enumerate(self.nets):
                g = torch.linspace(float(lo[i]), float(hi[i]), xgrid)
                if float(xs[i]) < 1e-6:
                    gn = torch.zeros(xgrid, 1)  # feature never moved in training
                else:
                    gn = ((g - xm[i]) / xs[i]).unsqueeze(1)
                out[i] = (g.numpy(), net(gn).numpy().ravel())
        return out


class EBMExpert(nn.Module):
    """One expert: 11 univariate shapes plus 3 pairwise heatmaps.

    The pair indices are feature positions. If you reorder FEATS in
    features.py, fix `pairs` too or you are training on nonsense.
    """

    def __init__(self, d=11, h=16, ph=10, pairs=None):
        super().__init__()
        self.uni = nn.ModuleList([
            nn.Sequential(nn.Linear(1, h), nn.Tanh(), nn.Linear(h, 1))
            for _ in range(d)
        ])
        self.bi = nn.ModuleList([
            nn.Sequential(nn.Linear(2, ph), nn.Tanh(), nn.Linear(ph, 1))
            for _ in range(3)
        ])
        self.bias = nn.Parameter(torch.zeros(1))
        # (EMcross near, MMR near), (EMcross far, MMR far), (MEGNO, EMfracstd)
        self.pairs = pairs or [(0, 3), (4, 7), (8, 1)]

    def forward(self, x):
        o = self.bias + sum([n(x[:, i:i + 1]) for i, n in enumerate(self.uni)])
        for net, (a, b) in zip(self.bi, self.pairs):
            o = o + net(torch.stack([x[:, a], x[:, b]], 1))
        return o


class MoE(nn.Module):
    """Mixture of EBM experts with a linear router.

    The router only sees compactness and log-resonantness. That is deliberate:
    it is small enough to inspect, and it keeps the gates physical.
    """

    def __init__(self, K=3, d=11):
        super().__init__()
        self.experts = nn.ModuleList([EBMExpert(d) for _ in range(K)])
        self.router = nn.Linear(2, K)
        self.K = K

    def forward(self, x, r, ret_gates=False):
        g = torch.softmax(self.router(r), 1)
        outs = [e(x) for e in self.experts]
        mix = sum([g[:, k:k + 1] * outs[k] for k in range(self.K)])
        if ret_gates:
            return mix, g
        return mix


def load_artifact(path, arch="nam"):
    """Load a ../models/*.pt file. Returns (model, meta).

    The file holds the state dict, the feature order, and Xm/Xs/Ym/Ys.
    Everything needed for inference is in there.
    """
    d = torch.load(path, map_location="cpu", weights_only=False)
    feats = d["feats"]
    if arch == "nam":
        m = NAM(len(feats))
    elif arch == "moe":
        raise NotImplementedError(
            "The MoE artifacts in models/ were saved with a legacy key layout. "
            "load_artifact does not support them; see models/index.json.")
    else:
        raise ValueError(f"unknown arch {arch!r}; use 'nam' or 'moe'")
    m.load_state_dict(d["w"])
    m.eval()
    return m, d


def _normalize(meta, X):
    """Scale raw features with the artifact's mean/std.

    A few artifacts were trained with a feature that never moved (the saved
    std is an epsilon). Dividing by that epsilon turns a harmless feature into
    a 1e8 input and wrecks the prediction. For those columns we feed the
    training value, i.e. zero after scaling. That is what the model saw.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    xm = np.asarray(meta["Xm"], dtype=np.float32)
    xs = np.asarray(meta["Xs"], dtype=np.float32)
    constant = xs < 1e-6
    z = (X - xm) / np.where(constant, 1.0, xs)
    z[:, constant] = 0.0
    return torch.from_numpy(z)


def predict_logit(model, meta, X):
    """Raw log-odds. X is raw (unscaled) features, one row per system."""
    if "Ym" not in meta or "Ys" not in meta:
        raise KeyError(
            "artifact has no Ym/Ys; it is not a predict_logit artifact. "
            "Use a model saved by the standard training path, e.g. nam_spock_exact.pt.")
    with torch.no_grad():
        out = model(_normalize(meta, X))
        if isinstance(out, tuple):
            out = out[0]
        return out.numpy().ravel() * meta["Ys"] + meta["Ym"]


def predict(model, meta, X):
    """Probability of stability. Same input as predict_logit."""
    z = predict_logit(model, meta, X)
    return 1.0 / (1.0 + np.exp(-z))
