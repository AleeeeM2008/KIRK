"""kirk: glass-box distilled SPOCK and a scheduler for N-body work.

Three pieces, in order of how often you touch them:

  glass     -- additive models you can read and edit
  scheduler -- decide which systems deserve a real integration
  features  -- the SPOCK feature pipeline, ported in-tree

REBOUND does the N-body. We do not re-implement integrators.

License: GPL-3.0 because SPOCK is GPL-3.0. See LICENSE for upstream credits.
"""
from .features import generate_features, predict_instability_time, predict_stable  # noqa: E402,F401
from .scheduler import triage  # noqa: E402,F401

__all__ = ["generate_features", "predict_stable", "predict_instability_time", "triage"]
__version__ = "0.1.0"
