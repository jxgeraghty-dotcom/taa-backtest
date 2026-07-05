"""Map config strings to signal instances."""
from __future__ import annotations

from .momentum import AbsoluteMomentum, TrendMA
from .value import Valuation
from .carry import Carry
from .macro import MacroNowcast
from .composite import Composite

REGISTRY = {
    "abs_mom": AbsoluteMomentum,
    "trend_ma": TrendMA,
    "value": Valuation,
    "carry": Carry,
    "macro": MacroNowcast,
    "composite": Composite,
}


def build_signal(spec: dict):
    """spec example: {"type": "abs_mom", "lookback": 12}"""
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in REGISTRY:
        raise KeyError(f"unknown signal '{kind}', options: {list(REGISTRY)}")
    return REGISTRY[kind](**spec)
