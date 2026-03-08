"""Sobol sequence sampling for experiment design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc


@dataclass(frozen=True)
class Dimension:
    name: str
    kind: Literal["uniform", "log", "normal"]
    lower: float | None = None
    upper: float | None = None
    mean: float | None = None
    std: float | None = None

    def transform(self, samples: np.ndarray) -> np.ndarray:
        if self.kind == "uniform":
            return self.lower + samples * (self.upper - self.lower)
        elif self.kind == "log":
            log_lower = np.log(self.lower)
            log_upper = np.log(self.upper)
            return np.exp(log_lower + samples * (log_upper - log_lower))
        elif self.kind == "normal":
            # Clamp to avoid -inf/inf at the boundaries of the Sobol sequence
            clamped = np.clip(samples, 1e-10, 1 - 1e-10)
            return norm.ppf(clamped, loc=self.mean, scale=self.std)
        raise ValueError(f"Unknown kind: {self.kind}")


def uniform(name: str, lower: float = 0, upper: float = 1) -> Dimension:
    return Dimension(name=name, kind="uniform", lower=lower, upper=upper)


def log(name: str, lower: float = 0, upper: float = 1) -> Dimension:
    if lower <= 0:
        raise ValueError(f"lower bound must be positive for log dimension, got {lower}")
    return Dimension(name=name, kind="log", lower=lower, upper=upper)


def normal(name: str, mean: float = 0, std: float = 1) -> Dimension:
    return Dimension(name=name, kind="normal", mean=mean, std=std)


def df(n: int, *dimensions: Dimension) -> pd.DataFrame:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a positive power of 2, got {n}")

    names = [d.name for d in dimensions]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate dimension names: {names}")

    d = len(dimensions)
    sampler = qmc.Sobol(d=d, scramble=False)
    raw = sampler.random(n)  # shape (n, d), values in [0, 1)

    data = {}
    for i, dim in enumerate(dimensions):
        data[dim.name] = dim.transform(raw[:, i])

    return pd.DataFrame(data)
