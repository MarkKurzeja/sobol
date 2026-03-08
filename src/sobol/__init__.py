"""Sobol sequence sampling for experiment design."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc


@dataclass(frozen=True)
class Dimension:
    name: str
    kind: Literal["uniform", "log", "normal", "cat"]
    lower: float | None = None
    upper: float | None = None
    mean: float | None = None
    std: float | None = None
    categories: tuple[str, ...] = field(default_factory=tuple)
    weights: tuple[float, ...] = field(default_factory=tuple)

    def transform(self, samples: np.ndarray) -> np.ndarray | list:
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
        elif self.kind == "cat":
            # Map [0, 1) samples to categories using cumulative weights
            cumulative = np.cumsum(self.weights)
            indices = np.searchsorted(cumulative, samples, side="right")
            indices = np.clip(indices, 0, len(self.categories) - 1)
            return [self.categories[i] for i in indices]
        raise ValueError(f"Unknown kind: {self.kind}")


def uniform(name: str, lower: float = 0, upper: float = 1) -> Dimension:
    return Dimension(name=name, kind="uniform", lower=lower, upper=upper)


def log(name: str, lower: float = 0, upper: float = 1) -> Dimension:
    if lower <= 0:
        raise ValueError(f"lower bound must be positive for log dimension, got {lower}")
    return Dimension(name=name, kind="log", lower=lower, upper=upper)


def normal(name: str, mean: float = 0, std: float = 1) -> Dimension:
    return Dimension(name=name, kind="normal", mean=mean, std=std)


def cat(name: str, categories: list | set | dict) -> Dimension:
    if isinstance(categories, dict):
        cats = tuple(categories.keys())
        weights = tuple(categories.values())
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
    else:
        cats = tuple(categories)
        if len(cats) == 0:
            raise ValueError("Categories must not be empty")
        weights = tuple(1.0 / len(cats) for _ in cats)

    if len(cats) == 0:
        raise ValueError("Categories must not be empty")

    return Dimension(name=name, kind="cat", categories=cats, weights=weights)


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
