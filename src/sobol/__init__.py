"""Sobol sequence sampling for experiment design."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator
from scipy.stats import norm, qmc


class Dimension(BaseModel, frozen=True):
    name: str
    kind: Literal["uniform", "log", "normal", "choice"]
    lower: float | None = None
    upper: float | None = None
    mean: float | None = None
    std: float | None = None
    categories: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> Dimension:
        if self.kind == "uniform":
            if self.lower is None or self.upper is None:
                raise ValueError("uniform dimension requires lower and upper")
        elif self.kind == "log":
            if self.lower is None or self.upper is None:
                raise ValueError("log dimension requires lower and upper")
            if self.lower <= 0:
                raise ValueError(
                    f"lower bound must be positive for log dimension, got {self.lower}"
                )
        elif self.kind == "normal":
            if self.mean is None or self.std is None:
                raise ValueError("normal dimension requires mean and std")
        elif self.kind == "choice":
            if len(self.categories) == 0:
                raise ValueError("Categories must not be empty")
            if len(self.weights) != len(self.categories):
                raise ValueError("weights and categories must have the same length")
        return self

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
        elif self.kind == "choice":
            # Map [0, 1) samples to categories using cumulative weights
            cumulative = np.cumsum(self.weights)
            indices = np.searchsorted(cumulative, samples, side="right")
            indices = np.clip(indices, 0, len(self.categories) - 1)
            return [self.categories[i] for i in indices]
        raise ValueError(f"Unknown kind: {self.kind}")


def uniform(name: str, lower: float = 0, upper: float = 1) -> Dimension:
    return Dimension(name=name, kind="uniform", lower=lower, upper=upper)


def log(name: str, lower: float = 0, upper: float = 1) -> Dimension:
    return Dimension(name=name, kind="log", lower=lower, upper=upper)


def normal(name: str, mean: float = 0, std: float = 1) -> Dimension:
    return Dimension(name=name, kind="normal", mean=mean, std=std)


def choice(name: str, categories: list | set | dict) -> Dimension:
    if isinstance(categories, dict):
        cats = tuple(categories.keys())
        raw_weights = tuple(categories.values())
    else:
        cats = tuple(categories)
        raw_weights = tuple(1.0 for _ in cats)

    if len(cats) == 0:
        raise ValueError("Categories must not be empty")

    total = sum(raw_weights)
    weights = tuple(w / total for w in raw_weights)

    return Dimension(name=name, kind="choice", categories=cats, weights=weights)


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
