"""Sobol sequence sampling for experiment design."""

from __future__ import annotations

import itertools
from typing import Callable, Generator, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator
from scipy.stats import norm, qmc


class Dimension(BaseModel, frozen=True):
    name: str = ""
    kind: Literal["uniform", "log", "normal", "choice", "integer", "boolean"]
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
        elif self.kind == "integer":
            if self.lower is None or self.upper is None:
                raise ValueError("integer dimension requires lower and upper")
            if self.lower > self.upper:
                raise ValueError(
                    f"lower must be <= upper, got {self.lower} > {self.upper}"
                )
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
        elif self.kind == "integer":
            lo = int(self.lower)
            hi = int(self.upper)
            # Map [0, 1) to integers in [lo, hi] inclusive
            return np.floor(lo + samples * (hi - lo + 1)).astype(int).clip(lo, hi)
        elif self.kind == "boolean":
            return samples >= 0.5
        raise ValueError(f"Unknown kind: {self.kind}")


def uniform(lower: float = 0, upper: float = 1) -> Dimension:
    return Dimension(kind="uniform", lower=lower, upper=upper)


def log(lower: float = 0, upper: float = 1) -> Dimension:
    return Dimension(kind="log", lower=lower, upper=upper)


def normal(mean: float = 0, std: float = 1) -> Dimension:
    return Dimension(kind="normal", mean=mean, std=std)


def choice(categories: list | set | dict) -> Dimension:
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

    return Dimension(kind="choice", categories=cats, weights=weights)


def integer(lower: int, upper: int) -> Dimension:
    return Dimension(kind="integer", lower=float(lower), upper=float(upper))


def boolean() -> Dimension:
    return Dimension(kind="boolean")


def _resolve_dimensions(
    positional: tuple[Dimension, ...], kwargs: dict[str, Dimension]
) -> tuple[Dimension, ...]:
    """Merge positional dimensions with keyword dimensions."""
    dims = list(positional)
    for name, dim in kwargs.items():
        if not isinstance(dim, Dimension):
            raise TypeError(f"Expected a Dimension for {name!r}, got {type(dim).__name__}")
        dims.append(dim.model_copy(update={"name": name}))
    return tuple(dims)


def rows(
    n: int,
    *dimensions: Dimension,
    offset: int = 0,
    where: Callable[[pd.Series], bool] | None = None,
    **kwargs: Dimension,
) -> Generator[dict, None, None]:
    result = sample(n, *dimensions, offset=offset, where=where, **kwargs)
    for _, row in result.iterrows():
        yield row.to_dict()


def grid(*dimensions: Dimension, **kwargs: Dimension) -> pd.DataFrame:
    dimensions = _resolve_dimensions(dimensions, kwargs)
    names = [d.name for d in dimensions]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate dimension names: {names}")

    value_lists = []
    for dim in dimensions:
        if dim.kind == "choice":
            value_lists.append(list(dim.categories))
        elif dim.kind == "boolean":
            value_lists.append([False, True])
        elif dim.kind == "integer":
            value_lists.append(list(range(int(dim.lower), int(dim.upper) + 1)))
        else:
            raise ValueError(
                f"grid() only supports choice, boolean, and integer dimensions, "
                f"got {dim.kind!r} for {dim.name!r}"
            )

    rows_list = list(itertools.product(*value_lists))
    return pd.DataFrame(rows_list, columns=names)


def sample(
    n: int,
    *dimensions: Dimension,
    offset: int = 0,
    where: Callable[[pd.Series], bool] | None = None,
    **kwargs: Dimension,
) -> pd.DataFrame:
    dimensions = _resolve_dimensions(dimensions, kwargs)
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a positive power of 2, got {n}")

    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    if offset > 0 and (offset & (offset - 1)) != 0:
        raise ValueError(f"offset must be a power of 2, got {offset}")

    names = [d.name for d in dimensions]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate dimension names: {names}")

    if where is None:
        # Simple path: generate exactly n points
        d = len(dimensions)
        sampler = qmc.Sobol(d=d, scramble=False)
        if offset > 0:
            sampler.fast_forward(offset)
        raw = sampler.random(n)

        data = {}
        for i, dim in enumerate(dimensions):
            data[dim.name] = dim.transform(raw[:, i])

        return pd.DataFrame(data)
    else:
        # Constraint path: oversample and filter
        d = len(dimensions)
        collected = []
        cursor = offset
        batch_size = n

        while len(collected) < n:
            sampler = qmc.Sobol(d=d, scramble=False)
            if cursor > 0:
                sampler.fast_forward(cursor)
            raw = sampler.random(batch_size)
            cursor += batch_size

            data = {}
            for i, dim in enumerate(dimensions):
                data[dim.name] = dim.transform(raw[:, i])
            batch_df = pd.DataFrame(data)

            mask = batch_df.apply(where, axis=1)
            collected.append(batch_df[mask])

            # Double batch size for next attempt
            batch_size = min(batch_size * 2, 2**20)

        result = pd.concat(collected, ignore_index=True).head(n)
        if len(result) < n:
            raise ValueError(
                f"Could not find {n} samples satisfying the constraint"
            )
        return result


# Backwards compatibility alias
df = sample
