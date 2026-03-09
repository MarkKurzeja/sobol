# qsweep

A Python library for generating quasi-random Sobol sequences for experiment design and hyperparameter search.

Instead of grid search (which wastes samples when only a few dimensions matter) or random search (which can leave gaps), Sobol sequences fill the parameter space evenly and deterministically. This means better coverage with fewer experiments.

Inspired by [*Critical Hyper-Parameters: No Random, No Cry*](https://arxiv.org/abs/1706.03200) (Bousquet et al., 2017), which shows that low-discrepancy sequences like Sobol consistently outperform random search for hyperparameter optimization across deep learning tasks.

## Install

```bash
uv add qsweep
```

## Quick start

```python
import qsweep

# Generate 16 experiment configurations
df = qsweep.sample(
    16,
    learning_rate=qsweep.log(1e-5, 1e-1),
    dropout=qsweep.uniform(0.0, 0.5),
    optimizer=qsweep.choice({"adam": 3, "sgd": 1}),
    batch_size=qsweep.choice([32, 64, 128, 256]),
)
print(df)
```

## Dimension types

| Constructor | Description | Example |
|---|---|---|
| `uniform(lower, upper)` | Uniform over `[lower, upper]` | `qsweep.uniform(0, 1)` |
| `log(lower, upper)` | Log-uniform over `[lower, upper]` (lower must be > 0) | `qsweep.log(1e-5, 1e-1)` |
| `normal(mean, std)` | Normal distribution via inverse CDF | `qsweep.normal(0, 1)` |
| `integer(lower, upper)` | Integers in `[lower, upper]` inclusive | `qsweep.integer(1, 10)` |
| `boolean()` | `True` or `False` | `qsweep.boolean()` |
| `choice(categories)` | Categorical; pass a list, set, or weighted dict | `qsweep.choice(["a", "b"])` |

### Weighted choices

Pass a dict to assign relative weights:

```python
qsweep.choice({"adam": 3, "sgd": 1})  # adam sampled 3x more often
```

## API

### `qsweep.sample(n, *, offset=0, where=None, **dimensions) -> pd.DataFrame`

Generate `n` quasi-random samples. `n` must be a power of 2.

```python
df = qsweep.sample(64, x=qsweep.uniform(0, 10), y=qsweep.uniform(0, 10))
```

**Parameters:**
- `n` — Number of samples (must be a power of 2).
- `offset` — Skip the first `offset` points in the sequence (must be 0 or a power of 2). Useful for extending a previous experiment without repeating points.
- `where` — A filter function `(row: pd.Series) -> bool`. Only rows where this returns `True` are kept. The sampler over-generates internally to fill the quota.
- `**dimensions` — Keyword arguments mapping names to dimension specs.

### `qsweep.rows(n, *, offset=0, where=None, **dimensions) -> Generator[dict]`

Same as `sample()`, but yields one `dict` per row. Convenient for feeding configs into a training loop:

```python
for config in qsweep.rows(16, lr=qsweep.log(1e-4, 1e-1), wd=qsweep.log(1e-6, 1e-2)):
    train(lr=config["lr"], wd=config["wd"])
```

### `qsweep.grid(**dimensions) -> pd.DataFrame`

Full Cartesian product of discrete dimensions (`choice`, `integer`, `boolean`). Rejects continuous dimensions.

```python
df = qsweep.grid(
    optimizer=qsweep.choice(["adam", "sgd"]),
    use_scheduler=qsweep.boolean(),
)
# Returns 4 rows: all combinations
```

## Extending experiments with `offset`

Run an initial 16-point experiment, then extend with 16 more points that continue the Sobol sequence:

```python
batch_1 = qsweep.sample(16, x=qsweep.uniform(0, 1))
batch_2 = qsweep.sample(16, x=qsweep.uniform(0, 1), offset=16)
```

The combined 32 points have the same low-discrepancy properties as generating 32 at once.

## Filtering with `where`

```python
df = qsweep.sample(
    32,
    x=qsweep.uniform(0, 10),
    y=qsweep.uniform(0, 10),
    where=lambda row: row["x"] + row["y"] < 15,
)
# All 32 rows satisfy x + y < 15
```

## Why Sobol over random?

Sobol sequences are *quasi-random* — they're deterministic and designed to cover the unit hypercube as evenly as possible. Compared to pseudorandom sampling:

- **Better coverage**: no clusters or gaps, even in high dimensions.
- **Deterministic**: same inputs always produce the same experiments.
- **Extensible**: generate more points later without re-running earlier ones.

For hyperparameter search specifically, [Bousquet et al. (2017)](https://arxiv.org/abs/1706.03200) showed that low-discrepancy sequences find good hyperparameters with significantly fewer evaluations than random search (p-value = 0.0002), validated on LSTM language models and image classifiers.

## References

- Bousquet, O., Gelly, S., Kurach, K., Teytaud, O., & Vincent, D. (2017). [*Critical Hyper-Parameters: No Random, No Cry*](https://arxiv.org/abs/1706.03200). arXiv:1706.03200.
- Sobol', I.M. (1967). *On the distribution of points in a cube and the approximate evaluation of integrals*. USSR Computational Mathematics and Mathematical Physics, 7(4), 86–112.
