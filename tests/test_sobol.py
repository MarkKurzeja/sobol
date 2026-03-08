import numpy as np
import pandas as pd
import pytest

import sobol


def test_df_returns_dataframe():
    result = sobol.df(4, sobol.uniform("x", 0, 1))
    assert isinstance(result, pd.DataFrame)


def test_df_correct_shape():
    result = sobol.df(8, sobol.uniform("a", 0, 1), sobol.uniform("b", 0, 1))
    assert result.shape == (8, 2)


def test_df_column_names():
    result = sobol.df(4, sobol.uniform("taco", 0, 5), sobol.log("burrito", 1, 100))
    assert list(result.columns) == ["taco", "burrito"]


def test_uniform_bounds():
    result = sobol.df(64, sobol.uniform("x", 3, 7))
    assert result["x"].min() >= 3
    assert result["x"].max() <= 7


def test_log_bounds():
    result = sobol.df(64, sobol.log("y", 0.01, 1000))
    assert result["y"].min() >= 0.01
    assert result["y"].max() <= 1000


def test_log_distribution_in_log_space():
    """Values should be spread across orders of magnitude, not clustered near zero."""
    result = sobol.df(64, sobol.log("y", 1, 10000))
    log_values = np.log10(result["y"])
    # Should span most of the log range [0, 4]
    assert log_values.max() - log_values.min() > 3.0


def test_n_must_be_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        sobol.df(3, sobol.uniform("x", 0, 1))


def test_n_must_be_positive():
    with pytest.raises(ValueError):
        sobol.df(0, sobol.uniform("x", 0, 1))


def test_deterministic():
    a = sobol.df(16, sobol.uniform("x", 0, 1), sobol.log("y", 1, 100))
    b = sobol.df(16, sobol.uniform("x", 0, 1), sobol.log("y", 1, 100))
    pd.testing.assert_frame_equal(a, b)


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        sobol.df(4, sobol.uniform("x", 0, 1), sobol.uniform("x", 0, 1))


def test_log_lower_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        sobol.log("x", 0, 100)

    with pytest.raises(ValueError, match="positive"):
        sobol.log("x", -1, 100)


def test_normal_distribution():
    result = sobol.df(64, sobol.normal("z", mean=10, std=2))
    # Mean should be close to 10
    assert abs(result["z"].mean() - 10) < 1.0
    # Std should be roughly close to 2
    assert 1.0 < result["z"].std() < 4.0


def test_normal_symmetry():
    """With enough Sobol points, the sample mean should closely match the specified mean."""
    result = sobol.df(256, sobol.normal("z", mean=50, std=5))
    assert abs(result["z"].mean() - 50) < 0.5


def test_mixed_dimensions():
    """All three dimension types work together."""
    result = sobol.df(
        16,
        sobol.uniform("a", 0, 1),
        sobol.log("b", 1, 100),
        sobol.normal("c", mean=0, std=1),
    )
    assert result.shape == (16, 3)
    assert list(result.columns) == ["a", "b", "c"]
    assert result["a"].between(0, 1).all()
    assert result["b"].between(1, 100).all()
