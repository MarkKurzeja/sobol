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


def test_choice_with_list():
    """Uniform categorical from a list."""
    result = sobol.df(8, sobol.choice("color", ["red", "green", "blue"]))
    assert result.shape == (8, 1)
    assert set(result["color"].unique()).issubset({"red", "green", "blue"})


def test_choice_with_dict():
    """Weighted categorical from a dict."""
    result = sobol.df(64, sobol.choice("color", {"red": 0.5, "green": 0.2, "blue": 0.3}))
    assert set(result["color"].unique()).issubset({"red", "green", "blue"})
    # Red should appear most often with weight 0.5
    counts = result["color"].value_counts()
    assert counts["red"] >= counts["green"]
    assert counts["red"] >= counts["blue"]


def test_choice_with_set():
    """Uniform categorical from a set."""
    result = sobol.df(16, sobol.choice("fruit", {"apple", "banana"}))
    assert set(result["fruit"].unique()).issubset({"apple", "banana"})


def test_choice_deterministic():
    a = sobol.df(16, sobol.choice("x", ["a", "b", "c"]))
    b = sobol.df(16, sobol.choice("x", ["a", "b", "c"]))
    pd.testing.assert_frame_equal(a, b)


def test_choice_relative_weights():
    """Weights are relative - {5, 2, 3} should behave like {0.5, 0.2, 0.3}."""
    a = sobol.df(64, sobol.choice("x", {"a": 5, "b": 2, "c": 3}))
    b = sobol.df(64, sobol.choice("x", {"a": 0.5, "b": 0.2, "c": 0.3}))
    pd.testing.assert_frame_equal(a, b)


def test_choice_empty_rejected():
    with pytest.raises(ValueError, match="[Ee]mpty"):
        sobol.choice("x", [])


def test_integer_bounds():
    result = sobol.df(64, sobol.integer("x", 1, 10))
    assert result["x"].min() >= 1
    assert result["x"].max() <= 10
    assert (result["x"] == result["x"].astype(int)).all()


def test_integer_covers_range():
    """With enough points, all integers in range should appear."""
    result = sobol.df(64, sobol.integer("x", 1, 4))
    assert set(result["x"].unique()) == {1, 2, 3, 4}


def test_integer_single_value():
    result = sobol.df(4, sobol.integer("x", 5, 5))
    assert (result["x"] == 5).all()


def test_boolean_values():
    result = sobol.df(16, sobol.boolean("flag"))
    assert result["flag"].dtype == bool
    assert set(result["flag"].unique()) == {True, False}


def test_boolean_deterministic():
    a = sobol.df(16, sobol.boolean("x"))
    b = sobol.df(16, sobol.boolean("x"))
    pd.testing.assert_frame_equal(a, b)


def test_offset_skips_points():
    """Offset=n should give the same result as generating 2n and taking the second half."""
    full = sobol.df(16, sobol.uniform("x", 0, 1))
    second_half = sobol.df(8, sobol.uniform("x", 0, 1), offset=8)
    pd.testing.assert_frame_equal(
        full.iloc[8:].reset_index(drop=True),
        second_half,
    )


def test_offset_extends_experiment():
    """First batch + offset batch should equal a single larger batch."""
    first = sobol.df(8, sobol.uniform("x", 0, 1), sobol.log("y", 1, 100))
    second = sobol.df(8, sobol.uniform("x", 0, 1), sobol.log("y", 1, 100), offset=8)
    combined = pd.concat([first, second], ignore_index=True)
    full = sobol.df(16, sobol.uniform("x", 0, 1), sobol.log("y", 1, 100))
    pd.testing.assert_frame_equal(combined, full)


def test_offset_must_be_nonnegative():
    with pytest.raises(ValueError, match="non-negative"):
        sobol.df(4, sobol.uniform("x", 0, 1), offset=-1)


def test_offset_must_be_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        sobol.df(4, sobol.uniform("x", 0, 1), offset=3)


def test_where_filters_rows():
    result = sobol.df(
        16,
        sobol.uniform("a", 0, 1),
        sobol.uniform("b", 0, 1),
        where=lambda row: row["a"] > row["b"],
    )
    assert len(result) == 16
    assert (result["a"] > result["b"]).all()


def test_where_with_choice():
    result = sobol.df(
        8,
        sobol.uniform("x", 0, 10),
        sobol.choice("color", ["red", "blue"]),
        where=lambda row: row["color"] == "red",
    )
    assert len(result) == 8
    assert (result["color"] == "red").all()


def test_rows_yields_dicts():
    result = list(sobol.rows(4, sobol.uniform("x", 0, 1), sobol.integer("y", 1, 5)))
    assert len(result) == 4
    assert all(isinstance(r, dict) for r in result)
    assert all("x" in r and "y" in r for r in result)


def test_rows_matches_df():
    dims = (sobol.uniform("a", 0, 1), sobol.choice("b", ["x", "y"]))
    row_list = list(sobol.rows(8, *dims))
    frame = sobol.df(8, *dims)
    for i, row in enumerate(row_list):
        assert row["a"] == frame.iloc[i]["a"]
        assert row["b"] == frame.iloc[i]["b"]


def test_rows_with_where():
    result = list(sobol.rows(
        8,
        sobol.uniform("x", 0, 1),
        sobol.uniform("y", 0, 1),
        where=lambda r: r["x"] > 0.5,
    ))
    assert len(result) == 8
    assert all(r["x"] > 0.5 for r in result)


def test_grid_choice():
    result = sobol.grid(
        sobol.choice("color", ["red", "blue"]),
        sobol.choice("size", ["S", "M", "L"]),
    )
    assert len(result) == 6  # 2 * 3
    assert set(result.columns) == {"color", "size"}


def test_grid_boolean():
    result = sobol.grid(sobol.boolean("a"), sobol.boolean("b"))
    assert len(result) == 4  # 2 * 2


def test_grid_integer():
    result = sobol.grid(sobol.integer("x", 1, 3), sobol.choice("y", ["a", "b"]))
    assert len(result) == 6  # 3 * 2
    assert set(result["x"].unique()) == {1, 2, 3}


def test_grid_rejects_continuous():
    with pytest.raises(ValueError, match="grid.*only supports"):
        sobol.grid(sobol.uniform("x", 0, 1))


def test_grid_duplicate_names():
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        sobol.grid(sobol.boolean("x"), sobol.boolean("x"))


def test_mixed_dimensions():
    """All dimension types work together."""
    result = sobol.df(
        16,
        sobol.uniform("a", 0, 1),
        sobol.log("b", 1, 100),
        sobol.normal("c", mean=0, std=1),
        sobol.choice("d", ["x", "y"]),
        sobol.integer("e", 1, 10),
        sobol.boolean("f"),
    )
    assert result.shape == (16, 6)
    assert list(result.columns) == ["a", "b", "c", "d", "e", "f"]
    assert result["a"].between(0, 1).all()
    assert result["b"].between(1, 100).all()
    assert set(result["d"].unique()).issubset({"x", "y"})
    assert result["e"].between(1, 10).all()
    assert result["f"].dtype == bool
