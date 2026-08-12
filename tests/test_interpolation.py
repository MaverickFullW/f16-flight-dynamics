import numpy as np
import pytest

from src.f16sim.interpolation import bilinear_interpolate, linear_interpolate


def test_linear_interpolate_returns_y0_at_x0():
    assert np.isclose(linear_interpolate(1.0, 1.0, 5.0, 3.0, 11.0), 3.0)


def test_linear_interpolate_returns_y1_at_x1():
    assert np.isclose(linear_interpolate(5.0, 1.0, 5.0, 3.0, 11.0), 11.0)


def test_linear_interpolate_returns_midpoint():
    assert np.isclose(linear_interpolate(3.0, 1.0, 5.0, 3.0, 11.0), 7.0)


def test_linear_interpolate_extrapolates_outside_interval():
    assert np.isclose(linear_interpolate(7.0, 1.0, 5.0, 3.0, 11.0), 15.0)


def test_linear_interpolate_raises_if_x_coordinates_are_equal():
    with pytest.raises(ValueError):
        linear_interpolate(1.0, 2.0, 2.0, 3.0, 4.0)


def _linear_function(x, y):
    return 2.0 * x + 3.0 * y + 5.0


def _interpolate_linear_function(x, y, x0=1.0, x1=4.0, y0=-2.0, y1=3.0):
    return bilinear_interpolate(
        x,
        y,
        x0,
        x1,
        y0,
        y1,
        _linear_function(x0, y0),
        _linear_function(x1, y0),
        _linear_function(x0, y1),
        _linear_function(x1, y1),
    )


def test_bilinear_interpolate_reproduces_linear_function_inside_cell():
    x = 2.5
    y = 1.0

    assert np.isclose(_interpolate_linear_function(x, y), _linear_function(x, y))


def test_bilinear_interpolate_extrapolates_linear_function_outside_cell():
    x = 6.0
    y = -4.0

    assert np.isclose(_interpolate_linear_function(x, y), _linear_function(x, y))


@pytest.mark.parametrize(
    ("x", "y"),
    [(1.0, -2.0), (4.0, -2.0), (1.0, 3.0), (4.0, 3.0)],
)
def test_bilinear_interpolate_returns_exact_corner_values(x, y):
    assert _interpolate_linear_function(x, y) == _linear_function(x, y)


def test_bilinear_interpolate_raises_if_x_coordinates_are_equal():
    with pytest.raises(ValueError):
        bilinear_interpolate(1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0)


def test_bilinear_interpolate_raises_if_y_coordinates_are_equal():
    with pytest.raises(ValueError):
        bilinear_interpolate(1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0)
