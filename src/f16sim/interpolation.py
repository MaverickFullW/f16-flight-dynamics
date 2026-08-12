"""Reusable interpolation utilities for aerodynamic lookup tables."""


def linear_interpolate(x, x0, x1, y0, y1):
    """
    Linearly interpolate or extrapolate a scalar value.

    Parameters
    ----------
    x : float
        Coordinate at which to evaluate the interpolant.
    x0 : float
        Coordinate of the first known point.
    x1 : float
        Coordinate of the second known point.
    y0 : float
        Value at ``x0``.
    y1 : float
        Value at ``x1``.

    Returns
    -------
    float
        Interpolated or extrapolated value at ``x``.

    Raises
    ------
    ValueError
        If ``x0`` and ``x1`` are equal.
    """
    if x0 == x1:
        raise ValueError("x0 and x1 must be distinct")

    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def bilinear_interpolate(x, y, x0, x1, y0, y1, f00, f10, f01, f11):
    """
    Bilinearly interpolate or extrapolate a scalar value.

    Parameters
    ----------
    x : float
        First coordinate at which to evaluate the interpolant.
    y : float
        Second coordinate at which to evaluate the interpolant.
    x0 : float
        Lower cell coordinate along the first axis.
    x1 : float
        Upper cell coordinate along the first axis.
    y0 : float
        Lower cell coordinate along the second axis.
    y1 : float
        Upper cell coordinate along the second axis.
    f00 : float
        Value at ``(x0, y0)``.
    f10 : float
        Value at ``(x1, y0)``.
    f01 : float
        Value at ``(x0, y1)``.
    f11 : float
        Value at ``(x1, y1)``.

    Returns
    -------
    float
        Interpolated or extrapolated value at ``(x, y)``.

    Raises
    ------
    ValueError
        If ``x0`` equals ``x1`` or ``y0`` equals ``y1``.
    """
    value_at_y0 = linear_interpolate(x, x0, x1, f00, f10)
    value_at_y1 = linear_interpolate(x, x0, x1, f01, f11)
    return linear_interpolate(y, y0, y1, value_at_y0, value_at_y1)
