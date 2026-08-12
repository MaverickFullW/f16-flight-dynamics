"""F-16 engine command and power-response functions."""

import numpy as np

from .interpolation import bilinear_interpolate


# Engine thrust data from Stevens, Lewis & Johnson Appendix A, FUNCTION
# THRUST. Values remain in the original lbf units; rows are altitude and
# columns are Mach number.
ALTITUDE_GRID_FT = np.array([0.0, 10000.0, 20000.0, 30000.0, 40000.0, 50000.0])

MACH_GRID = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

IDLE_THRUST_TABLE_LBF = np.array(
    [
        [1060.0,   635.0,    60.0, -1020.0, -2700.0, -3600.0],
        [ 670.0,   425.0,    25.0,  -710.0, -1900.0, -1400.0],
        [ 880.0,   690.0,   345.0,  -300.0, -1300.0,  -595.0],
        [1140.0,  1010.0,   755.0,   350.0,  -247.0,  -342.0],
        [1500.0,  1330.0,  1130.0,   910.0,   600.0,  -200.0],
        [1860.0,  1700.0,  1525.0,  1360.0,  1100.0,   700.0],
    ]
)

MIL_THRUST_TABLE_LBF = np.array(
    [
        [12680.0, 12680.0, 12610.0, 12640.0, 12390.0, 11680.0],
        [9150.0, 9150.0, 9312.0, 9839.0, 10176.0, 9848.0],
        [6200.0, 6313.0, 6610.0, 7090.0, 7750.0, 8050.0],
        [3950.0, 4040.0, 4290.0, 4660.0, 5320.0, 6100.0],
        [2450.0, 2470.0, 2600.0, 2840.0, 3250.0, 3800.0],
        [1400.0, 1400.0, 1560.0, 1660.0, 1930.0, 2310.0],
    ]
)

MAX_THRUST_TABLE_LBF = np.array(
    [
        [20000.0, 21420.0, 22700.0, 24240.0, 26070.0, 28886.0],
        [15000.0, 15700.0, 16860.0, 18910.0, 21075.0, 23319.0],
        [10800.0, 11225.0, 12250.0, 13760.0, 15975.0, 18300.0],
        [7000.0, 7323.0, 8154.0, 9285.0, 11115.0, 13484.0],
        [4000.0, 4435.0, 5000.0, 5700.0, 6860.0, 8642.0],
        [2500.0, 2600.0, 2835.0, 3215.0, 3950.0, 5057.0],
    ]
)


def _surrounding_indices(value, grid):
    """Return indices of the containing or nearest edge cell."""
    upper = int(np.clip(np.searchsorted(grid, value), 1, len(grid) - 1))
    return upper - 1, upper


def thrust_lbf(power, altitude_ft, mach):
    """
    Return F-16 engine thrust in lbf.

    Parameters
    ----------
    power : float
        Actual engine power in the original F-16 engine power units.
    altitude_ft : float
        Altitude in feet.
    mach : float
        Mach number.

    Returns
    -------
    float
        Engine thrust in pounds-force.
    """
    altitude0_index, altitude1_index = _surrounding_indices(
        altitude_ft, ALTITUDE_GRID_FT
    )
    mach0_index, mach1_index = _surrounding_indices(
        mach, MACH_GRID
    )

    altitude0 = ALTITUDE_GRID_FT[altitude0_index]
    altitude1 = ALTITUDE_GRID_FT[altitude1_index]
    mach0 = MACH_GRID[mach0_index]
    mach1 = MACH_GRID[mach1_index]

    thrust_mil = bilinear_interpolate(
        altitude_ft,
        mach,
        altitude0,
        altitude1,
        mach0,
        mach1,
        MIL_THRUST_TABLE_LBF[altitude0_index, mach0_index],
        MIL_THRUST_TABLE_LBF[altitude1_index, mach0_index],
        MIL_THRUST_TABLE_LBF[altitude0_index, mach1_index],
        MIL_THRUST_TABLE_LBF[altitude1_index, mach1_index],
    )

    if power < 50.0:
        thrust_idle = bilinear_interpolate(
            altitude_ft,
            mach,
            altitude0,
            altitude1,
            mach0,
            mach1,
            IDLE_THRUST_TABLE_LBF[altitude0_index, mach0_index],
            IDLE_THRUST_TABLE_LBF[altitude1_index, mach0_index],
            IDLE_THRUST_TABLE_LBF[altitude0_index, mach1_index],
            IDLE_THRUST_TABLE_LBF[altitude1_index, mach1_index],
        )

        return (
            thrust_idle
            + (thrust_mil - thrust_idle) * power * 0.02
        )

    thrust_max = bilinear_interpolate(
        altitude_ft,
        mach,
        altitude0,
        altitude1,
        mach0,
        mach1,
        MAX_THRUST_TABLE_LBF[altitude0_index, mach0_index],
        MAX_THRUST_TABLE_LBF[altitude1_index, mach0_index],
        MAX_THRUST_TABLE_LBF[altitude0_index, mach1_index],
        MAX_THRUST_TABLE_LBF[altitude1_index, mach1_index],
    )

    return (
        thrust_mil
        + (thrust_max - thrust_mil) * (power - 50.0) * 0.02
    )


def tgear(throttle):
    """
    Convert throttle position to commanded engine power.

    Parameters
    ----------
    throttle : float
        Dimensionless throttle position. Values are not clamped.

    Returns
    -------
    float
        Commanded power in the percent-like units of the original F-16
        model.
    """
    if throttle <= 0.77:
        return 64.94 * throttle
    return 217.38 * throttle - 117.38


def rtau(delta_power):
    """
    Return the reciprocal engine power-response time constant.

    Parameters
    ----------
    delta_power : float
        Power difference in the percent-like units of the original model.

    Returns
    -------
    float
        Reciprocal time constant for the engine power response.
    """
    if delta_power <= 25.0:
        return 1.0
    if delta_power >= 50.0:
        return 0.1
    return 1.9 - 0.036 * delta_power


def pdot(actual_power, commanded_power):
    """
    Compute the F-16 engine power-state derivative.

    Parameters
    ----------
    actual_power : float
        Current engine power in the percent-like units of the original
        F-16 model.
    commanded_power : float
        Commanded engine power in the same units as ``actual_power``.

    Returns
    -------
    float
        Rate of change of engine power in power units per second.
    """
    if commanded_power >= 50.0:
        if actual_power >= 50.0:
            target_power = commanded_power
            rate_factor = 5.0
        else:
            target_power = 60.0
            rate_factor = rtau(target_power - actual_power)
    else:
        if actual_power >= 50.0:
            target_power = 40.0
            rate_factor = 5.0
        else:
            target_power = commanded_power
            rate_factor = rtau(target_power - actual_power)

    return rate_factor * (target_power - actual_power)
