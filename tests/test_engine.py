import numpy as np
import pytest

from src.f16sim.engine import (
    ALTITUDE_GRID_FT,
    IDLE_THRUST_TABLE_LBF,
    MACH_GRID,
    MAX_THRUST_TABLE_LBF,
    MIL_THRUST_TABLE_LBF,
    pdot,
    rtau,
    tgear,
    thrust_lbf,
)


def test_tgear_at_zero_throttle():
    assert tgear(0.0) == pytest.approx(0.0)


def test_tgear_in_lower_branch():
    assert tgear(0.5) == pytest.approx(64.94 * 0.5)


def test_tgear_at_full_throttle():
    assert tgear(1.0) == pytest.approx(100.0)

def test_tgear_uses_lower_branch_at_boundary():
    assert np.isclose(tgear(0.77), 64.94 * 0.77)


def test_tgear_uses_upper_branch_above_boundary():
    throttle = 0.770001
    assert np.isclose(
        tgear(throttle),
        217.38 * throttle - 117.38,
    )

@pytest.mark.parametrize(
    ("throttle", "expected"),
    [(-0.2, 64.94 * -0.2), (1.2, 217.38 * 1.2 - 117.38)],
)
def test_tgear_does_not_clamp_outside_nominal_range(throttle, expected):
    assert tgear(throttle) == pytest.approx(expected)


def test_rtau_at_zero_delta_power():
    assert rtau(0.0) == pytest.approx(1.0)


def test_rtau_at_lower_boundary():
    assert rtau(25.0) == pytest.approx(1.0)


def test_rtau_at_upper_boundary():
    assert rtau(50.0) == pytest.approx(0.1)


def test_rtau_interior_value():
    delta_power = 37.5

    assert rtau(delta_power) == pytest.approx(1.9 - 0.036 * delta_power)


def test_rtau_below_nominal_range():
    assert rtau(-10.0) == pytest.approx(1.0)


def test_rtau_above_upper_boundary():
    assert rtau(75.0) == pytest.approx(0.1)


def test_pdot_high_command_high_actual_branch():
    actual_power = 60.0
    commanded_power = 80.0

    assert pdot(actual_power, commanded_power) == pytest.approx(
        5.0 * (commanded_power - actual_power)
    )


def test_pdot_high_command_low_actual_uses_sixty_target_and_rtau():
    actual_power = 20.0
    commanded_power = 80.0
    target_power = 60.0
    expected = rtau(target_power - actual_power) * (target_power - actual_power)

    assert pdot(actual_power, commanded_power) == pytest.approx(expected)


def test_pdot_low_command_high_actual_uses_forty_target_and_rate_five():
    actual_power = 60.0
    commanded_power = 30.0
    target_power = 40.0

    assert pdot(actual_power, commanded_power) == pytest.approx(
        5.0 * (target_power - actual_power)
    )


def test_pdot_low_command_low_actual_uses_command_target_and_rtau():
    actual_power = 10.0
    commanded_power = 40.0
    delta_power = commanded_power - actual_power
    expected = rtau(delta_power) * delta_power

    assert pdot(actual_power, commanded_power) == pytest.approx(expected)


def test_pdot_is_zero_at_equal_power_below_fifty():
    assert pdot(30.0, 30.0) == pytest.approx(0.0)


def test_pdot_is_zero_at_equal_power_above_fifty():
    assert pdot(70.0, 70.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("altitude_index", "mach_index"),
    [
        (altitude_index, mach_index)
        for altitude_index in range(len(ALTITUDE_GRID_FT))
        for mach_index in range(len(MACH_GRID))
    ],
)
def test_thrust_lbf_reproduces_idle_table_nodes(altitude_index, mach_index):
    assert np.isclose(
        thrust_lbf(
            0.0,
            ALTITUDE_GRID_FT[altitude_index],
            MACH_GRID[mach_index],
        ),
        IDLE_THRUST_TABLE_LBF[altitude_index, mach_index],
    )


@pytest.mark.parametrize(
    ("altitude_index", "mach_index"),
    [
        (altitude_index, mach_index)
        for altitude_index in range(len(ALTITUDE_GRID_FT))
        for mach_index in range(len(MACH_GRID))
    ],
)
def test_thrust_lbf_reproduces_military_table_nodes(altitude_index, mach_index):
    assert np.isclose(
        thrust_lbf(
            50.0,
            ALTITUDE_GRID_FT[altitude_index],
            MACH_GRID[mach_index],
        ),
        MIL_THRUST_TABLE_LBF[altitude_index, mach_index],
    )


@pytest.mark.parametrize(
    ("altitude_index", "mach_index"),
    [
        (altitude_index, mach_index)
        for altitude_index in range(len(ALTITUDE_GRID_FT))
        for mach_index in range(len(MACH_GRID))
    ],
)
def test_thrust_lbf_reproduces_maximum_table_nodes(altitude_index, mach_index):
    assert np.isclose(
        thrust_lbf(
            100.0,
            ALTITUDE_GRID_FT[altitude_index],
            MACH_GRID[mach_index],
        ),
        MAX_THRUST_TABLE_LBF[altitude_index, mach_index],
    )


def test_thrust_lbf_interpolates_power_below_military():
    altitude_index = 2
    mach_index = 3
    idle = IDLE_THRUST_TABLE_LBF[altitude_index, mach_index]
    military = MIL_THRUST_TABLE_LBF[altitude_index, mach_index]
    expected = idle + 0.5 * (military - idle)

    assert np.isclose(
        thrust_lbf(25.0, ALTITUDE_GRID_FT[altitude_index], MACH_GRID[mach_index]),
        expected,
    )


def test_thrust_lbf_interpolates_power_above_military():
    altitude_index = 2
    mach_index = 3
    military = MIL_THRUST_TABLE_LBF[altitude_index, mach_index]
    maximum = MAX_THRUST_TABLE_LBF[altitude_index, mach_index]
    expected = military + 0.5 * (maximum - military)

    assert np.isclose(
        thrust_lbf(75.0, ALTITUDE_GRID_FT[altitude_index], MACH_GRID[mach_index]),
        expected,
    )


def test_thrust_lbf_bilinearly_interpolates_altitude_and_mach():
    surrounding_military_values = MIL_THRUST_TABLE_LBF[:2, :2]
    expected = np.mean(surrounding_military_values)

    assert np.isclose(thrust_lbf(50.0, 5000.0, 0.1), expected)


def test_thrust_lbf_combines_spatial_and_power_interpolation():
    idle_midpoint = np.mean(IDLE_THRUST_TABLE_LBF[:2, :2])
    military_midpoint = np.mean(MIL_THRUST_TABLE_LBF[:2, :2])
    expected = idle_midpoint + 0.5 * (military_midpoint - idle_midpoint)

    assert np.isclose(thrust_lbf(25.0, 5000.0, 0.1), expected)


def test_thrust_lbf_extrapolates_below_altitude_grid():
    expected = 1.5 * MIL_THRUST_TABLE_LBF[0, 0] - 0.5 * MIL_THRUST_TABLE_LBF[1, 0]

    assert np.isclose(thrust_lbf(50.0, -5000.0, 0.0), expected)
    assert not np.isclose(thrust_lbf(50.0, -5000.0, 0.0), MIL_THRUST_TABLE_LBF[0, 0])


def test_thrust_lbf_extrapolates_above_altitude_grid():
    expected = 1.5 * MIL_THRUST_TABLE_LBF[-1, 0] - 0.5 * MIL_THRUST_TABLE_LBF[-2, 0]

    assert np.isclose(thrust_lbf(50.0, 55000.0, 0.0), expected)
    assert not np.isclose(thrust_lbf(50.0, 55000.0, 0.0), MIL_THRUST_TABLE_LBF[-1, 0])


def test_thrust_lbf_extrapolates_below_mach_grid():
    altitude_ft = 20000.0
    mach = -0.1

    expected = (
        1.5 * MIL_THRUST_TABLE_LBF[2, 0]
        - 0.5 * MIL_THRUST_TABLE_LBF[2, 1]
    )

    actual = thrust_lbf(
        50.0,
        altitude_ft,
        mach,
    )

    assert np.isclose(actual, expected)
    assert not np.isclose(
        actual,
        MIL_THRUST_TABLE_LBF[2, 0],
    )


def test_thrust_lbf_extrapolates_above_mach_grid():
    expected = 1.5 * MIL_THRUST_TABLE_LBF[1, -1] - 0.5 * MIL_THRUST_TABLE_LBF[1, -2]

    assert np.isclose(thrust_lbf(50.0, 10000.0, 1.1), expected)
    assert not np.isclose(thrust_lbf(50.0, 10000.0, 1.1), MIL_THRUST_TABLE_LBF[1, -1])
