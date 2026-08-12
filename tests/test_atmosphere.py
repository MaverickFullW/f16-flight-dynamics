import numpy as np
import pytest

from src.f16sim.atmosphere import f16_air_data


FT_TO_METER = 0.3048
SLUG_TO_KILOGRAM = 14.593902937206363
LBF_TO_NEWTON = 4.4482216152605


def _appendix_a_expected(true_airspeed, altitude_m):
    velocity_ft_s = true_airspeed / FT_TO_METER
    altitude_ft = altitude_m / FT_TO_METER
    tfac = 1.0 - 0.703e-5 * altitude_ft
    temperature = 390.0 if altitude_ft >= 35000.0 else 519.0 * tfac
    density_slug_ft3 = 2.377e-3 * tfac**4.14
    mach = velocity_ft_s / np.sqrt(1.4 * 1716.3 * temperature)
    dynamic_pressure_psf = 0.5 * density_slug_ft3 * velocity_ft_s**2

    density = density_slug_ft3 * SLUG_TO_KILOGRAM / FT_TO_METER**3
    dynamic_pressure = dynamic_pressure_psf * LBF_TO_NEWTON / FT_TO_METER**2
    return density, mach, dynamic_pressure


def test_f16_air_data_at_sea_level():
    true_airspeed = 100.0
    expected = _appendix_a_expected(true_airspeed, 0.0)

    actual = f16_air_data(true_airspeed, 0.0)

    assert np.allclose(actual, expected)


def test_sea_level_density_matches_converted_appendix_a_density():
    expected_density = 2.377e-3 * SLUG_TO_KILOGRAM / FT_TO_METER**3

    density, _, _ = f16_air_data(100.0, 0.0)

    assert np.isclose(density, expected_density)


def test_dynamic_pressure_satisfies_si_relation():
    true_airspeed = 100.0
    density, _, dynamic_pressure = f16_air_data(true_airspeed, 0.0)

    assert np.isclose(dynamic_pressure, 0.5 * density * true_airspeed**2)


def test_f16_air_data_below_35000_feet():
    true_airspeed = 150.0
    altitude_m = 10000.0 * FT_TO_METER
    expected = _appendix_a_expected(true_airspeed, altitude_m)

    assert np.allclose(f16_air_data(true_airspeed, altitude_m), expected)


def test_f16_air_data_uses_390_rankine_at_35000_feet():
    true_airspeed = 200.0
    altitude_m = 35000.0 * FT_TO_METER
    velocity_ft_s = true_airspeed / FT_TO_METER
    expected_mach = velocity_ft_s / np.sqrt(1.4 * 1716.3 * 390.0)

    _, mach, _ = f16_air_data(true_airspeed, altitude_m)

    assert np.isclose(mach, expected_mach)


def test_f16_air_data_above_35000_feet():
    true_airspeed = 200.0
    altitude_m = 40000.0 * FT_TO_METER
    expected = _appendix_a_expected(true_airspeed, altitude_m)

    assert np.allclose(f16_air_data(true_airspeed, altitude_m), expected)


def test_mach_increases_with_true_airspeed_at_fixed_altitude():
    altitude_m = 20000.0 * FT_TO_METER
    _, lower_mach, _ = f16_air_data(100.0, altitude_m)
    _, higher_mach, _ = f16_air_data(200.0, altitude_m)

    assert higher_mach > lower_mach


def test_density_decreases_from_sea_level_to_30000_feet():
    sea_level_density, _, _ = f16_air_data(100.0, 0.0)
    high_altitude_density, _, _ = f16_air_data(
        100.0, 30000.0 * FT_TO_METER
    )

    assert high_altitude_density < sea_level_density


def test_zero_true_airspeed_raises_value_error():
    with pytest.raises(ValueError):
        f16_air_data(0.0, 0.0)


def test_negative_true_airspeed_raises_value_error():
    with pytest.raises(ValueError):
        f16_air_data(-1.0, 0.0)


def test_negative_altitude_is_not_clamped():
    true_airspeed = 100.0
    altitude_m = -1000.0 * FT_TO_METER
    expected = _appendix_a_expected(true_airspeed, altitude_m)
    actual = f16_air_data(true_airspeed, altitude_m)
    sea_level = f16_air_data(true_airspeed, 0.0)

    assert np.allclose(actual, expected)
    assert not np.isclose(actual[0], sea_level[0])
