"""Simplified F-16 standard-atmosphere and air-data model."""

import numpy as np


# Exact unit conversions used at the SI boundary of the Appendix A model.
FT_TO_METER = 0.3048
SLUG_TO_KILOGRAM = 14.593902937206363
LBF_TO_NEWTON = 4.4482216152605

METER_TO_FT = 1.0 / FT_TO_METER
SLUG_PER_FT3_TO_KG_PER_M3 = SLUG_TO_KILOGRAM / FT_TO_METER**3
PSF_TO_PASCAL = LBF_TO_NEWTON / FT_TO_METER**2


def f16_air_data(true_airspeed, altitude_m):
    """
    Compute F-16 atmospheric density, Mach number, and dynamic pressure.

    This reproduces the simplified atmosphere in Stevens, Lewis & Johnson
    Appendix A ``SUBROUTINE ADC``. Inputs and outputs use SI units, while the
    original equations are evaluated internally in feet, seconds, and slugs.
    Altitude is not clamped.

    Parameters
    ----------
    true_airspeed : float
        True airspeed in meters per second.
    altitude_m : float
        Geometric altitude in meters.

    Returns
    -------
    density : float
        Air density in kilograms per cubic meter.
    mach : float
        Dimensionless Mach number.
    dynamic_pressure : float
        Dynamic pressure in pascals.

    Raises
    ------
    ValueError
        If ``true_airspeed`` is not positive.
    """
    if true_airspeed <= 0.0:
        raise ValueError("true_airspeed must be positive")

    velocity_ft_s = true_airspeed * METER_TO_FT
    altitude_ft = altitude_m * METER_TO_FT

    r0 = 2.377e-3
    tfac = 1.0 - 0.703e-5 * altitude_ft
    temperature_rankine = 519.0 * tfac
    if altitude_m >= 35000.0 * FT_TO_METER:
        temperature_rankine = 390.0

    rho_slug_ft3 = r0 * tfac**4.14
    mach = velocity_ft_s / np.sqrt(1.4 * 1716.3 * temperature_rankine)
    qbar_psf = 0.5 * rho_slug_ft3 * velocity_ft_s**2

    density = rho_slug_ft3 * SLUG_PER_FT3_TO_KG_PER_M3
    dynamic_pressure = qbar_psf * PSF_TO_PASCAL

    return float(density), float(mach), float(dynamic_pressure)
