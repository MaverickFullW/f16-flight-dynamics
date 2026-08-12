"""Aerodynamic coefficient lookup functions for the reduced F-16 model."""

import numpy as np

from .air_data import air_data_from_body_velocity
from .interpolation import bilinear_interpolate, linear_interpolate
from .parameters import (
    mean_aerodynamic_chord,
    reference_cg_fraction,
    span,
    wing_area,
)


# Grids and CX data from Stevens, Lewis & Johnson, Appendix A, for the
# reduced F-16 aerodynamic model. Rows are angle of attack and columns are
# elevator deflection; all grid coordinates are expressed in degrees.
ALPHA_GRID_DEG = np.array(
    [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
)

ELEVATOR_GRID_DEG = np.array([-24.0, -12.0, 0.0, 12.0, 24.0])

CX_TABLE = np.array(
    [
        [-0.099, -0.048, -0.022, -0.040, -0.083],
        [-0.081, -0.038, -0.020, -0.038, -0.073],
        [-0.081, -0.040, -0.021, -0.039, -0.076],
        [-0.063, -0.021, -0.004, -0.025, -0.072],
        [-0.025, 0.016, 0.032, 0.006, -0.046],
        [0.044, 0.083, 0.094, 0.062, 0.012],
        [0.097, 0.127, 0.128, 0.087, 0.024],
        [0.113, 0.137, 0.130, 0.085, 0.025],
        [0.145, 0.162, 0.154, 0.100, 0.043],
        [0.167, 0.177, 0.161, 0.110, 0.053],
        [0.174, 0.179, 0.155, 0.104, 0.047],
        [0.166, 0.167, 0.138, 0.091, 0.040],
    ]
)

CZ_ALPHA_TABLE = np.array(
    [
        0.770,
        0.241,
        -0.100,
        -0.416,
        -0.731,
        -1.053,
        -1.366,
        -1.646,
        -1.917,
        -2.120,
        -2.248,
        -2.229,
    ]
)

CM_TABLE = np.array(
    [
        [0.205, 0.081, -0.046, -0.174, -0.259],
        [0.168, 0.077, -0.020, -0.145, -0.202],
        [0.186, 0.107, -0.009, -0.121, -0.184],
        [0.196, 0.110, -0.005, -0.127, -0.193],
        [0.213, 0.110, -0.006, -0.129, -0.199],
        [0.251, 0.141, 0.010, -0.102, -0.150],
        [0.245, 0.127, 0.006, -0.097, -0.160],
        [0.238, 0.119, -0.001, -0.113, -0.167],
        [0.252, 0.133, 0.014, -0.087, -0.104],
        [0.231, 0.108, 0.000, -0.084, -0.076],
        [0.198, 0.081, -0.013, -0.069, -0.041],
        [0.192, 0.093, 0.032, -0.006, -0.005],
    ]
)

BETA_ABS_GRID_DEG = np.array([0, 5, 10, 15, 20, 25, 30], dtype=float)

BETA_10_GRID_DEG = np.array([-30, -20, -10, 0, 10, 20, 30], dtype=float)

CL_TABLE = np.array(
    [
        [0.000, -0.001, -0.003, -0.001, 0.000, 0.007, 0.009],
        [0.000, -0.004, -0.009, -0.010, -0.010, -0.010, -0.011],
        [0.000, -0.008, -0.017, -0.020, -0.022, -0.023, -0.023],
        [0.000, -0.012, -0.024, -0.030, -0.034, -0.034, -0.037],
        [0.000, -0.016, -0.030, -0.039, -0.047, -0.049, -0.050],
        [0.000, -0.019, -0.034, -0.044, -0.046, -0.046, -0.047],
        [0.000, -0.020, -0.040, -0.050, -0.059, -0.068, -0.074],
        [0.000, -0.020, -0.037, -0.049, -0.061, -0.071, -0.079],
        [0.000, -0.015, -0.016, -0.023, -0.033, -0.060, -0.091],
        [0.000, -0.008, -0.002, -0.006, -0.036, -0.058, -0.076],
        [0.000, -0.013, -0.010, -0.014, -0.035, -0.062, -0.077],
        [0.000, -0.015, -0.019, -0.027, -0.035, -0.059, -0.076],
    ]
)

CN_TABLE = np.array(
    [
        [0.000, 0.018, 0.038, 0.056, 0.064, 0.074, 0.079],
        [0.000, 0.019, 0.042, 0.057, 0.077, 0.086, 0.090],
        [0.000, 0.018, 0.042, 0.059, 0.076, 0.093, 0.106],
        [0.000, 0.019, 0.042, 0.058, 0.074, 0.089, 0.106],
        [0.000, 0.019, 0.043, 0.058, 0.073, 0.080, 0.096],
        [0.000, 0.018, 0.039, 0.053, 0.057, 0.062, 0.080],
        [0.000, 0.013, 0.030, 0.032, 0.029, 0.049, 0.068],
        [0.000, 0.007, 0.017, 0.012, 0.007, 0.022, 0.030],
        [0.000, 0.004, 0.004, 0.002, 0.012, 0.028, 0.064],
        [0.000, -0.014, -0.035, -0.046, -0.034, -0.012, 0.015],
        [0.000, -0.017, -0.047, -0.071, -0.065, -0.002, 0.011],
        [0.000, -0.033, -0.057, -0.073, -0.041, -0.013, -0.001],
    ]
)

# Damping derivatives from Stevens, Lewis & Johnson Appendix A. Rows follow
# ALPHA_GRID_DEG; columns are CXq, CYr, CYp, CZq, Clr, Clp, Cmq, Cnr, Cnp.
DAMP_TABLE = np.array(
    [
        [-0.267, 0.882, -0.108, -8.80, -0.126, -0.360, -7.21, -0.380, 0.061],
        [-0.110, 0.852, -0.108, -25.8, -0.026, -0.359, -0.54, -0.363, 0.052],
        [0.308, 0.876, -0.188, -28.9, 0.063, -0.443, -5.23, -0.378, 0.052],
        [1.34, 0.958, 0.110, -31.4, 0.113, -0.420, -5.26, -0.386, -0.012],
        [2.08, 0.962, 0.258, -31.2, 0.208, -0.383, -6.11, -0.370, -0.013],
        [2.91, 0.974, 0.226, -30.7, 0.230, -0.375, -6.64, -0.453, -0.024],
        [2.76, 0.819, 0.344, -27.7, 0.319, -0.329, -5.69, -0.550, 0.050],
        [2.05, 0.483, 0.362, -28.2, 0.437, -0.294, -6.00, -0.582, 0.150],
        [1.50, 0.590, 0.611, -29.0, 0.680, -0.230, -6.20, -0.595, 0.130],
        [1.49, 1.21, 0.529, -29.8, 0.100, -0.210, -6.40, -0.637, 0.158],
        [1.83, -0.493, 0.298, -38.3, 0.447, -0.120, -6.60, -1.02, 0.240],
        [1.21, -1.04, -2.27, -35.3, -0.330, -0.100, -6.00, -0.840, 0.150],
    ]
)

# Rolling-moment derivative due to aileron from Stevens, Lewis & Johnson
# Appendix A. Rows are alpha and columns are signed sideslip in degrees.
DLDA_TABLE = np.array(
    [
        [-0.041, -0.041, -0.042, -0.040, -0.043, -0.044, -0.043],
        [-0.052, -0.053, -0.053, -0.052, -0.049, -0.048, -0.049],
        [-0.053, -0.053, -0.052, -0.051, -0.048, -0.048, -0.047],
        [-0.056, -0.053, -0.051, -0.052, -0.049, -0.047, -0.045],
        [-0.050, -0.050, -0.049, -0.048, -0.043, -0.042, -0.042],
        [-0.056, -0.051, -0.049, -0.048, -0.042, -0.041, -0.037],
        [-0.082, -0.066, -0.043, -0.042, -0.042, -0.020, -0.003],
        [-0.059, -0.043, -0.035, -0.037, -0.036, -0.028, -0.013],
        [-0.042, -0.038, -0.026, -0.031, -0.025, -0.013, -0.010],
        [-0.038, -0.027, -0.016, -0.026, -0.021, -0.014, -0.003],
        [-0.027, -0.023, -0.018, -0.017, -0.016, -0.011, -0.007],
        [-0.017, -0.016, -0.014, -0.012, -0.011, -0.010, -0.008],
    ]
)

# Rolling-moment derivative due to rudder from Stevens, Lewis & Johnson
# Appendix A. Rows are alpha and columns are signed sideslip in degrees.
DLDR_TABLE = np.array(
    [
        [0.005, 0.007, 0.013, 0.018, 0.015, 0.021, 0.023],
        [0.017, 0.016, 0.013, 0.015, 0.014, 0.011, 0.010],
        [0.014, 0.014, 0.011, 0.015, 0.013, 0.010, 0.011],
        [0.010, 0.014, 0.012, 0.014, 0.013, 0.011, 0.011],
        [-0.005, 0.013, 0.011, 0.014, 0.012, 0.010, 0.011],
        [0.009, 0.009, 0.009, 0.014, 0.011, 0.009, 0.010],
        [0.019, 0.012, 0.008, 0.014, 0.011, 0.008, 0.008],
        [0.005, 0.005, 0.005, 0.015, 0.010, 0.010, 0.010],
        [0.000, 0.000, -0.002, 0.013, 0.008, 0.006, 0.006],
        [-0.005, 0.004, 0.005, 0.011, 0.008, 0.005, 0.014],
        [-0.011, 0.009, 0.003, 0.006, 0.007, 0.000, 0.020],
        [0.008, 0.007, 0.005, 0.001, 0.003, 0.001, 0.000],
    ]
)

# Yawing-moment derivative due to aileron from Stevens, Lewis & Johnson
# Appendix A. Rows are alpha and columns are signed sideslip in degrees.
DNDA_TABLE = np.array(
    [
        [0.001, 0.002, -0.006, -0.011, -0.015, -0.024, -0.022],
        [-0.027, -0.014, -0.008, -0.011, -0.015, -0.010, 0.002],
        [-0.017, -0.016, -0.006, -0.010, -0.014, -0.004, -0.003],
        [-0.013, -0.016, -0.006, -0.009, -0.012, -0.002, -0.005],
        [-0.012, -0.014, -0.005, -0.008, -0.011, -0.001, -0.003],
        [-0.016, -0.019, -0.008, -0.006, -0.008, 0.003, -0.001],
        [0.001, -0.021, -0.005, 0.000, -0.002, 0.014, -0.009],
        [0.017, 0.002, 0.007, 0.004, 0.002, 0.006, -0.009],
        [0.011, 0.012, 0.004, 0.007, 0.006, -0.001, -0.001],
        [0.017, 0.015, 0.007, 0.010, 0.012, 0.004, 0.003],
        [0.008, 0.015, 0.006, 0.004, 0.011, 0.004, -0.002],
        [0.016, 0.011, 0.006, 0.010, 0.011, 0.006, 0.001],
    ]
)

# Yawing-moment derivative due to rudder from Stevens, Lewis & Johnson
# Appendix A. Rows are alpha and columns are signed sideslip in degrees.
DNDR_TABLE = np.array(
    [
        [-0.018, -0.028, -0.037, -0.048, -0.043, -0.052, -0.062],
        [-0.052, -0.051, -0.041, -0.045, -0.044, -0.034, -0.034],
        [-0.052, -0.043, -0.038, -0.045, -0.041, -0.036, -0.027],
        [-0.052, -0.046, -0.040, -0.045, -0.041, -0.036, -0.028],
        [-0.054, -0.045, -0.040, -0.044, -0.040, -0.035, -0.027],
        [-0.049, -0.049, -0.038, -0.045, -0.038, -0.028, -0.027],
        [-0.059, -0.057, -0.037, -0.047, -0.034, -0.024, -0.023],
        [-0.051, -0.052, -0.030, -0.048, -0.035, -0.023, -0.023],
        [-0.030, -0.030, -0.027, -0.049, -0.035, -0.020, -0.019],
        [-0.037, -0.033, -0.024, -0.045, -0.029, -0.016, -0.009],
        [-0.026, -0.030, -0.019, -0.033, -0.022, -0.010, -0.025],
        [-0.013, -0.008, -0.013, -0.016, -0.009, -0.014, -0.010],
    ]
)


def _surrounding_indices(value, grid):
    """Return indices of the containing or nearest edge cell."""
    upper = int(np.clip(np.searchsorted(grid, value), 1, len(grid) - 1))
    return upper - 1, upper


def cx(alpha_deg, elevator_deg):
    """
    Return the F-16 base x-axis aerodynamic coefficient.

    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.
    elevator_deg : float
        Elevator deflection in degrees.

    Returns
    -------
    float
        Dimensionless base x-axis aerodynamic coefficient. Values outside
        the tabulated grids are extrapolated using the nearest edge cell.
    """
    alpha0_index, alpha1_index = _surrounding_indices(alpha_deg, ALPHA_GRID_DEG)
    elevator0_index, elevator1_index = _surrounding_indices(
        elevator_deg, ELEVATOR_GRID_DEG
    )

    alpha0 = ALPHA_GRID_DEG[alpha0_index]
    alpha1 = ALPHA_GRID_DEG[alpha1_index]
    elevator0 = ELEVATOR_GRID_DEG[elevator0_index]
    elevator1 = ELEVATOR_GRID_DEG[elevator1_index]

    return bilinear_interpolate(
        alpha_deg,
        elevator_deg,
        alpha0,
        alpha1,
        elevator0,
        elevator1,
        CX_TABLE[alpha0_index, elevator0_index],
        CX_TABLE[alpha1_index, elevator0_index],
        CX_TABLE[alpha0_index, elevator1_index],
        CX_TABLE[alpha1_index, elevator1_index],
    )


def cy(beta_deg, aileron_deg, rudder_deg):
    """
    Return the F-16 base side-force aerodynamic coefficient.

    Positive and negative signs follow the original F-16 body-axis
    convention used by Stevens, Lewis & Johnson Appendix A.

    Parameters
    ----------
    beta_deg : float
        Sideslip angle in degrees.
    aileron_deg : float
        Aileron deflection in degrees.
    rudder_deg : float
        Rudder deflection in degrees.

    Returns
    -------
    float
        Dimensionless base side-force aerodynamic coefficient.
    """
    return (
        -0.02 * beta_deg
        + 0.021 * (aileron_deg / 20.0)
        + 0.086 * (rudder_deg / 30.0)
    )


def cz(alpha_deg, beta_deg, elevator_deg):
    """
    Return the F-16 base z-axis aerodynamic coefficient.

    The coefficient follows the Stevens, Lewis & Johnson Appendix A reduced
    F-16 model. Values outside the angle-of-attack grid are extrapolated from
    the nearest edge interval.

    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.
    beta_deg : float
        Sideslip angle in degrees.
    elevator_deg : float
        Elevator deflection in degrees.

    Returns
    -------
    float
        Dimensionless base z-axis aerodynamic coefficient.
    """
    alpha0_index, alpha1_index = _surrounding_indices(alpha_deg, ALPHA_GRID_DEG)
    cz_alpha = linear_interpolate(
        alpha_deg,
        ALPHA_GRID_DEG[alpha0_index],
        ALPHA_GRID_DEG[alpha1_index],
        CZ_ALPHA_TABLE[alpha0_index],
        CZ_ALPHA_TABLE[alpha1_index],
    )

    return cz_alpha * (1.0 - (beta_deg / 57.3) ** 2) - 0.19 * (
        elevator_deg / 25.0
    )


def cm(alpha_deg, elevator_deg):
    """
    Return the F-16 base pitching-moment aerodynamic coefficient.

    Values outside the Stevens, Lewis & Johnson Appendix A grids are
    extrapolated using the nearest edge cell.

    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.
    elevator_deg : float
        Elevator deflection in degrees.

    Returns
    -------
    float
        Dimensionless base pitching-moment aerodynamic coefficient.
    """
    alpha0_index, alpha1_index = _surrounding_indices(alpha_deg, ALPHA_GRID_DEG)
    elevator0_index, elevator1_index = _surrounding_indices(
        elevator_deg, ELEVATOR_GRID_DEG
    )

    return bilinear_interpolate(
        alpha_deg,
        elevator_deg,
        ALPHA_GRID_DEG[alpha0_index],
        ALPHA_GRID_DEG[alpha1_index],
        ELEVATOR_GRID_DEG[elevator0_index],
        ELEVATOR_GRID_DEG[elevator1_index],
        CM_TABLE[alpha0_index, elevator0_index],
        CM_TABLE[alpha1_index, elevator0_index],
        CM_TABLE[alpha0_index, elevator1_index],
        CM_TABLE[alpha1_index, elevator1_index],
    )


def cl(alpha_deg, beta_deg):
    """
    Return the F-16 base rolling-moment aerodynamic coefficient.

    The Stevens, Lewis & Johnson Appendix A table is interpolated against
    angle of attack and absolute sideslip. Values outside either grid are
    extrapolated using the nearest edge cell, and odd sideslip symmetry is
    restored after interpolation.

    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.
    beta_deg : float
        Sideslip angle in degrees.

    Returns
    -------
    float
        Dimensionless base rolling-moment aerodynamic coefficient.
    """
    if beta_deg == 0:
        return 0.0

    beta_abs_deg = abs(beta_deg)
    alpha0_index, alpha1_index = _surrounding_indices(alpha_deg, ALPHA_GRID_DEG)
    beta0_index, beta1_index = _surrounding_indices(
        beta_abs_deg, BETA_ABS_GRID_DEG
    )

    magnitude = bilinear_interpolate(
        alpha_deg,
        beta_abs_deg,
        ALPHA_GRID_DEG[alpha0_index],
        ALPHA_GRID_DEG[alpha1_index],
        BETA_ABS_GRID_DEG[beta0_index],
        BETA_ABS_GRID_DEG[beta1_index],
        CL_TABLE[alpha0_index, beta0_index],
        CL_TABLE[alpha1_index, beta0_index],
        CL_TABLE[alpha0_index, beta1_index],
        CL_TABLE[alpha1_index, beta1_index],
    )

    # Appendix A prints ``CL = DUM + SIGN(1.0, BETA)``. The plus sign is a
    # known typographical error; the intended operation restores the sign by
    # multiplication, ``CL = DUM * sign(beta)``.
    return magnitude * (1.0 if beta_deg > 0.0 else -1.0)


def cn(alpha_deg, beta_deg):
    """
    Return the F-16 base yawing-moment aerodynamic coefficient.

    Following Stevens, Lewis & Johnson Appendix A, the table uses angle of
    attack and absolute sideslip and the same interpolation procedure as the
    rolling-moment coefficient. Values outside either grid are extrapolated
    using the nearest edge cell, then odd sideslip symmetry is restored.

    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.
    beta_deg : float
        Sideslip angle in degrees.

    Returns
    -------
    float
        Dimensionless base yawing-moment aerodynamic coefficient.
    """
    if beta_deg == 0:
        return 0.0

    beta_abs_deg = abs(beta_deg)
    alpha0_index, alpha1_index = _surrounding_indices(alpha_deg, ALPHA_GRID_DEG)
    beta0_index, beta1_index = _surrounding_indices(
        beta_abs_deg, BETA_ABS_GRID_DEG
    )

    magnitude = bilinear_interpolate(
        alpha_deg,
        beta_abs_deg,
        ALPHA_GRID_DEG[alpha0_index],
        ALPHA_GRID_DEG[alpha1_index],
        BETA_ABS_GRID_DEG[beta0_index],
        BETA_ABS_GRID_DEG[beta1_index],
        CN_TABLE[alpha0_index, beta0_index],
        CN_TABLE[alpha1_index, beta0_index],
        CN_TABLE[alpha0_index, beta1_index],
        CN_TABLE[alpha1_index, beta1_index],
    )

    return magnitude * (1.0 if beta_deg > 0.0 else -1.0)


def damp(alpha_deg):
    """
    Return the F-16 aerodynamic damping derivatives at an angle of attack.

    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.

    Returns
    -------
    numpy.ndarray
        Nine-element array in the original Stevens, Lewis & Johnson
        Appendix A order: ``[CXq, CYr, CYp, CZq, Clr, Clp, Cmq, Cnr,
        Cnp]``. Values outside the angle-of-attack grid are extrapolated
        using the nearest edge interval.
    """
    alpha0_index, alpha1_index = _surrounding_indices(alpha_deg, ALPHA_GRID_DEG)

    return np.array(
        [
            linear_interpolate(
                alpha_deg,
                ALPHA_GRID_DEG[alpha0_index],
                ALPHA_GRID_DEG[alpha1_index],
                DAMP_TABLE[alpha0_index, derivative_index],
                DAMP_TABLE[alpha1_index, derivative_index],
            )
            for derivative_index in range(DAMP_TABLE.shape[1])
        ]
    )


def dlda(alpha_deg, beta_deg):
    """
    Return the F-16 rolling-moment derivative due to aileron.

    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.
    beta_deg : float
        Signed sideslip angle in degrees.

    Returns
    -------
    float
        Dimensionless rolling-moment derivative due to aileron. Values
        outside either grid are extrapolated using the nearest edge cell.
    """
    alpha0_index, alpha1_index = _surrounding_indices(alpha_deg, ALPHA_GRID_DEG)
    beta0_index, beta1_index = _surrounding_indices(beta_deg, BETA_10_GRID_DEG)

    return bilinear_interpolate(
        alpha_deg,
        beta_deg,
        ALPHA_GRID_DEG[alpha0_index],
        ALPHA_GRID_DEG[alpha1_index],
        BETA_10_GRID_DEG[beta0_index],
        BETA_10_GRID_DEG[beta1_index],
        DLDA_TABLE[alpha0_index, beta0_index],
        DLDA_TABLE[alpha1_index, beta0_index],
        DLDA_TABLE[alpha0_index, beta1_index],
        DLDA_TABLE[alpha1_index, beta1_index],
    )


def dldr(alpha_deg, beta_deg):
    """
    Return the F-16 rolling-moment derivative due to rudder.

    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.
    beta_deg : float
        Signed sideslip angle in degrees.

    Returns
    -------
    float
        Dimensionless rolling-moment derivative due to rudder. Values
        outside either grid are extrapolated using the nearest edge cell.
    """
    alpha0_index, alpha1_index = _surrounding_indices(alpha_deg, ALPHA_GRID_DEG)
    beta0_index, beta1_index = _surrounding_indices(beta_deg, BETA_10_GRID_DEG)

    return bilinear_interpolate(
        alpha_deg,
        beta_deg,
        ALPHA_GRID_DEG[alpha0_index],
        ALPHA_GRID_DEG[alpha1_index],
        BETA_10_GRID_DEG[beta0_index],
        BETA_10_GRID_DEG[beta1_index],
        DLDR_TABLE[alpha0_index, beta0_index],
        DLDR_TABLE[alpha1_index, beta0_index],
        DLDR_TABLE[alpha0_index, beta1_index],
        DLDR_TABLE[alpha1_index, beta1_index],
    )


def dnda(alpha_deg, beta_deg):
    """
    Return the F-16 yawing-moment derivative due to aileron.

    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.
    beta_deg : float
        Signed sideslip angle in degrees.

    Returns
    -------
    float
        Dimensionless yawing-moment derivative due to aileron. Values
        outside either grid are extrapolated using the nearest edge cell.
    """
    alpha0_index, alpha1_index = _surrounding_indices(alpha_deg, ALPHA_GRID_DEG)
    beta0_index, beta1_index = _surrounding_indices(beta_deg, BETA_10_GRID_DEG)

    return bilinear_interpolate(
        alpha_deg,
        beta_deg,
        ALPHA_GRID_DEG[alpha0_index],
        ALPHA_GRID_DEG[alpha1_index],
        BETA_10_GRID_DEG[beta0_index],
        BETA_10_GRID_DEG[beta1_index],
        DNDA_TABLE[alpha0_index, beta0_index],
        DNDA_TABLE[alpha1_index, beta0_index],
        DNDA_TABLE[alpha0_index, beta1_index],
        DNDA_TABLE[alpha1_index, beta1_index],
    )


def dndr(alpha_deg, beta_deg):
    """
    Return the F-16 yawing-moment derivative due to rudder.

    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.
    beta_deg : float
        Signed sideslip angle in degrees.

    Returns
    -------
    float
        Dimensionless yawing-moment derivative due to rudder. Values
        outside either grid are extrapolated using the nearest edge cell.
    """
    alpha0_index, alpha1_index = _surrounding_indices(alpha_deg, ALPHA_GRID_DEG)
    beta0_index, beta1_index = _surrounding_indices(beta_deg, BETA_10_GRID_DEG)

    return bilinear_interpolate(
        alpha_deg,
        beta_deg,
        ALPHA_GRID_DEG[alpha0_index],
        ALPHA_GRID_DEG[alpha1_index],
        BETA_10_GRID_DEG[beta0_index],
        BETA_10_GRID_DEG[beta1_index],
        DNDR_TABLE[alpha0_index, beta0_index],
        DNDR_TABLE[alpha1_index, beta0_index],
        DNDR_TABLE[alpha0_index, beta1_index],
        DNDR_TABLE[alpha1_index, beta1_index],
    )


def aerodynamic_coefficients(
    alpha_deg,
    beta_deg,
    elevator_deg,
    aileron_deg,
    rudder_deg,
    p,
    q,
    r,
    true_airspeed,
    cg_fraction=reference_cg_fraction,
):
    """
    Assemble the total nondimensional F-16 aerodynamic coefficients.

    Parameters
    ----------
    alpha_deg : float
        Angle of attack in degrees.
    beta_deg : float
        Sideslip angle in degrees.
    elevator_deg : float
        Elevator deflection in degrees.
    aileron_deg : float
        Aileron deflection in degrees.
    rudder_deg : float
        Rudder deflection in degrees.
    p : float
        Body roll rate in radians per second.
    q : float
        Body pitch rate in radians per second.
    r : float
        Body yaw rate in radians per second.
    true_airspeed : float
        True airspeed in meters per second.
    cg_fraction : float, optional
        Center-of-gravity position as a fraction of mean aerodynamic chord.
        The default is the Figure 3.5-2 reference CG fraction.

    Returns
    -------
    numpy.ndarray
        Nondimensional body-axis coefficients in the order
        ``[CX, CY, CZ, Cl, Cm, Cn]``.

    Raises
    ------
    ValueError
        If ``true_airspeed`` is not positive.
    """
    if true_airspeed <= 0.0:
        raise ValueError("true_airspeed must be positive")

    dail = aileron_deg / 20.0
    drdr = rudder_deg / 30.0

    cxt = cx(alpha_deg, elevator_deg)
    cyt = cy(beta_deg, aileron_deg, rudder_deg)
    czt = cz(alpha_deg, beta_deg, elevator_deg)
    clt = (
        cl(alpha_deg, beta_deg)
        + dlda(alpha_deg, beta_deg) * dail
        + dldr(alpha_deg, beta_deg) * drdr
    )
    cmt = cm(alpha_deg, elevator_deg)
    cnt = (
        cn(alpha_deg, beta_deg)
        + dnda(alpha_deg, beta_deg) * dail
        + dndr(alpha_deg, beta_deg) * drdr
    )

    d = damp(alpha_deg)
    b2v = span / (2.0 * true_airspeed)
    cq = mean_aerodynamic_chord * q / (2.0 * true_airspeed)

    cxt = cxt + cq * d[0]
    cyt = cyt + b2v * (d[1] * r + d[2] * p)
    czt = czt + cq * d[3]
    clt = clt + b2v * (d[4] * r + d[5] * p)
    cmt = cmt + cq * d[6] + czt * (reference_cg_fraction - cg_fraction)
    cnt = (
        cnt
        + b2v * (d[7] * r + d[8] * p)
        - cyt
        * (reference_cg_fraction - cg_fraction)
        * mean_aerodynamic_chord
        / span
    )

    return np.array([cxt, cyt, czt, clt, cmt, cnt])


def aerodynamic_loads(coefficients, air_density, true_airspeed):
    """
    Convert aerodynamic coefficients to dimensional body-axis loads.

    The coefficient and load signs follow the forward-right-down (FRD)
    body-axis convention.

    Parameters
    ----------
    coefficients : array_like
        Nondimensional coefficients ordered as ``[CX, CY, CZ, Cl, Cm, Cn]``.
    air_density : float
        Air density in kilograms per cubic meter.
    true_airspeed : float
        True airspeed in meters per second.

    Returns
    -------
    forces_body : numpy.ndarray
        Body-axis aerodynamic forces ``[X, Y, Z]`` in newtons, with shape
        ``(3,)``.
    moments_body : numpy.ndarray
        Body-axis aerodynamic moments ``[L, M, N]`` in newton-meters, with
        shape ``(3,)``.

    Raises
    ------
    ValueError
        If ``coefficients`` does not have shape ``(6,)``, or if
        ``air_density`` or ``true_airspeed`` is not positive.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (6,):
        raise ValueError("coefficients must have shape (6,)")
    if air_density <= 0.0:
        raise ValueError("air_density must be positive")
    if true_airspeed <= 0.0:
        raise ValueError("true_airspeed must be positive")

    cx_value, cy_value, cz_value, cl_value, cm_value, cn_value = coefficients
    dynamic_pressure = 0.5 * air_density * true_airspeed**2
    qs = dynamic_pressure * wing_area

    forces_body = np.array(
        [qs * cx_value, qs * cy_value, qs * cz_value]
    )
    moments_body = np.array(
        [
            qs * span * cl_value,
            qs * mean_aerodynamic_chord * cm_value,
            qs * span * cn_value,
        ]
    )

    return forces_body, moments_body


def f16_aerodynamic_loads(
    velocity_body,
    omega_body,
    elevator_deg,
    aileron_deg,
    rudder_deg,
    air_density,
    cg_fraction=reference_cg_fraction,
):
    """
    Compute complete F-16 aerodynamic loads from flight state and controls.

    Inputs and outputs use the forward-right-down (FRD) body-axis convention.
    Gravity and engine thrust are not included.

    Parameters
    ----------
    velocity_body : array_like
        Body-frame velocity ``[u, v, w]`` in meters per second.
    omega_body : array_like
        Body angular rates ``[p, q, r]`` in radians per second.
    elevator_deg : float
        Elevator deflection in degrees.
    aileron_deg : float
        Aileron deflection in degrees.
    rudder_deg : float
        Rudder deflection in degrees.
    air_density : float
        Air density in kilograms per cubic meter.
    cg_fraction : float, optional
        Center-of-gravity position as a fraction of mean aerodynamic chord.
        The default is the Figure 3.5-2 reference CG fraction.

    Returns
    -------
    forces_body : numpy.ndarray
        FRD body-axis aerodynamic forces ``[X, Y, Z]`` in newtons.
    moments_body : numpy.ndarray
        FRD body-axis aerodynamic moments ``[L, M, N]`` in newton-meters.

    Raises
    ------
    ValueError
        If ``omega_body`` does not have shape ``(3,)``. Lower-level input
        validation errors are propagated unchanged.
    """
    omega_body = np.asarray(omega_body, dtype=float)
    if omega_body.shape != (3,):
        raise ValueError("omega_body must have shape (3,)")

    true_airspeed, alpha_deg, beta_deg = air_data_from_body_velocity(
        velocity_body
    )
    p, q, r = omega_body

    coefficients = aerodynamic_coefficients(
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        elevator_deg=elevator_deg,
        aileron_deg=aileron_deg,
        rudder_deg=rudder_deg,
        p=p,
        q=q,
        r=r,
        true_airspeed=true_airspeed,
        cg_fraction=cg_fraction,
    )
    forces_body, moments_body = aerodynamic_loads(
        coefficients=coefficients,
        air_density=air_density,
        true_airspeed=true_airspeed,
    )

    return forces_body, moments_body
