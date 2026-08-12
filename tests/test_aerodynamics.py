import numpy as np
import pytest

from src.f16sim.aerodynamics import (
    ALPHA_GRID_DEG,
    BETA_10_GRID_DEG,
    BETA_ABS_GRID_DEG,
    CL_TABLE,
    CM_TABLE,
    CN_TABLE,
    CX_TABLE,
    CZ_ALPHA_TABLE,
    DAMP_TABLE,
    DLDA_TABLE,
    DLDR_TABLE,
    DNDA_TABLE,
    DNDR_TABLE,
    ELEVATOR_GRID_DEG,
    aerodynamic_coefficients,
    cl,
    cm,
    cn,
    cx,
    cy,
    cz,
    damp,
    dlda,
    dldr,
    dnda,
    dndr,
)
from src.f16sim.parameters import (
    mean_aerodynamic_chord,
    reference_cg_fraction,
    span,
)


def test_cx_at_zero_alpha_and_zero_elevator():
    assert cx(alpha_deg=0.0, elevator_deg=0.0) == pytest.approx(-0.021)


def test_cx_at_fifteen_alpha_and_negative_twelve_elevator():
    assert cx(alpha_deg=15.0, elevator_deg=-12.0) == pytest.approx(0.083)


def test_cx_at_maximum_alpha_and_elevator():
    assert cx(alpha_deg=45.0, elevator_deg=24.0) == pytest.approx(0.040)


@pytest.mark.parametrize(
    ("alpha_index", "elevator_index"),
    [
        (alpha_index, elevator_index)
        for alpha_index in range(len(ALPHA_GRID_DEG))
        for elevator_index in range(len(ELEVATOR_GRID_DEG))
    ],
)
def test_cx_reproduces_every_table_value(alpha_index, elevator_index):
    actual = cx(
        alpha_deg=ALPHA_GRID_DEG[alpha_index],
        elevator_deg=ELEVATOR_GRID_DEG[elevator_index],
    )

    assert actual == pytest.approx(CX_TABLE[alpha_index, elevator_index])


def test_cx_bilinear_interpolation_inside_cell():
    assert cx(alpha_deg=2.5, elevator_deg=6.0) == pytest.approx(-0.02225)


def test_cx_interpolation_only_in_alpha():
    assert cx(alpha_deg=2.5, elevator_deg=0.0) == pytest.approx(-0.0125)


def test_cx_interpolation_only_in_elevator():
    assert cx(alpha_deg=0.0, elevator_deg=6.0) == pytest.approx(-0.0300)


def test_cx_extrapolates_below_alpha_table_without_clamping():
    extrapolated = cx(alpha_deg=-15.0, elevator_deg=0.0)
    boundary = cx(alpha_deg=ALPHA_GRID_DEG[0], elevator_deg=0.0)

    assert extrapolated == pytest.approx(-0.024)
    assert not np.isclose(extrapolated, boundary)


def test_cx_extrapolates_above_elevator_table_without_clamping():
    extrapolated = cx(alpha_deg=0.0, elevator_deg=30.0)
    boundary = cx(alpha_deg=0.0, elevator_deg=ELEVATOR_GRID_DEG[-1])

    assert extrapolated == pytest.approx(-0.0945)
    assert not np.isclose(extrapolated, boundary)


def test_cy_is_zero_for_zero_inputs():
    assert np.isclose(cy(0.0, 0.0, 0.0), 0.0)


def test_cy_for_positive_sideslip_and_zero_controls():
    assert np.isclose(cy(1.0, 0.0, 0.0), -0.02)


def test_cy_for_positive_aileron_and_other_inputs_zero():
    assert np.isclose(cy(0.0, 20.0, 0.0), 0.021)


def test_cy_for_positive_rudder_and_other_inputs_zero():
    assert np.isclose(cy(0.0, 0.0, 30.0), 0.086)


def test_cy_combined_case_matches_appendix_a_expression():
    beta_deg = 5.0
    aileron_deg = -10.0
    rudder_deg = 15.0
    expected = (
        -0.02 * beta_deg
        + 0.021 * (aileron_deg / 20.0)
        + 0.086 * (rudder_deg / 30.0)
    )

    assert np.isclose(cy(beta_deg, aileron_deg, rudder_deg), expected)


def test_cz_at_zero_alpha_with_zero_beta_and_elevator():
    assert np.isclose(cz(alpha_deg=0.0, beta_deg=0.0, elevator_deg=0.0), -0.100)


def test_cz_at_fifteen_alpha_with_zero_beta_and_elevator():
    assert np.isclose(
        cz(alpha_deg=15.0, beta_deg=0.0, elevator_deg=0.0), -1.053
    )


@pytest.mark.parametrize("alpha_index", range(len(ALPHA_GRID_DEG)))
def test_cz_reproduces_every_alpha_table_value(alpha_index):
    actual = cz(
        alpha_deg=ALPHA_GRID_DEG[alpha_index],
        beta_deg=0.0,
        elevator_deg=0.0,
    )

    assert np.isclose(actual, CZ_ALPHA_TABLE[alpha_index])


def test_cz_interpolates_in_alpha():
    expected = (-0.100 + -0.416) / 2.0

    assert np.isclose(cz(alpha_deg=2.5, beta_deg=0.0, elevator_deg=0.0), expected)


def test_cz_applies_elevator_correction():
    expected = -0.100 - 0.19

    assert np.isclose(cz(alpha_deg=0.0, beta_deg=0.0, elevator_deg=25.0), expected)


def test_cz_applies_sideslip_correction():
    expected = -0.100 * (1.0 - (10.0 / 57.3) ** 2)

    assert np.isclose(cz(alpha_deg=0.0, beta_deg=10.0, elevator_deg=0.0), expected)


def test_cz_combined_case_matches_appendix_a_expression():
    alpha_deg = 2.5
    beta_deg = 10.0
    elevator_deg = -12.5
    cz_alpha = (-0.100 + -0.416) / 2.0
    expected = cz_alpha * (1.0 - (beta_deg / 57.3) ** 2) - 0.19 * (
        elevator_deg / 25.0
    )

    assert np.isclose(cz(alpha_deg, beta_deg, elevator_deg), expected)


def test_cz_extrapolates_below_alpha_grid_without_clamping():
    extrapolated = cz(alpha_deg=-15.0, beta_deg=0.0, elevator_deg=0.0)
    boundary = cz(alpha_deg=ALPHA_GRID_DEG[0], beta_deg=0.0, elevator_deg=0.0)
    expected = 0.770 + (0.770 - 0.241)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


def test_cz_extrapolates_above_alpha_grid_without_clamping():
    extrapolated = cz(alpha_deg=50.0, beta_deg=0.0, elevator_deg=0.0)
    boundary = cz(alpha_deg=ALPHA_GRID_DEG[-1], beta_deg=0.0, elevator_deg=0.0)
    expected = -2.229 + (-2.229 - -2.248)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


def test_cm_at_zero_alpha_and_elevator():
    assert np.isclose(cm(alpha_deg=0.0, elevator_deg=0.0), -0.009)


def test_cm_at_fifteen_alpha_and_negative_twelve_elevator():
    assert np.isclose(cm(alpha_deg=15.0, elevator_deg=-12.0), 0.141)


def test_cm_at_maximum_alpha_and_elevator():
    assert np.isclose(cm(alpha_deg=45.0, elevator_deg=24.0), -0.005)


@pytest.mark.parametrize(
    ("alpha_index", "elevator_index"),
    [
        (alpha_index, elevator_index)
        for alpha_index in range(len(ALPHA_GRID_DEG))
        for elevator_index in range(len(ELEVATOR_GRID_DEG))
    ],
)
def test_cm_reproduces_every_table_value(alpha_index, elevator_index):
    actual = cm(
        alpha_deg=ALPHA_GRID_DEG[alpha_index],
        elevator_deg=ELEVATOR_GRID_DEG[elevator_index],
    )

    assert np.isclose(actual, CM_TABLE[alpha_index, elevator_index])


def test_cm_bilinear_interpolation_inside_cell():
    expected = (-0.009 + -0.005 + -0.121 + -0.127) / 4.0

    assert np.isclose(cm(alpha_deg=2.5, elevator_deg=6.0), expected)


def test_cm_interpolation_only_in_alpha():
    expected = (-0.009 + -0.005) / 2.0

    assert np.isclose(cm(alpha_deg=2.5, elevator_deg=0.0), expected)


def test_cm_interpolation_only_in_elevator():
    expected = (-0.009 + -0.121) / 2.0

    assert np.isclose(cm(alpha_deg=0.0, elevator_deg=6.0), expected)


def test_cm_extrapolates_below_alpha_grid_without_clamping():
    extrapolated = cm(alpha_deg=-15.0, elevator_deg=0.0)
    boundary = cm(alpha_deg=ALPHA_GRID_DEG[0], elevator_deg=0.0)
    expected = -0.046 + (-0.046 - -0.020)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


def test_cm_extrapolates_above_elevator_grid_without_clamping():
    extrapolated = cm(alpha_deg=0.0, elevator_deg=30.0)
    boundary = cm(alpha_deg=0.0, elevator_deg=ELEVATOR_GRID_DEG[-1])
    expected = -0.184 + 0.5 * (-0.184 - -0.121)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


def test_cl_at_zero_sideslip_returns_exactly_zero():
    assert cl(alpha_deg=0.0, beta_deg=0.0) == 0.0


def test_cl_at_positive_beta_table_point():
    assert np.isclose(cl(alpha_deg=0.0, beta_deg=10.0), -0.017)


def test_cl_at_negative_beta_restores_odd_symmetry():
    assert np.isclose(cl(alpha_deg=0.0, beta_deg=-10.0), 0.017)


@pytest.mark.parametrize("alpha_deg", [-7.5, 2.5, 17.5, 42.5])
@pytest.mark.parametrize("beta_deg", [2.5, 7.5, 17.5, 27.5])
def test_cl_is_odd_in_nonzero_sideslip(alpha_deg, beta_deg):
    assert np.isclose(cl(alpha_deg, -beta_deg), -cl(alpha_deg, beta_deg))


@pytest.mark.parametrize(
    ("alpha_index", "beta_index"),
    [
        (alpha_index, beta_index)
        for alpha_index in range(len(ALPHA_GRID_DEG))
        for beta_index in range(1, len(BETA_ABS_GRID_DEG))
    ],
)
def test_cl_reproduces_every_positive_beta_table_value(alpha_index, beta_index):
    actual = cl(
        alpha_deg=ALPHA_GRID_DEG[alpha_index],
        beta_deg=BETA_ABS_GRID_DEG[beta_index],
    )

    assert np.isclose(actual, CL_TABLE[alpha_index, beta_index])


def test_cl_bilinear_interpolation_inside_cell():
    expected = (-0.008 + -0.012 + -0.017 + -0.024) / 4.0

    assert np.isclose(cl(alpha_deg=2.5, beta_deg=7.5), expected)


def test_cl_interpolation_only_in_alpha():
    expected = (-0.017 + -0.024) / 2.0

    assert np.isclose(cl(alpha_deg=2.5, beta_deg=10.0), expected)


def test_cl_interpolation_only_in_absolute_beta():
    expected = (-0.008 + -0.017) / 2.0

    assert np.isclose(cl(alpha_deg=0.0, beta_deg=7.5), expected)


def test_cl_extrapolates_beyond_positive_beta_grid_without_clamping():
    extrapolated = cl(alpha_deg=30.0, beta_deg=35.0)
    boundary = cl(alpha_deg=30.0, beta_deg=BETA_ABS_GRID_DEG[-1])
    expected = -0.091 + (-0.091 - -0.060)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


@pytest.mark.parametrize(
    ("alpha_deg", "expected"),
    [(-15.0, 0.003), (50.0, -0.028)],
)
def test_cl_extrapolates_beyond_alpha_grid_without_clamping(alpha_deg, expected):
    extrapolated = cl(alpha_deg=alpha_deg, beta_deg=10.0)
    boundary_alpha = ALPHA_GRID_DEG[0] if alpha_deg < 0.0 else ALPHA_GRID_DEG[-1]
    boundary = cl(alpha_deg=boundary_alpha, beta_deg=10.0)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


def test_cn_at_zero_sideslip_returns_exactly_zero():
    assert cn(alpha_deg=0.0, beta_deg=0.0) == 0.0


def test_cn_at_positive_beta_table_point():
    assert np.isclose(cn(alpha_deg=0.0, beta_deg=10.0), 0.042)


def test_cn_at_negative_beta_restores_odd_symmetry():
    assert np.isclose(cn(alpha_deg=0.0, beta_deg=-10.0), -0.042)


@pytest.mark.parametrize("alpha_deg", [-7.5, 2.5, 17.5, 42.5])
@pytest.mark.parametrize("beta_deg", [2.5, 7.5, 17.5, 27.5])
def test_cn_is_odd_in_nonzero_sideslip(alpha_deg, beta_deg):
    assert np.isclose(cn(alpha_deg, -beta_deg), -cn(alpha_deg, beta_deg))


@pytest.mark.parametrize(
    ("alpha_index", "beta_index"),
    [
        (alpha_index, beta_index)
        for alpha_index in range(len(ALPHA_GRID_DEG))
        for beta_index in range(1, len(BETA_ABS_GRID_DEG))
    ],
)
def test_cn_reproduces_every_positive_beta_table_value(alpha_index, beta_index):
    actual = cn(
        alpha_deg=ALPHA_GRID_DEG[alpha_index],
        beta_deg=BETA_ABS_GRID_DEG[beta_index],
    )

    assert np.isclose(actual, CN_TABLE[alpha_index, beta_index])


def test_cn_bilinear_interpolation_inside_cell():
    expected = (0.018 + 0.019 + 0.042 + 0.042) / 4.0

    assert np.isclose(cn(alpha_deg=2.5, beta_deg=7.5), expected)


def test_cn_interpolation_only_in_alpha():
    expected = (0.043 + 0.039) / 2.0

    assert np.isclose(cn(alpha_deg=12.5, beta_deg=10.0), expected)


def test_cn_interpolation_only_in_absolute_beta():
    expected = (0.018 + 0.042) / 2.0

    assert np.isclose(cn(alpha_deg=0.0, beta_deg=7.5), expected)


def test_cn_extrapolates_beyond_positive_beta_grid_without_clamping():
    extrapolated = cn(alpha_deg=0.0, beta_deg=35.0)
    boundary = cn(alpha_deg=0.0, beta_deg=BETA_ABS_GRID_DEG[-1])
    expected = 0.106 + (0.106 - 0.093)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


@pytest.mark.parametrize(
    ("alpha_deg", "expected"),
    [(-15.0, 0.034), (50.0, -0.067)],
)
def test_cn_extrapolates_beyond_alpha_grid_without_clamping(alpha_deg, expected):
    extrapolated = cn(alpha_deg=alpha_deg, beta_deg=10.0)
    boundary_alpha = ALPHA_GRID_DEG[0] if alpha_deg < 0.0 else ALPHA_GRID_DEG[-1]
    boundary = cn(alpha_deg=boundary_alpha, beta_deg=10.0)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


def test_damp_reproduces_first_table_row():
    assert np.allclose(damp(alpha_deg=-10.0), DAMP_TABLE[0])


def test_damp_reproduces_second_table_row_with_expected_cmq():
    expected = DAMP_TABLE[1].copy()
    expected[6] = -0.540

    assert np.allclose(damp(alpha_deg=-5.0), expected)


def test_damp_reproduces_last_table_row():
    assert np.allclose(damp(alpha_deg=45.0), DAMP_TABLE[-1])


@pytest.mark.parametrize("alpha_index", range(len(ALPHA_GRID_DEG)))
def test_damp_reproduces_every_table_row(alpha_index):
    assert np.allclose(
        damp(alpha_deg=ALPHA_GRID_DEG[alpha_index]),
        DAMP_TABLE[alpha_index],
    )


def test_damp_interpolates_all_derivatives_at_alpha_midpoint():
    alpha0_index = 2
    alpha1_index = 3
    expected = (DAMP_TABLE[alpha0_index] + DAMP_TABLE[alpha1_index]) / 2.0

    assert np.allclose(damp(alpha_deg=2.5), expected)


def test_damp_extrapolates_below_alpha_grid_without_clamping():
    extrapolated = damp(alpha_deg=-15.0)
    expected = DAMP_TABLE[0] + (DAMP_TABLE[0] - DAMP_TABLE[1])

    assert np.allclose(extrapolated, expected)
    assert not np.allclose(extrapolated, DAMP_TABLE[0])


def test_damp_extrapolates_above_alpha_grid_without_clamping():
    extrapolated = damp(alpha_deg=50.0)
    expected = DAMP_TABLE[-1] + (DAMP_TABLE[-1] - DAMP_TABLE[-2])

    assert np.allclose(extrapolated, expected)
    assert not np.allclose(extrapolated, DAMP_TABLE[-1])


def test_damp_returns_nine_element_numpy_array():
    result = damp(alpha_deg=0.0)

    assert isinstance(result, np.ndarray)
    assert result.shape == (9,)


def test_dlda_at_minimum_alpha_and_beta():
    assert np.isclose(dlda(alpha_deg=-10.0, beta_deg=-30.0), -0.041)


def test_dlda_at_zero_alpha_and_beta():
    assert np.isclose(dlda(alpha_deg=0.0, beta_deg=0.0), -0.051)


def test_dlda_at_maximum_alpha_and_beta():
    assert np.isclose(dlda(alpha_deg=45.0, beta_deg=30.0), -0.008)


@pytest.mark.parametrize(
    ("alpha_index", "beta_index"),
    [
        (alpha_index, beta_index)
        for alpha_index in range(len(ALPHA_GRID_DEG))
        for beta_index in range(len(BETA_10_GRID_DEG))
    ],
)
def test_dlda_reproduces_every_table_value(alpha_index, beta_index):
    actual = dlda(
        alpha_deg=ALPHA_GRID_DEG[alpha_index],
        beta_deg=BETA_10_GRID_DEG[beta_index],
    )

    assert np.isclose(actual, DLDA_TABLE[alpha_index, beta_index])


def test_dlda_bilinear_interpolation_inside_cell():
    expected = (-0.051 + -0.052 + -0.048 + -0.049) / 4.0

    assert np.isclose(dlda(alpha_deg=2.5, beta_deg=5.0), expected)


def test_dlda_interpolation_only_in_alpha():
    expected = (-0.051 + -0.052) / 2.0

    assert np.isclose(dlda(alpha_deg=2.5, beta_deg=0.0), expected)


def test_dlda_interpolation_only_in_beta():
    expected = (-0.051 + -0.048) / 2.0

    assert np.isclose(dlda(alpha_deg=0.0, beta_deg=5.0), expected)


@pytest.mark.parametrize(
    ("alpha_deg", "expected"),
    [(-15.0, -0.030), (50.0, -0.007)],
)
def test_dlda_extrapolates_beyond_alpha_grid_without_clamping(alpha_deg, expected):
    extrapolated = dlda(alpha_deg=alpha_deg, beta_deg=-30.0)
    boundary_alpha = ALPHA_GRID_DEG[0] if alpha_deg < 0.0 else ALPHA_GRID_DEG[-1]
    boundary = dlda(alpha_deg=boundary_alpha, beta_deg=-30.0)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


@pytest.mark.parametrize(
    ("beta_deg", "expected"),
    [(-40.0, -0.098), (40.0, 0.014)],
)
def test_dlda_extrapolates_beyond_beta_grid_without_clamping(beta_deg, expected):
    extrapolated = dlda(alpha_deg=20.0, beta_deg=beta_deg)
    boundary_beta = BETA_10_GRID_DEG[0] if beta_deg < 0.0 else BETA_10_GRID_DEG[-1]
    boundary = dlda(alpha_deg=20.0, beta_deg=boundary_beta)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


def test_dldr_at_minimum_alpha_and_beta():
    assert np.isclose(dldr(alpha_deg=-10.0, beta_deg=-30.0), 0.005)


def test_dldr_at_zero_alpha_and_beta():
    assert np.isclose(dldr(alpha_deg=0.0, beta_deg=0.0), 0.015)


def test_dldr_at_maximum_alpha_and_beta():
    assert np.isclose(dldr(alpha_deg=45.0, beta_deg=30.0), 0.000)


@pytest.mark.parametrize(
    ("alpha_index", "beta_index"),
    [
        (alpha_index, beta_index)
        for alpha_index in range(len(ALPHA_GRID_DEG))
        for beta_index in range(len(BETA_10_GRID_DEG))
    ],
)
def test_dldr_reproduces_every_table_value(alpha_index, beta_index):
    actual = dldr(
        alpha_deg=ALPHA_GRID_DEG[alpha_index],
        beta_deg=BETA_10_GRID_DEG[beta_index],
    )

    assert np.isclose(actual, DLDR_TABLE[alpha_index, beta_index])


def test_dldr_bilinear_interpolation_inside_cell():
    expected = (0.015 + 0.014 + 0.013 + 0.013) / 4.0

    assert np.isclose(dldr(alpha_deg=2.5, beta_deg=5.0), expected)


def test_dldr_interpolation_only_in_alpha():
    expected = (0.015 + 0.014) / 2.0

    assert np.isclose(dldr(alpha_deg=2.5, beta_deg=0.0), expected)


def test_dldr_interpolation_only_in_beta():
    expected = (0.015 + 0.013) / 2.0

    assert np.isclose(dldr(alpha_deg=0.0, beta_deg=5.0), expected)


@pytest.mark.parametrize(
    ("alpha_deg", "expected"),
    [(-15.0, -0.007), (50.0, 0.027)],
)
def test_dldr_extrapolates_beyond_alpha_grid_without_clamping(alpha_deg, expected):
    extrapolated = dldr(alpha_deg=alpha_deg, beta_deg=-30.0)
    boundary_alpha = ALPHA_GRID_DEG[0] if alpha_deg < 0.0 else ALPHA_GRID_DEG[-1]
    boundary = dldr(alpha_deg=boundary_alpha, beta_deg=-30.0)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


@pytest.mark.parametrize(
    ("beta_deg", "expected"),
    [(-40.0, 0.003), (40.0, 0.025)],
)
def test_dldr_extrapolates_beyond_beta_grid_without_clamping(beta_deg, expected):
    extrapolated = dldr(alpha_deg=-10.0, beta_deg=beta_deg)
    boundary_beta = BETA_10_GRID_DEG[0] if beta_deg < 0.0 else BETA_10_GRID_DEG[-1]
    boundary = dldr(alpha_deg=-10.0, beta_deg=boundary_beta)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


def test_dnda_at_minimum_alpha_and_beta():
    assert np.isclose(dnda(alpha_deg=-10.0, beta_deg=-30.0), 0.001)


def test_dnda_at_zero_alpha_and_beta():
    assert np.isclose(dnda(alpha_deg=0.0, beta_deg=0.0), -0.010)


def test_dnda_at_maximum_alpha_and_beta():
    assert np.isclose(dnda(alpha_deg=45.0, beta_deg=30.0), 0.001)


@pytest.mark.parametrize(
    ("alpha_index", "beta_index"),
    [
        (alpha_index, beta_index)
        for alpha_index in range(len(ALPHA_GRID_DEG))
        for beta_index in range(len(BETA_10_GRID_DEG))
    ],
)
def test_dnda_reproduces_every_table_value(alpha_index, beta_index):
    actual = dnda(
        alpha_deg=ALPHA_GRID_DEG[alpha_index],
        beta_deg=BETA_10_GRID_DEG[beta_index],
    )

    assert np.isclose(actual, DNDA_TABLE[alpha_index, beta_index])


def test_dnda_bilinear_interpolation_inside_cell():
    expected = (-0.010 + -0.009 + -0.014 + -0.012) / 4.0

    assert np.isclose(dnda(alpha_deg=2.5, beta_deg=5.0), expected)


def test_dnda_interpolation_only_in_alpha():
    expected = (-0.010 + -0.009) / 2.0

    assert np.isclose(dnda(alpha_deg=2.5, beta_deg=0.0), expected)


def test_dnda_interpolation_only_in_beta():
    expected = (-0.010 + -0.014) / 2.0

    assert np.isclose(dnda(alpha_deg=0.0, beta_deg=5.0), expected)


@pytest.mark.parametrize(
    ("alpha_deg", "expected"),
    [(-15.0, 0.029), (50.0, 0.024)],
)
def test_dnda_extrapolates_beyond_alpha_grid_without_clamping(alpha_deg, expected):
    extrapolated = dnda(alpha_deg=alpha_deg, beta_deg=-30.0)
    boundary_alpha = ALPHA_GRID_DEG[0] if alpha_deg < 0.0 else ALPHA_GRID_DEG[-1]
    boundary = dnda(alpha_deg=boundary_alpha, beta_deg=-30.0)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


@pytest.mark.parametrize(
    ("beta_deg", "expected"),
    [(-40.0, 0.000), (40.0, -0.020)],
)
def test_dnda_extrapolates_beyond_beta_grid_without_clamping(beta_deg, expected):
    extrapolated = dnda(alpha_deg=-10.0, beta_deg=beta_deg)
    boundary_beta = BETA_10_GRID_DEG[0] if beta_deg < 0.0 else BETA_10_GRID_DEG[-1]
    boundary = dnda(alpha_deg=-10.0, beta_deg=boundary_beta)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


def test_dndr_at_minimum_alpha_and_beta():
    assert np.isclose(dndr(alpha_deg=-10.0, beta_deg=-30.0), -0.018)


def test_dndr_at_zero_alpha_and_beta():
    assert np.isclose(dndr(alpha_deg=0.0, beta_deg=0.0), -0.045)


def test_dndr_at_maximum_alpha_and_beta():
    assert np.isclose(dndr(alpha_deg=45.0, beta_deg=30.0), -0.010)


@pytest.mark.parametrize(
    ("alpha_index", "beta_index"),
    [
        (alpha_index, beta_index)
        for alpha_index in range(len(ALPHA_GRID_DEG))
        for beta_index in range(len(BETA_10_GRID_DEG))
    ],
)
def test_dndr_reproduces_every_table_value(alpha_index, beta_index):
    actual = dndr(
        alpha_deg=ALPHA_GRID_DEG[alpha_index],
        beta_deg=BETA_10_GRID_DEG[beta_index],
    )

    assert np.isclose(actual, DNDR_TABLE[alpha_index, beta_index])


def test_dndr_bilinear_interpolation_inside_cell():
    expected = (-0.045 + -0.045 + -0.041 + -0.041) / 4.0

    assert np.isclose(dndr(alpha_deg=2.5, beta_deg=5.0), expected)


def test_dndr_interpolation_only_in_alpha():
    expected = (-0.044 + -0.045) / 2.0

    assert np.isclose(dndr(alpha_deg=12.5, beta_deg=0.0), expected)


def test_dndr_interpolation_only_in_beta():
    expected = (-0.045 + -0.041) / 2.0

    assert np.isclose(dndr(alpha_deg=0.0, beta_deg=5.0), expected)


@pytest.mark.parametrize(
    ("alpha_deg", "expected"),
    [(-15.0, 0.016), (50.0, 0.000)],
)
def test_dndr_extrapolates_beyond_alpha_grid_without_clamping(alpha_deg, expected):
    extrapolated = dndr(alpha_deg=alpha_deg, beta_deg=-30.0)
    boundary_alpha = ALPHA_GRID_DEG[0] if alpha_deg < 0.0 else ALPHA_GRID_DEG[-1]
    boundary = dndr(alpha_deg=boundary_alpha, beta_deg=-30.0)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


@pytest.mark.parametrize(
    ("beta_deg", "expected"),
    [(-40.0, -0.008), (40.0, -0.072)],
)
def test_dndr_extrapolates_beyond_beta_grid_without_clamping(beta_deg, expected):
    extrapolated = dndr(alpha_deg=-10.0, beta_deg=beta_deg)
    boundary_beta = BETA_10_GRID_DEG[0] if beta_deg < 0.0 else BETA_10_GRID_DEG[-1]
    boundary = dndr(alpha_deg=-10.0, beta_deg=boundary_beta)

    assert np.isclose(extrapolated, expected)
    assert not np.isclose(extrapolated, boundary)


@pytest.mark.parametrize("true_airspeed", [0.0, -1.0])
def test_aerodynamic_coefficients_rejects_nonpositive_airspeed(true_airspeed):
    with pytest.raises(ValueError):
        aerodynamic_coefficients(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, true_airspeed)


def _base_control_coefficients(alpha, beta, elevator, aileron, rudder):
    return np.array(
        [
            cx(alpha, elevator),
            cy(beta, aileron, rudder),
            cz(alpha, beta, elevator),
            cl(alpha, beta)
            + dlda(alpha, beta) * (aileron / 20.0)
            + dldr(alpha, beta) * (rudder / 30.0),
            cm(alpha, elevator),
            cn(alpha, beta)
            + dnda(alpha, beta) * (aileron / 20.0)
            + dndr(alpha, beta) * (rudder / 30.0),
        ]
    )


def test_aerodynamic_coefficients_zero_rates_at_reference_cg():
    alpha, beta = 10.0, 5.0
    elevator, aileron, rudder = -6.0, 8.0, -9.0
    expected = _base_control_coefficients(alpha, beta, elevator, aileron, rudder)

    actual = aerodynamic_coefficients(
        alpha, beta, elevator, aileron, rudder, 0.0, 0.0, 0.0, 150.0
    )

    assert np.allclose(actual, expected)


def test_aerodynamic_coefficients_pitch_rate_corrections():
    alpha, beta, elevator = 10.0, 5.0, -6.0
    q, true_airspeed = 0.4, 150.0
    expected = _base_control_coefficients(alpha, beta, elevator, 0.0, 0.0)
    d = damp(alpha)
    cq = mean_aerodynamic_chord * q / (2.0 * true_airspeed)
    expected[[0, 2, 4]] += cq * d[[0, 3, 6]]

    actual = aerodynamic_coefficients(
        alpha, beta, elevator, 0.0, 0.0, 0.0, q, 0.0, true_airspeed
    )

    assert np.allclose(actual, expected)


def test_aerodynamic_coefficients_roll_rate_corrections():
    alpha, beta, elevator = 10.0, 5.0, -6.0
    p, true_airspeed = 0.3, 150.0
    expected = _base_control_coefficients(alpha, beta, elevator, 0.0, 0.0)
    d = damp(alpha)
    b2v = span / (2.0 * true_airspeed)
    expected[[1, 3, 5]] += b2v * p * d[[2, 5, 8]]

    actual = aerodynamic_coefficients(
        alpha, beta, elevator, 0.0, 0.0, p, 0.0, 0.0, true_airspeed
    )

    assert np.allclose(actual, expected)


def test_aerodynamic_coefficients_yaw_rate_corrections():
    alpha, beta, elevator = 10.0, 5.0, -6.0
    r, true_airspeed = -0.2, 150.0
    expected = _base_control_coefficients(alpha, beta, elevator, 0.0, 0.0)
    d = damp(alpha)
    b2v = span / (2.0 * true_airspeed)
    expected[[1, 3, 5]] += b2v * r * d[[1, 4, 7]]

    actual = aerodynamic_coefficients(
        alpha, beta, elevator, 0.0, 0.0, 0.0, 0.0, r, true_airspeed
    )

    assert np.allclose(actual, expected)


def test_aerodynamic_coefficients_applies_cg_shift():
    alpha, beta, elevator = 10.0, 5.0, -6.0
    aileron, rudder = 8.0, -9.0
    cg_fraction = 0.30
    expected = _base_control_coefficients(alpha, beta, elevator, aileron, rudder)
    cg_delta = reference_cg_fraction - cg_fraction
    expected[4] += expected[2] * cg_delta
    expected[5] -= expected[1] * cg_delta * mean_aerodynamic_chord / span

    actual = aerodynamic_coefficients(
        alpha,
        beta,
        elevator,
        aileron,
        rudder,
        0.0,
        0.0,
        0.0,
        150.0,
        cg_fraction,
    )

    assert np.allclose(actual, expected)


def test_aerodynamic_coefficients_returns_six_element_numpy_array():
    result = aerodynamic_coefficients(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0)

    assert isinstance(result, np.ndarray)
    assert result.shape == (6,)


def test_aerodynamic_coefficients_combined_case():
    alpha, beta, elevator = 12.5, -7.5, 4.0
    aileron, rudder = -8.0, 12.0
    p, q, r = 0.25, -0.35, 0.15
    true_airspeed = 175.0
    cg_fraction = 0.31
    expected = _base_control_coefficients(alpha, beta, elevator, aileron, rudder)
    d = damp(alpha)
    b2v = span / (2.0 * true_airspeed)
    cq = mean_aerodynamic_chord * q / (2.0 * true_airspeed)
    cg_delta = reference_cg_fraction - cg_fraction

    expected[0] += cq * d[0]
    expected[1] += b2v * (d[1] * r + d[2] * p)
    expected[2] += cq * d[3]
    expected[3] += b2v * (d[4] * r + d[5] * p)
    expected[4] += cq * d[6] + expected[2] * cg_delta
    expected[5] += b2v * (d[7] * r + d[8] * p)
    expected[5] -= expected[1] * cg_delta * mean_aerodynamic_chord / span

    actual = aerodynamic_coefficients(
        alpha,
        beta,
        elevator,
        aileron,
        rudder,
        p,
        q,
        r,
        true_airspeed,
        cg_fraction,
    )

    assert np.allclose(actual, expected)
