import numpy as np
import pytest

from src.f16sim.parameters import (
    J,
    mass,
    mean_aerodynamic_chord,
    reference_cg,
    span,
    wing_area,
)


def test_inertia_matrix_is_symmetric():
    assert np.allclose(J, J.T)


def test_inertia_matrix_is_invertible():
    assert np.linalg.matrix_rank(J) == J.shape[0]


def test_inertia_matrix_is_positive_definite():
    eigenvalues = np.linalg.eigvalsh(J)

    assert np.all(eigenvalues > 0.0)


def test_inertia_matrix_has_expected_shape():
    assert J.shape == (3, 3)


@pytest.mark.parametrize(
    "parameter",
    [mass, span, wing_area, mean_aerodynamic_chord, reference_cg],
)
def test_physical_parameters_are_positive(parameter):
    assert parameter > 0.0
