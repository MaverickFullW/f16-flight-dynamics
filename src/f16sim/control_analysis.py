"""Control-analysis utilities for reduced F-16 state-space models."""

import numpy as np
from scipy.signal import ss2tf


_LONGITUDINAL_OUTPUTS = {
    "VT": 0,
    "alpha": 1,
    "theta": 2,
    "q": 3,
}
_LONGITUDINAL_INPUTS = {
    "throttle": 0,
    "elevator": 1,
}


def longitudinal_transfer_function(A, B, output, input_name):
    """Derive a SISO transfer function from the longitudinal state model.

    Parameters
    ----------
    A : array_like
        Longitudinal state matrix with shape ``(4, 4)``.
    B : array_like
        Longitudinal input matrix with shape ``(4, 2)``.
    output : {"VT", "alpha", "theta", "q"}
        State to use as the scalar output.
    input_name : {"throttle", "elevator"}
        Control input to use for the SISO transfer function.

    Returns
    -------
    dict
        Numerator and denominator polynomial coefficients, their zeros and
        poles, and the selected output and input names.

    Raises
    ------
    ValueError
        If ``A`` or ``B`` has the wrong shape, or a requested input or output
        is unsupported.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != (4, 4):
        raise ValueError("A must have shape (4, 4)")
    if B.shape != (4, 2):
        raise ValueError("B must have shape (4, 2)")
    if output not in _LONGITUDINAL_OUTPUTS:
        raise ValueError(f"unsupported longitudinal output: {output!r}")
    if input_name not in _LONGITUDINAL_INPUTS:
        raise ValueError(f"unsupported longitudinal input: {input_name!r}")

    C = np.zeros((1, 4), dtype=float)
    C[0, _LONGITUDINAL_OUTPUTS[output]] = 1.0
    D = np.zeros((1, 2), dtype=float)
    numerator_rows, denominator = ss2tf(
        A,
        B,
        C,
        D,
        input=_LONGITUDINAL_INPUTS[input_name],
    )
    numerator = numerator_rows[0]

    return {
        "numerator": numerator,
        "denominator": denominator,
        "poles": np.roots(denominator),
        "zeros": np.roots(numerator),
        "output": output,
        "input": input_name,
    }


def pitch_rate_feedback_poles(numerator, denominator, gains):
    """Return poles for positive pitch-rate feedback over a gain sequence.

    The F-16 elevator-to-pitch-rate plant has negative gain, so positive
    damping feedback follows the characteristic equation
    ``D(s) - Kq N(s) = 0``.

    Parameters
    ----------
    numerator : array_like
        Pitch-rate/elevator numerator coefficients in descending powers.
    denominator : array_like
        Pitch-rate/elevator denominator coefficients in descending powers.
    gains : array_like
        One-dimensional sequence of finite, nonnegative feedback gains.

    Returns
    -------
    numpy.ndarray
        Closed-loop poles with shape ``(number_of_gains, system_order)``.

    Raises
    ------
    ValueError
        If a coefficient or gain array is invalid, a gain is negative, or a
        closed-loop polynomial loses order.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    gains = np.asarray(gains, dtype=float)

    if numerator.ndim != 1 or numerator.size == 0:
        raise ValueError("numerator must be a nonempty one-dimensional array")
    if denominator.ndim != 1 or denominator.size < 2:
        raise ValueError(
            "denominator must be a one-dimensional polynomial of order at least one"
        )
    if numerator.size > denominator.size:
        raise ValueError("numerator degree cannot exceed denominator degree")
    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(denominator)):
        raise ValueError("polynomial coefficients must be finite")
    if denominator[0] == 0.0:
        raise ValueError("denominator leading coefficient must be nonzero")
    if gains.ndim != 1 or gains.size == 0:
        raise ValueError("gains must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(gains)) or np.any(gains < 0.0):
        raise ValueError("gains must be finite and nonnegative")

    padded_numerator = np.pad(
        numerator, (denominator.size - numerator.size, 0)
    )
    system_order = denominator.size - 1
    poles = np.empty((gains.size, system_order), dtype=complex)

    for index, gain in enumerate(gains):
        characteristic = denominator - gain * padded_numerator
        if characteristic[0] == 0.0:
            raise ValueError("closed-loop characteristic polynomial loses order")
        roots = np.roots(characteristic)
        if roots.shape != (system_order,) or not np.all(np.isfinite(roots)):
            raise ValueError("closed-loop poles must be finite and retain system order")
        poles[index] = np.sort_complex(roots)

    return poles


def pitch_attitude_feedback_poles(A, B, k_theta, Kq=5.0):
    """Return poles of the cascaded pitch-rate and pitch-attitude loops.

    The homogeneous closed-loop matrix is
    ``A + B_e Kq (C_q + Ktheta C_theta)``. Elevator is measured in degrees,
    pitch rate in radians per second, and ``Ktheta`` has units of inverse
    seconds.

    Parameters
    ----------
    A : array_like
        Longitudinal state matrix with shape ``(4, 4)``.
    B : array_like
        Longitudinal input matrix with shape ``(4, 2)``.
    k_theta : float or array_like
        One or more finite, nonnegative outer-loop attitude gains.
    Kq : float, optional
        Positive inner-loop pitch-rate gain in degrees per radian per second.

    Returns
    -------
    numpy.ndarray
        Closed-loop poles with shape ``(number_of_gains, 4)``.

    Raises
    ------
    ValueError
        If a matrix has the wrong shape or a gain is invalid.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != (4, 4):
        raise ValueError("A must have shape (4, 4)")
    if B.shape != (4, 2):
        raise ValueError("B must have shape (4, 2)")

    k_theta = np.asarray(k_theta, dtype=float)
    if k_theta.ndim == 0:
        k_theta = k_theta.reshape(1)
    if k_theta.ndim != 1 or k_theta.size == 0:
        raise ValueError("k_theta must be a scalar or nonempty one-dimensional array")
    if not np.all(np.isfinite(k_theta)) or np.any(k_theta < 0.0):
        raise ValueError("k_theta gains must be finite and nonnegative")

    Kq = np.asarray(Kq, dtype=float)
    if Kq.shape != () or not np.isfinite(Kq) or Kq <= 0.0:
        raise ValueError("Kq must be a finite positive scalar")
    Kq = float(Kq)

    elevator_column = B[:, 1:2]
    theta_output = np.array([[0.0, 0.0, 1.0, 0.0]])
    pitch_rate_output = np.array([[0.0, 0.0, 0.0, 1.0]])
    poles = np.empty((k_theta.size, 4), dtype=complex)

    for index, attitude_gain in enumerate(k_theta):
        feedback_output = pitch_rate_output + attitude_gain * theta_output
        closed_loop_matrix = A + Kq * (elevator_column @ feedback_output)
        roots = np.linalg.eigvals(closed_loop_matrix)
        if not np.all(np.isfinite(roots)):
            raise ValueError("closed-loop poles must be finite")
        poles[index] = np.sort_complex(roots)

    return poles


def pitch_attitude_pi_feedback_poles(
    A,
    B,
    ki,
    Kq=5.0,
    Ktheta=0.5,
):
    """Return poles of the augmented pitch-attitude PI architecture.

    The augmented state is ``[VT, alpha, theta, q, xi]``, with homogeneous
    integrator dynamics ``xi_dot = -theta``. The matrix is constructed
    directly from the plant and controller equations.

    Parameters
    ----------
    A : array_like
        Longitudinal state matrix with shape ``(4, 4)``.
    B : array_like
        Longitudinal input matrix with shape ``(4, 2)``.
    ki : float or array_like
        One or more finite, nonnegative outer-loop integral gains in
        inverse-seconds squared.
    Kq : float, optional
        Positive inner pitch-rate gain in degrees per radian per second.
    Ktheta : float, optional
        Nonnegative outer proportional attitude gain in inverse seconds.

    Returns
    -------
    numpy.ndarray
        Augmented closed-loop poles with shape ``(number_of_gains, 5)``.

    Raises
    ------
    ValueError
        If a matrix has the wrong shape or a gain is invalid.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != (4, 4):
        raise ValueError("A must have shape (4, 4)")
    if B.shape != (4, 2):
        raise ValueError("B must have shape (4, 2)")

    ki = np.asarray(ki, dtype=float)
    if ki.ndim == 0:
        ki = ki.reshape(1)
    if ki.ndim != 1 or ki.size == 0:
        raise ValueError("ki must be a scalar or nonempty one-dimensional array")
    if not np.all(np.isfinite(ki)) or np.any(ki < 0.0):
        raise ValueError("ki gains must be finite and nonnegative")

    Kq = np.asarray(Kq, dtype=float)
    if Kq.shape != () or not np.isfinite(Kq) or Kq <= 0.0:
        raise ValueError("Kq must be a finite positive scalar")
    Kq = float(Kq)
    Ktheta = np.asarray(Ktheta, dtype=float)
    if Ktheta.shape != () or not np.isfinite(Ktheta) or Ktheta < 0.0:
        raise ValueError("Ktheta must be a finite nonnegative scalar")
    Ktheta = float(Ktheta)

    elevator_column = B[:, 1:2]
    theta_output = np.array([[0.0, 0.0, 1.0, 0.0]])
    pitch_rate_output = np.array([[0.0, 0.0, 0.0, 1.0]])
    proportional_output = pitch_rate_output + Ktheta * theta_output
    proportional_matrix = A + Kq * (elevator_column @ proportional_output)

    poles = np.empty((ki.size, 5), dtype=complex)
    for index, integral_gain in enumerate(ki):
        augmented_matrix = np.zeros((5, 5), dtype=float)
        augmented_matrix[:4, :4] = proportional_matrix
        augmented_matrix[:4, 4:5] = -Kq * integral_gain * elevator_column
        augmented_matrix[4:5, :4] = -theta_output
        roots = np.linalg.eigvals(augmented_matrix)
        if not np.all(np.isfinite(roots)):
            raise ValueError("augmented closed-loop poles must be finite")
        poles[index] = np.sort_complex(roots)

    return poles


def lateral_roll_rate_feedback_poles(A_lat, B_lat, k_p, Kr=50.0):
    """Return lateral poles with fixed yaw damping and roll-rate feedback."""
    A_lat = np.asarray(A_lat, dtype=float)
    B_lat = np.asarray(B_lat, dtype=float)
    if A_lat.shape != (4, 4):
        raise ValueError("A_lat must have shape (4, 4)")
    if B_lat.shape != (4, 2):
        raise ValueError("B_lat must have shape (4, 2)")

    k_p = np.asarray(k_p, dtype=float)
    if k_p.ndim == 0:
        k_p = k_p.reshape(1)
    if k_p.ndim != 1 or k_p.size == 0 or not np.all(np.isfinite(k_p)):
        raise ValueError("k_p must be a finite scalar or one-dimensional array")
    Kr = np.asarray(Kr, dtype=float)
    if Kr.shape != () or not np.isfinite(Kr):
        raise ValueError("Kr must be a finite scalar")

    aileron_column = B_lat[:, 0:1]
    rudder_column = B_lat[:, 1:2]
    roll_rate_output = np.array([[0.0, 0.0, 1.0, 0.0]])
    yaw_rate_output = np.array([[0.0, 0.0, 0.0, 1.0]])
    yaw_damped_matrix = A_lat + float(Kr) * (rudder_column @ yaw_rate_output)
    poles = np.empty((k_p.size, 4), dtype=complex)
    for index, gain in enumerate(k_p):
        matrix = yaw_damped_matrix + gain * (
            aileron_column @ roll_rate_output
        )
        poles[index] = np.sort_complex(np.linalg.eigvals(matrix))
    return poles


def lateral_bank_angle_feedback_poles(
    A_lat,
    B_lat,
    k_phi,
    Kp,
    Kr=50.0,
):
    """Return poles of cascaded bank-angle/roll-rate and yaw-rate feedback."""
    A_lat = np.asarray(A_lat, dtype=float)
    B_lat = np.asarray(B_lat, dtype=float)
    if A_lat.shape != (4, 4):
        raise ValueError("A_lat must have shape (4, 4)")
    if B_lat.shape != (4, 2):
        raise ValueError("B_lat must have shape (4, 2)")

    k_phi = np.asarray(k_phi, dtype=float)
    if k_phi.ndim == 0:
        k_phi = k_phi.reshape(1)
    if k_phi.ndim != 1 or k_phi.size == 0:
        raise ValueError("k_phi must be a scalar or nonempty one-dimensional array")
    if not np.all(np.isfinite(k_phi)) or np.any(k_phi < 0.0):
        raise ValueError("k_phi gains must be finite and nonnegative")
    Kp = np.asarray(Kp, dtype=float)
    Kr = np.asarray(Kr, dtype=float)
    if Kp.shape != () or not np.isfinite(Kp):
        raise ValueError("Kp must be a finite scalar")
    if Kr.shape != () or not np.isfinite(Kr):
        raise ValueError("Kr must be a finite scalar")

    aileron_column = B_lat[:, 0:1]
    rudder_column = B_lat[:, 1:2]
    bank_output = np.array([[0.0, 1.0, 0.0, 0.0]])
    roll_rate_output = np.array([[0.0, 0.0, 1.0, 0.0]])
    yaw_rate_output = np.array([[0.0, 0.0, 0.0, 1.0]])
    yaw_damped_matrix = A_lat + float(Kr) * (rudder_column @ yaw_rate_output)
    poles = np.empty((k_phi.size, 4), dtype=complex)
    for index, gain in enumerate(k_phi):
        feedback_output = roll_rate_output + gain * bank_output
        matrix = yaw_damped_matrix + float(Kp) * (
            aileron_column @ feedback_output
        )
        poles[index] = np.sort_complex(np.linalg.eigvals(matrix))
    return poles
