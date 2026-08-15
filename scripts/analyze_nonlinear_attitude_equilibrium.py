"""Diagnose the nonlinear equilibrium behind the pitch-attitude PI response."""

from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.f16sim.linearization import longitudinal_state_derivative
from src.f16sim.parameters import FT_TO_METER
from src.f16sim.simulation import simulate_f16_feedback_augmented
from src.f16sim.trim import trim_straight_level


TRUE_AIRSPEED = 502.0 * FT_TO_METER
ALTITUDE_M = 0.0
CG_FRACTION = 0.30
KQ = 5.0
KTHETA = 0.5
KI = 0.05
COMMAND_INCREMENT = np.deg2rad(5.0)
DURATION = 200.0
DT = 0.01


def _pitch_angle(quaternion):
    quaternion = np.asarray(quaternion, dtype=float)
    quaternion = quaternion / np.linalg.norm(quaternion)
    q0, q1, q2, q3 = quaternion
    sin_theta = 2.0 * (q0 * q2 - q3 * q1)
    return np.arcsin(np.clip(sin_theta, -1.0, 1.0))


def _equilibrium_rates(alpha_deg, throttle, elevator_deg, theta_target):
    state = np.array(
        [
            TRUE_AIRSPEED,
            np.deg2rad(alpha_deg),
            theta_target,
            0.0,
        ]
    )
    controls = np.array([throttle, elevator_deg])
    derivative = longitudinal_state_derivative(
        state,
        controls,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )
    return derivative[[0, 1, 3]]


def _solve_target_equilibrium(trim, theta_target):
    def scaled_residual(variables):
        alpha_deg, throttle, elevator_deg = variables
        vt_dot, alpha_dot, q_dot = _equilibrium_rates(
            alpha_deg,
            throttle,
            elevator_deg,
            theta_target,
        )
        return np.array(
            [vt_dot / FT_TO_METER, 10.0 * alpha_dot, np.sqrt(10.0) * q_dot]
        )

    result = least_squares(
        scaled_residual,
        x0=np.array(
            [trim["alpha_deg"], trim["throttle"], trim["elevator_deg"]]
        ),
        bounds=(np.array([-20.0, 0.0, -25.0]), np.array([45.0, 1.0, 25.0])),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=4000,
    )
    alpha_deg, throttle, elevator_deg = result.x
    rates = _equilibrium_rates(
        alpha_deg,
        throttle,
        elevator_deg,
        theta_target,
    )
    valid = bool(
        result.success
        and abs(rates[0]) < 1e-4
        and abs(rates[1]) < 1e-4
        and abs(rates[2]) < 1e-4
    )
    return {
        "valid": valid,
        "message": result.message,
        "alpha_deg": float(alpha_deg),
        "throttle": float(throttle),
        "elevator_deg": float(elevator_deg),
        "rates": rates,
    }


def _classify_equilibrium(equilibrium, theta_target):
    if not equilibrium["valid"]:
        return "cannot satisfy the requested constraints"
    gamma_deg = np.rad2deg(theta_target) - equilibrium["alpha_deg"]
    if abs(gamma_deg) <= 0.05:
        return "level flight"
    if gamma_deg > 0.0:
        return "climb"
    return "descent"


def _simulate_pi_controller(trim, theta_target):
    def attitude_error(state):
        return theta_target - _pitch_angle(state[6:10])

    def control_law(time, state, controller_state):
        q_command = KTHETA * attitude_error(state) + KI * controller_state[0]
        return [
            trim["throttle"],
            trim["elevator_deg"] + KQ * (state[11] - q_command),
            0.0,
            0.0,
        ]

    def controller_derivative(time, state, controller_state):
        return np.array([attitude_error(state)])

    return simulate_f16_feedback_augmented(
        initial_state=trim["state"],
        initial_controller_state=np.zeros(1),
        duration=DURATION,
        dt=DT,
        control_law=control_law,
        controller_state_derivative=controller_derivative,
        cg_fraction=CG_FRACTION,
    )


def main():
    trim = trim_straight_level(
        true_airspeed=TRUE_AIRSPEED,
        altitude_m=ALTITUDE_M,
        cg_fraction=CG_FRACTION,
    )
    if not trim["success"]:
        raise RuntimeError(f"Unable to obtain the baseline trim: {trim['message']}")

    theta_trim = _pitch_angle(trim["state"][6:10])
    theta_target = theta_trim + COMMAND_INCREMENT
    q_trim = trim["state"][11]

    print("Baseline straight-and-level trim")
    print(f"  theta_trim: {np.rad2deg(theta_trim):.9f} deg")
    print(f"  alpha_trim: {trim['alpha_deg']:.9f} deg")
    print(f"  throttle_trim: {trim['throttle']:.9f}")
    print(f"  elevator_trim: {trim['elevator_deg']:.9f} deg")
    print(f"  q_trim: {np.rad2deg(q_trim):.9f} deg/s")
    print(f"  VT_trim: {trim['true_airspeed'] / FT_TO_METER:.9f} ft/s")
    print(f"Target theta: {np.rad2deg(theta_target):.9f} deg")

    equilibrium = _solve_target_equilibrium(trim, theta_target)
    gamma_deg = np.rad2deg(theta_target) - equilibrium["alpha_deg"]
    vt_dot, alpha_dot, q_dot = equilibrium["rates"]
    classification = _classify_equilibrium(equilibrium, theta_target)

    print("\nTarget-attitude local nonlinear equilibrium")
    print(f"  status: {classification}")
    if not equilibrium["valid"]:
        print(f"  optimizer message: {equilibrium['message']}")
    print(f"  equilibrium alpha: {equilibrium['alpha_deg']:.9f} deg")
    print(f"  equilibrium throttle: {equilibrium['throttle']:.9f}")
    print(f"  equilibrium elevator: {equilibrium['elevator_deg']:.9f} deg")
    print(f"  flight-path angle gamma: {gamma_deg:.9f} deg")
    print(f"  VT_dot: {vt_dot:.12e} m/s^2")
    print(f"  alpha_dot: {alpha_dot:.12e} rad/s")
    print(f"  q_dot: {q_dot:.12e} rad/s^2")
    if classification in ("climb", "descent"):
        print(
            "  note: this is a local steady-flight condition; altitude and "
            "atmospheric properties change along the flight path"
        )

    _, states, controller_states = _simulate_pi_controller(trim, theta_target)
    final_state = states[-1]
    final_xi = controller_states[-1, 0]
    final_theta = _pitch_angle(final_state[6:10])
    final_alpha = np.arctan2(final_state[5], final_state[3])
    final_q = final_state[11]
    final_q_command = (
        KTHETA * (theta_target - final_theta) + KI * final_xi
    )
    final_elevator = trim["elevator_deg"] + KQ * (
        final_q - final_q_command
    )

    print("\nFinal nonlinear PI simulation state")
    print(f"  theta: {np.rad2deg(final_theta):.9f} deg")
    print(f"  alpha: {np.rad2deg(final_alpha):.9f} deg")
    print(f"  q: {np.rad2deg(final_q):.9f} deg/s")
    print(f"  throttle: {trim['throttle']:.9f}")
    print(f"  elevator: {final_elevator:.9f} deg")

    print("\nFinal state minus calculated equilibrium")
    print(
        f"  theta difference: "
        f"{np.rad2deg(final_theta - theta_target):.9f} deg"
    )
    print(
        f"  alpha difference: "
        f"{np.rad2deg(final_alpha) - equilibrium['alpha_deg']:.9f} deg"
    )
    print(f"  q difference: {np.rad2deg(final_q):.9f} deg/s")
    print(f"  throttle difference: {trim['throttle'] - equilibrium['throttle']:.9f}")
    print(
        f"  elevator difference: "
        f"{final_elevator - equilibrium['elevator_deg']:.9f} deg"
    )


if __name__ == "__main__":
    main()
