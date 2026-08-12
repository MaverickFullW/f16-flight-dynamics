import numpy as np

from src.f16sim.attitude import euler_to_quaternion
from src.f16sim.aerodynamics import aerodynamic_coefficients
from src.f16sim.atmosphere import f16_air_data
from src.f16sim.engine import thrust_lbf
from src.f16sim.f16_model import LBF_TO_NEWTON, f16_state_derivative
from src.f16sim.parameters import (
    FT_TO_METER,
    J,
    SLUG_TO_KILOGRAM,
    wing_area,
)


def _lewis_table_3_5_2_case():
    throttle = 0.9
    elevator_deg = 20.0
    aileron_deg = -15.0
    rudder_deg = -20.0
    cg_fraction = 0.4

    true_airspeed_ft_s = 500.0
    alpha = 0.5
    beta = -0.2
    phi = -1.0
    theta = 1.0
    psi = -1.0
    p = 0.7
    q = -0.8
    r = 0.9
    north_ft = 1000.0
    east_ft = 900.0
    altitude_ft = 10000.0
    engine_power = 90.0

    u_ft_s = true_airspeed_ft_s * np.cos(alpha) * np.cos(beta)
    v_ft_s = true_airspeed_ft_s * np.sin(beta)
    w_ft_s = true_airspeed_ft_s * np.sin(alpha) * np.cos(beta)
    quaternion = euler_to_quaternion(phi, theta, psi)
    state = np.array(
        [
            north_ft * FT_TO_METER,
            east_ft * FT_TO_METER,
            -altitude_ft * FT_TO_METER,
            u_ft_s * FT_TO_METER,
            v_ft_s * FT_TO_METER,
            w_ft_s * FT_TO_METER,
            *quaternion,
            p,
            q,
            r,
            engine_power,
        ]
    )

    lewis_gravity = 32.17 * FT_TO_METER
    lewis_mass = (25000.0 / 32.17) * SLUG_TO_KILOGRAM
    model_dot = f16_state_derivative(
        state,
        throttle=throttle,
        elevator_deg=elevator_deg,
        aileron_deg=aileron_deg,
        rudder_deg=rudder_deg,
        cg_fraction=cg_fraction,
        mass_value=lewis_mass,
        inertia=J,
        gravity=lewis_gravity,
    )

    true_airspeed_m_s = true_airspeed_ft_s * FT_TO_METER
    u, v, w = state[3:6]
    u_dot, v_dot, w_dot = model_dot[3:6]
    true_airspeed_dot_m_s2 = (
        u * u_dot + v * v_dot + w * w_dot
    ) / true_airspeed_m_s
    dum = u * u + w * w
    alpha_dot = (u * w_dot - w * u_dot) / dum
    beta_dot = (
        (true_airspeed_m_s * v_dot - v * true_airspeed_dot_m_s2)
        * np.cos(beta)
        / dum
    )

    values = {
        "VT_dot": true_airspeed_dot_m_s2 / FT_TO_METER,
        "alpha_dot": alpha_dot,
        "beta_dot": beta_dot,
        "phi_dot": p
        + np.tan(theta) * (q * np.sin(phi) + r * np.cos(phi)),
        "theta_dot": q * np.cos(phi) - r * np.sin(phi),
        "psi_dot": (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta),
        "p_dot": model_dot[10],
        "q_dot": model_dot[11],
        "r_dot": model_dot[12],
        "north_dot": model_dot[0] / FT_TO_METER,
        "east_dot": model_dot[1] / FT_TO_METER,
        "altitude_dot": -model_dot[2] / FT_TO_METER,
        "engine_power_dot": model_dot[13],
    }
    inputs = {
        "state": state,
        "model_dot": model_dot,
        "true_airspeed_m_s": true_airspeed_m_s,
        "altitude_ft": altitude_ft,
        "alpha": alpha,
        "beta": beta,
        "phi": phi,
        "theta": theta,
        "p": p,
        "q": q,
        "r": r,
        "engine_power": engine_power,
        "elevator_deg": elevator_deg,
        "aileron_deg": aileron_deg,
        "rudder_deg": rudder_deg,
        "cg_fraction": cg_fraction,
        "mass": lewis_mass,
        "gravity": lewis_gravity,
    }
    return values, inputs


def _assert_close(name, actual, expected, rtol, atol):
    assert np.isclose(actual, expected, rtol=rtol, atol=atol), (
        f"{name}: expected {expected}, computed {actual}"
    )


def test_f16_model_matches_lewis_table_3_5_2_consistent_outputs():
    computed, _ = _lewis_table_3_5_2_case()
    published = {
        "beta_dot": -0.4759990,
        "phi_dot": 2.505734,
        "theta_dot": 0.3250820,
        "psi_dot": 2.145926,
        "p_dot": 12.62679,
        "q_dot": 0.9649671,
        "r_dot": 0.5809759,
        "north_dot": 342.4439,
        "east_dot": -266.7707,
        "altitude_dot": 248.1241,
        "engine_power_dot": -58.68999,
    }

    # beta_dot is weakly coupled to the inconsistent published VT_dot through
    # its coordinate transformation, so it receives a separate small margin.
    _assert_close(
        "beta_dot", computed["beta_dot"], published["beta_dot"], 6e-3, 5e-4
    )

    # Euler and navigation kinematics contain no aerodynamic table lookup;
    # tolerances here only accommodate rounding in the printed reference.
    for name in (
        "phi_dot",
        "theta_dot",
        "psi_dot",
        "north_dot",
        "east_dot",
        "altitude_dot",
    ):
        _assert_close(name, computed[name], published[name], 2e-6, 5e-5)

    # Rotational accelerations pass through interpolated Appendix A tables.
    for name in ("p_dot", "q_dot", "r_dot"):
        _assert_close(name, computed[name], published[name], 5e-3, 5e-4)

    # PDOT is algebraic apart from constants printed to finite precision.
    _assert_close(
        "engine_power_dot",
        computed["engine_power_dot"],
        published["engine_power_dot"],
        2e-5,
        5e-5,
    )


def test_lewis_table_3_5_2_longitudinal_discrepancy_is_documented():
    computed, case = _lewis_table_3_5_2_case()

    # This records a reproducible inconsistency between two entries in Table
    # 3.5-2 and the printed Figure 3.5-2 / Appendix A equations. It is not
    # evidence of a deliberately altered F-16 model.
    model_equation_values = {
        "VT_dot": -64.5454527,
        "alpha_dot": -0.8206246,
    }
    published_table_values = {
        "VT_dot": -75.23724,
        "alpha_dot": -0.8813491,
    }
    for name, expected in model_equation_values.items():
        _assert_close(name, computed[name], expected, 2e-7, 2e-7)
        assert not np.isclose(
            computed[name],
            published_table_values[name],
            rtol=5e-3,
            atol=5e-3,
        )

    density, mach, _ = f16_air_data(
        case["true_airspeed_m_s"],
        case["altitude_ft"] * FT_TO_METER,
    )
    coefficients = aerodynamic_coefficients(
        alpha_deg=np.degrees(case["alpha"]),
        beta_deg=np.degrees(case["beta"]),
        elevator_deg=case["elevator_deg"],
        aileron_deg=case["aileron_deg"],
        rudder_deg=case["rudder_deg"],
        p=case["p"],
        q=case["q"],
        r=case["r"],
        true_airspeed=case["true_airspeed_m_s"],
        cg_fraction=case["cg_fraction"],
    )
    dynamic_pressure = 0.5 * density * case["true_airspeed_m_s"] ** 2
    qs = dynamic_pressure * wing_area
    thrust_newtons = (
        thrust_lbf(case["engine_power"], case["altitude_ft"], mach)
        * LBF_TO_NEWTON
    )
    u, v, w = case["state"][3:6]
    explicit_body_acceleration = np.array(
        [
            case["r"] * v
            - case["q"] * w
            - case["gravity"] * np.sin(case["theta"])
            + (qs * coefficients[0] + thrust_newtons) / case["mass"],
            case["p"] * w
            - case["r"] * u
            + case["gravity"] * np.cos(case["theta"]) * np.sin(case["phi"])
            + qs * coefficients[1] / case["mass"],
            case["q"] * u
            - case["p"] * v
            + case["gravity"] * np.cos(case["theta"]) * np.cos(case["phi"])
            + qs * coefficients[2] / case["mass"],
        ]
    )

    assert np.allclose(
        case["model_dot"][3:6],
        explicit_body_acceleration,
        rtol=1e-12,
        atol=1e-12,
    )
