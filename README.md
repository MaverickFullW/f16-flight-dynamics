# F-16 Flight Dynamics — Nonlinear 6DoF Simulation and Control

This repository implements a Python-based F-16 flight-dynamics simulation and analysis environment: a nonlinear rigid-body model, straight-and-level trim, numerical linearization, dynamic-mode analysis, feedback-control design, local linear/nonlinear validation, and 3D flight visualization. The aircraft motion in the final animation is generated from the integrated nonlinear states and feedback-controller commands—not from a predefined animation path.

Final nonlinear flight demonstrations:

<div align="center">
  <img src="media/f16_descending_360_turn_readme.gif" alt="F-16 descending 360-degree turn" width="44%">
  &emsp;&emsp;&emsp;
  <img src="media/f16_flight_demo_readme.gif" alt="F-16 nonlinear 6DoF flight demonstration" width="44%">
</div>

## Flight Demo

The 110 s demonstration starts from a 502 ft/s sea-level trim and proceeds through level flight, a climbing turn, an opposite-direction S-turn, recovery, a commanded axial roll, a U-turn, a curved exit, another recovery, a final turn, and a straight departure. Position and attitude are produced by the integrated nonlinear state history. The low-poly aircraft mesh is a visual reference, not aerodynamic geometry.

## Engineering Workflow

```text
Nonlinear 6DoF Model → Trim → Linearization → Dynamic Modes
        → Control Design → Nonlinear Simulation → Validation
        → Flight Visualization
```

## Key Features

- Nonlinear rigid-body translational and rotational dynamics in SI units
- North-East-Down (NED) navigation frame and forward-right-down (FRD) body frame
- Scalar-first quaternion attitude propagation with normalization during RK4 integration
- Reduced F-16 aerodynamic tables, damping/control derivatives, and CG corrections
- Altitude/Mach-dependent thrust tables and a dynamic engine-power state
- Simplified source-model atmosphere and air-data calculations
- Symmetric straight-and-level trim by numerical optimization
- Centered finite-difference longitudinal and lateral-directional linearization
- Short-period, phugoid, Dutch-roll, roll-subsidence, and spiral-mode analysis
- Pitch-rate, pitch-attitude, yaw-rate, roll-rate, and bank-angle feedback analysis
- Nonlinear open-loop, closed-loop, and augmented-controller simulation
- Automated physics, source-reference, regression, controller, export, and visualization tests
- Matplotlib 3D trajectory and aircraft animation

## Nonlinear Equations of Motion

The state is

```math
\mathbf{x}
=
\begin{bmatrix}
N & E & D & u & v & w & q_0 & q_1 & q_2 & q_3 & p & q & r & P_e
\end{bmatrix}^{\mathsf T}.
```

Its principal vector components are

```math
\mathbf r_n
=
\begin{bmatrix}N\\E\\D\end{bmatrix},
\qquad
\mathbf v_b
=
\begin{bmatrix}u\\v\\w\end{bmatrix},
\qquad
\mathbf q
=
\begin{bmatrix}q_0\\q_1\\q_2\\q_3\end{bmatrix},
\qquad
\boldsymbol\omega_b
=
\begin{bmatrix}p\\q\\r\end{bmatrix}.
```

Here, $\mathbf r_n$ is NED position, $\mathbf v_b$ is body-FRD velocity, $\mathbf q$ is a scalar-first unit quaternion, $\boldsymbol\omega_b$ is body angular velocity, and $P_e$ is engine power. The quaternion and $\mathbf C_{n\rightarrow b}$ describe the NED-to-body rotation. The implemented rigid-body equations are

```math
\dot{\mathbf r}_n
=
\mathbf C_{n\rightarrow b}^{\mathsf T}\mathbf v_b.
```

```math
\dot{\mathbf v}_b
=
\frac{\mathbf F_b}{m}
+\mathbf C_{n\rightarrow b}
\begin{bmatrix}
0\\
0\\
g
\end{bmatrix}
-\boldsymbol\omega_b\times\mathbf v_b.
```

```math
\dot{\mathbf q}
=
\frac{1}{2}\,\boldsymbol\Omega\!\left(\boldsymbol\omega_b\right)\mathbf q.
```

```math
\dot{\boldsymbol\omega}_b
=
\mathbf J^{-1}
\left[
\mathbf M_b
-\boldsymbol\omega_b\times
\left(\mathbf J\boldsymbol\omega_b+\mathbf h_e\right)
\right].
```

Aerodynamic force and thrust are included in $\mathbf F_b$; gravity is added separately. The constant engine rotor angular momentum $\mathbf h_e$ lies along body $+x$. The control vector is

```math
\boldsymbol\delta
=
\begin{bmatrix}
\delta_T & \delta_e & \delta_a & \delta_r
\end{bmatrix}^{\mathsf T},
```

where $\delta_T$ is dimensionless throttle and $\delta_e$, $\delta_a$, and $\delta_r$ are elevator, aileron, and rudder angles in degrees.

The nonlinear plant is a rigid-aircraft, free-flight model; control-surface actuator dynamics, structural flexibility, aeroelasticity, and ground interaction are not modeled.

## F-16 Aerodynamics, Propulsion, and Atmosphere

The aerodynamics follow the reduced F-16 model documented in Stevens, Lewis, and Johnson. The body-axis aerodynamic-coefficient vector

```math
\mathbf C_A
=
\begin{bmatrix}
C_X & C_Y & C_Z & C_l & C_m & C_n
\end{bmatrix}^{\mathsf T}
```

combines tabulated or analytic base coefficients with nondimensional $p$, $q$, and $r$ damping terms, aileron/rudder derivatives, and pitching/yawing moment corrections for CG position. Tables are linearly or bilinearly interpolated; out-of-grid values are extrapolated from the nearest edge cell rather than clamped.

Dimensional loads use dynamic pressure, wing area, span, and mean aerodynamic chord. Propulsion interpolates idle, military, and maximum thrust versus altitude and Mach. Throttle maps to commanded power, while the fourteenth state supplies a piecewise engine response; thrust acts along body $+x$. Model validity is bounded by the reduced aerodynamic and propulsion tables. The source-model atmosphere supplies simplified density, temperature, Mach, and dynamic pressure without wind, turbulence, or weather.

## Straight-and-Level Trim

`trim_straight_level` solves for

```math
\mathbf z_{\mathrm{trim}}
=
\begin{bmatrix}
\delta_T & \delta_e & \alpha
\end{bmatrix}^{\mathsf T}
```

at specified airspeed, altitude, and CG. The straight-and-level constraints are

```math
\beta=\phi=\psi=p=q=r=0,
\qquad
\theta=\alpha,
```

with engine power initialized consistently with the throttle command. Nelder-Mead minimizes the weighted objective

```math
J_{\mathrm{trim}}
=
\left(\frac{\dot V_T}{1\ \mathrm{ft/s^2}}\right)^2
+100\dot\alpha^2
+10\dot q^2,
```

formed from the physical residuals

```math
\dot V_T,
\qquad
\dot\alpha,
\qquad
\dot q,
```

and accepts a solution only when

```math
\left|\dot V_T\right|<10^{-4}\ \mathrm{m/s^2},
\qquad
\left|\dot\alpha\right|<10^{-4}\ \mathrm{rad/s},
\qquad
\left|\dot q\right|<10^{-4}\ \mathrm{rad/s^2}.
```

The analysis and demo use 502 ft/s at sea level with CG at 0.30 mean aerodynamic chord. The resulting reference trim is approximately

```math
\delta_T=0.1485,
\qquad
\delta_e=-1.931^\circ,
\qquad
\alpha=2.255^\circ.
```

This is a symmetric straight-and-level solver, not a general turning-flight trim solver.

## Numerical Linearization

Centered finite differences produce

```math
\delta\dot{\mathbf x}
=
\mathbf A\,\delta\mathbf x
+\mathbf B\,\delta\mathbf u
```

about a trim point:

| Model | Perturbation states | Perturbation controls |
|---|---|---|
| Longitudinal | Vₜ, α, θ, q | throttle, δₑ |
| Lateral-directional | β, φ, p, r | δₐ, δᵣ |

Both reduced derivative functions map their variables into the full nonlinear 14-state model before extracting the corresponding rates.

## Aircraft Dynamic Modes

Eigenvalues identify the longitudinal short-period and phugoid pairs and the lateral Dutch-roll, roll-subsidence, and spiral modes. For a complex-conjugate pole pair,

```math
\lambda
=
\sigma\pm\mathrm j\omega_d,
\qquad
\omega_n
=
\lvert\lambda\rvert
=
\sqrt{\sigma^2+\omega_d^2},
\qquad
\zeta
=
-\frac{\sigma}{\omega_n}.
```

The longitudinal implementation classifies its pairs by natural frequency. Lateral scripts use eigenvalues and eigenvector participation to distinguish Dutch roll, fast roll subsidence, and the slow spiral pole.

## Control Design

The final demo uses proportional cascades whose gains are analyzed with pole/root-locus and response scripts. They are design-point values for 502 ft/s at sea level, not universally optimal or gain-scheduled values. These augmentation laws are not a complete F-16 flight-control computer and do not provide control allocation, envelope protection, or failure logic.

```math
\begin{aligned}
\theta_{\mathrm{cmd}}
&\xrightarrow{\;K_\theta=0.5\;}
q_{\mathrm{cmd}}
\xrightarrow{\;K_q=5.0\;}
\text{pitch-rate feedback}
\longrightarrow \delta_e,
\end{aligned}
```

<br><br>

```math
\begin{aligned}
\phi_{\mathrm{cmd}}
&\xrightarrow{\;K_\phi=1.0\;}
p_{\mathrm{cmd}}
\xrightarrow{\;K_p=5.0\;}
\text{roll-rate feedback}
\longrightarrow \delta_a,
\end{aligned}
```

<br><br>

```math
\begin{aligned}
r
&\xrightarrow{\;K_r=50.0\;}
\text{yaw-rate feedback}
\longrightarrow \delta_r.
\end{aligned}
```

Using the model's FRD and control signs,

```math
\begin{aligned}
\Delta\delta_e &= K_q\left(q-q_{\mathrm{cmd}}\right),\\
\delta_a &= K_p\left(p-p_{\mathrm{cmd}}\right),\\
\delta_r &= K_r r.
\end{aligned}
```

The reusable controller contains the pitch cascade; lateral laws are explicit in the analysis and maneuver scripts. Separate tools sweep pitch-rate, pitch-attitude, pitch-attitude PI, yaw-damper, roll-rate, and bank-angle gains. PI pitch attitude is analyzed and simulated separately, but the final demo uses the proportional cascade above.

## Verification and Validation

Reduced linear and full nonlinear models are initialized at the same trim and subjected to identical perturbations or feedback commands. Open-loop comparisons cover a 0.5° elevator pulse lasting 0.5 s and small initial sideslip and bank perturbations; closed-loop comparisons cover pitch and bank commands, together with lateral response with and without yaw damping. Agreement, and therefore the associated linear-model and control conclusions, is expected locally around the selected trim for small perturbations rather than across the full flight envelope.

The pytest suite covers vector/rotation utilities, quaternions, RK4 integration, rigid-body dynamics, air data and atmosphere, aerodynamic interpolation and loads, engine tables and dynamics, source-reference cases, trim, linearization, modal/control analysis, simulation, controllers, and visualization transforms. Some regression tests compare against printed F-16 cases while explicitly documenting discrepancies caused by internally inconsistent printed source values.

## Installation and Use

There is currently no packaging metadata or pinned dependency file. Create an environment with a recent Python 3 release and install the imported libraries:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install numpy scipy matplotlib pytest
```

Run tests and open the final Matplotlib demo:

```bash
python -m pytest -q
python scripts/animate_flight_demo.py
```

The demo supports `--camera {fixed,chase}`, `--fps`, `--speed`, `--aircraft-scale`, and `--vertical-exaggeration`. MP4 export additionally requires FFmpeg available to Matplotlib:

```bash
python scripts/animate_flight_demo.py --save-mp4 --output f16_flight_demo.mp4
```

## References

The flight-dynamics formulation, F-16 aerodynamic model, attitude
representation, and implementation choices in this study were guided
primarily by the following references.

1. B. L. Stevens, F. L. Lewis, and E. N. Johnson,
   *Aircraft Control and Simulation: Dynamics, Controls Design, and Autonomous
   Systems*, Wiley, 3rd edition, 2015.
   DOI: [10.1002/9781119174882](https://doi.org/10.1002/9781119174882)

2. F. L. Markley and J. L. Crassidis,
   *Fundamentals of Spacecraft Attitude Determination and Control*,
   Springer, 2014.
   DOI: [10.1007/978-1-4939-0802-8](https://doi.org/10.1007/978-1-4939-0802-8)

## Disclaimer

This is an independent educational and engineering portfolio project. It is not affiliated with or endorsed by Lockheed Martin or the U.S. Air Force. “F-16” is used descriptively for the modeled aircraft.
