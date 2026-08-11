"""F-16 physical parameters in SI units."""

import numpy as np


# Stevens, Lewis & Johnson, Aircraft Control and Simulation, Appendix A,
# tabulates the baseline aircraft data in US customary units. The source
# values are: weight = 20,500 lbf; Jxx = 9,450 slug*ft^2;
# Jyy = 55,814 slug*ft^2; Jzz = 63,100 slug*ft^2;
# Jxz = 982 slug*ft^2; span = 30 ft; wing area = 300 ft^2;
# mean aerodynamic chord = 11.32 ft; and reference CG = 3.962 ft.
# All runtime quantities below are stored in SI units.

mass = 9298.643585  # kg

Jxx = 12812.479612  # kg*m^2
Jyy = 75673.622968  # kg*m^2
Jzz = 85552.112540  # kg*m^2
Jxz = 1331.413225  # kg*m^2

span = 9.144  # m
wing_area = 27.870912  # m^2
mean_aerodynamic_chord = 3.450336  # m
reference_cg = 1.2076176  # m


# Inertia tensor about the body axes using the forward-right-down (FRD)
# convention. The x-z products of inertia therefore appear as -Jxz.
J = np.array([
    [Jxx, 0.0, -Jxz],
    [0.0, Jyy, 0.0],
    [-Jxz, 0.0, Jzz],
])
