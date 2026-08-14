"""Physical parameters for the F-16 configuration in Figure 3.5-2.

Dimensional runtime quantities are expressed in SI units and derived from
the US customary values in Stevens, Lewis & Johnson Figure 3.5-2.
"""

import numpy as np


# Exact unit conversions used to reproduce the Figure 3.5-2 configuration.
LBF_TO_NEWTON = 4.4482216152605
FT_TO_METER = 0.3048
SLUG_TO_KILOGRAM = 14.593902937206363
STANDARD_GRAVITY = 9.80665

# Figure 3.5-2 explicitly prints WEIGHT = 25000 lbf.
figure_3_5_2_weight_lbf = 25000.0
# The nominal simulation weight reproduces the published longitudinal trim
# data in Stevens, Lewis & Johnson Table 3.6-2.
weight_lbf = 20490.0
Jxx_slug_ft2 = 9496.0
Jyy_slug_ft2 = 55814.0
Jzz_slug_ft2 = 63100.0
Jxz_slug_ft2 = 982.0
span_ft = 30.0
wing_area_ft2 = 300.0
mean_aerodynamic_chord_ft = 11.32
# Figure 3.5-2 uses engine angular momentum HX = 160 slug*ft^2/s.
engine_angular_momentum_slug_ft2_s = 160.0

weight_newtons = weight_lbf * LBF_TO_NEWTON
mass = weight_newtons / STANDARD_GRAVITY

inertia_conversion = SLUG_TO_KILOGRAM * FT_TO_METER**2
Jxx = Jxx_slug_ft2 * inertia_conversion
Jyy = Jyy_slug_ft2 * inertia_conversion
Jzz = Jzz_slug_ft2 * inertia_conversion
Jxz = Jxz_slug_ft2 * inertia_conversion
engine_angular_momentum = (
    engine_angular_momentum_slug_ft2_s
    * SLUG_TO_KILOGRAM
    * FT_TO_METER**2
)

span = span_ft * FT_TO_METER
wing_area = wing_area_ft2 * FT_TO_METER**2
mean_aerodynamic_chord = mean_aerodynamic_chord_ft * FT_TO_METER

# XCGR is a fraction of mean aerodynamic chord in the aerodynamic model.
reference_cg_fraction = 0.35
reference_cg = reference_cg_fraction * mean_aerodynamic_chord


# Inertia tensor about the body axes using the forward-right-down (FRD)
# convention. The x-z products of inertia therefore appear as -Jxz.
J = np.array([
    [Jxx, 0.0, -Jxz],
    [0.0, Jyy, 0.0],
    [-Jxz, 0.0, Jzz],
])
