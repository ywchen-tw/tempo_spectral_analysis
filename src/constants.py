"""Physical constants used by the TEMPO O2-O2 workflow.

This module centralizes unit conversions and a few atmospheric constants so the
WP1-WP4 code stays readable and the unit handling stays consistent.
"""

GRAVITY_M_S2 = 9.80665
RD_DRY_AIR_J_KG_K = 287.05
BOLTZMANN_J_K = 1.380649e-23
X_O2_DRY_AIR = 0.2095
X_N2_DRY_AIR = 0.7808
X_N2O_DRY_AIR = 3.2e-7   # ~320 ppb climatological

M_DRY_AIR_G_MOL = 28.97
M_H2O_G_MOL = 18.015

# Unit conversions used throughout the atmosphere/profile and tau calculations.
M_TO_CM = 100.0
M3_TO_CM3 = 1.0e6
PA_TO_HPA = 0.01
