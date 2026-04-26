"""WP2: GEOS-CF collocation and atmospheric profile conversion.

This module turns a pixel table into layer-by-layer pressure, temperature,
moisture, and O2 number-density profiles for tau computation.

Pressure reconstruction note:
This module uses a fixed GEOS-72 hybrid-pressure edge table (Ap/Bp) together
with collocated surface pressure to reconstruct layer-edge and layer-mid
pressures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from constants import (
    BOLTZMANN_J_K,
    GRAVITY_M_S2,
    M_DRY_AIR_G_MOL,
    M_H2O_G_MOL,
    M_TO_CM,
    M3_TO_CM3,
    RD_DRY_AIR_J_KG_K,
    X_N2_DRY_AIR,
    X_N2O_DRY_AIR,
    X_O2_DRY_AIR,
)
from io_tempo import nearest_latlon_indices, read_geos_core


@dataclass
class ProfileConfig:
    """Configuration for how pressure is reconstructed in the profile step."""

    top_pressure_pa: float = 1.0


def _build_pressure_from_sigma(ps_pa: np.ndarray, nlev: int, top_pressure_pa: float) -> tuple:
    """Build layer-edge and layer-mid pressures from a simple sigma profile."""
    sigma_edges = np.linspace(0.0, 1.0, nlev + 1)
    sigma_mid = 0.5 * (sigma_edges[:-1] + sigma_edges[1:])

    p_mid = np.maximum(top_pressure_pa, ps_pa[:, None] * sigma_mid[None, :])
    p_top = np.maximum(top_pressure_pa, ps_pa[:, None] * sigma_edges[:-1][None, :])
    p_bot = np.maximum(top_pressure_pa, ps_pa[:, None] * sigma_edges[1:][None, :])

    return p_mid, p_top, p_bot


def _build_pressure_from_hybrid_apbp(ps_pa: np.ndarray, nlev: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build GEOS layer pressures from fixed Ap/Bp edge coefficients.

    Uses GEOS-72 hybrid grid edge coefficients (Ap in hPa, Bp unitless):
    P_edge(Pa) = Ap(hPa) * 100 + Bp * PS(Pa)

    Returns
    -------
    p_mid, p_top, p_bot : arrays with shape (npix, nlev)
        Layer mid, top-edge, and bottom-edge pressures in Pa.
    """
    ap_hpa = np.array(
        [
            0.000000e00, 4.804826e-02, 6.593752e00, 1.313480e01, 1.961311e01, 2.609201e01,
            3.257081e01, 3.898201e01, 4.533901e01, 5.169611e01, 5.805321e01, 6.436264e01,
            7.062198e01, 7.883422e01, 8.909992e01, 9.936521e01, 1.091817e02, 1.189586e02,
            1.286959e02, 1.429100e02, 1.562600e02, 1.696090e02, 1.816190e02, 1.930970e02,
            2.032590e02, 2.121500e02, 2.187760e02, 2.238980e02, 2.243630e02, 2.168650e02,
            2.011920e02, 1.769300e02, 1.503930e02, 1.278370e02, 1.086630e02, 9.236572e01,
            7.851231e01, 6.660341e01, 5.638791e01, 4.764391e01, 4.017541e01, 3.381001e01,
            2.836781e01, 2.373041e01, 1.979160e01, 1.645710e01, 1.364340e01, 1.127690e01,
            9.292942e00, 7.619842e00, 6.216801e00, 5.046801e00, 4.076571e00, 3.276431e00,
            2.620211e00, 2.084970e00, 1.650790e00, 1.300510e00, 1.019440e00, 7.951341e-01,
            6.167791e-01, 4.758061e-01, 3.650411e-01, 2.785261e-01, 2.113490e-01, 1.594950e-01,
            1.197030e-01, 8.934502e-02, 6.600001e-02, 4.758501e-02, 3.270000e-02, 2.000000e-02,
            1.000000e-02,
        ],
        dtype=float,
    )
    bp = np.array(
        [
            1.000000e00, 9.849520e-01, 9.634060e-01, 9.418650e-01, 9.203870e-01, 8.989080e-01,
            8.774290e-01, 8.560180e-01, 8.346609e-01, 8.133039e-01, 7.919469e-01, 7.706375e-01,
            7.493782e-01, 7.211660e-01, 6.858999e-01, 6.506349e-01, 6.158184e-01, 5.810415e-01,
            5.463042e-01, 4.945902e-01, 4.437402e-01, 3.928911e-01, 3.433811e-01, 2.944031e-01,
            2.467411e-01, 2.003501e-01, 1.562241e-01, 1.136021e-01, 6.372006e-02, 2.801004e-02,
            6.960025e-03, 8.175413e-09, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
            0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
            0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
            0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
            0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
            0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
            0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
            0.000000e00,
        ],
        dtype=float,
    )

    if ap_hpa.size != nlev + 1 or bp.size != nlev + 1:
        raise ValueError(
            f"Ap/Bp edge count ({ap_hpa.size}) does not match requested nlev+1 ({nlev + 1})."
        )

    p_edge = ap_hpa[np.newaxis, :] * 100.0 + bp[np.newaxis, :] * ps_pa[:, np.newaxis]

    # The Ap/Bp edge table above is ordered surface -> TOA, while GEOS-CF
    # level arrays (T, Q, species) in this file are indexed TOA -> surface.
    # Reverse layer order so pressure aligns with profile indexing.
    p_bot = p_edge[:, :-1][:, ::-1]
    p_top = p_edge[:, 1:][:, ::-1]
    p_mid = 0.5 * (p_bot + p_top)
    return p_mid, p_top, p_bot


def build_profiles_for_pixels(
    pixel_df: pd.DataFrame,
    geos_file: str,
    config: ProfileConfig | None = None,
) -> dict:
    """Collocate each pixel to GEOS-CF and derive P/T/Q/nO2/dz profiles."""
    cfg = config or ProfileConfig()
    geos = read_geos_core(geos_file)

    valid = pixel_df["valid_pixel"].to_numpy(dtype=bool)
    lat = pixel_df["latitude"].to_numpy(dtype=float)
    lon = pixel_df["longitude"].to_numpy(dtype=float)

    ilat, ilon = nearest_latlon_indices(geos["lat"], geos["lon"], lat, lon)

    ps = geos["PS"][0, ilat, ilon]
    t_raw = geos["T"][0, :, ilat, ilon]
    q_raw = geos["Q"][0, :, ilat, ilon]

    nlev_ref = int(len(geos["lev"]))
    if t_raw.shape[0] == nlev_ref:
        t = t_raw.T
        q = q_raw.T
    elif t_raw.shape[-1] == nlev_ref:
        t = t_raw
        q = q_raw
    else:
        raise ValueError(f"Unexpected GEOS profile shape: {t_raw.shape}")

    npix, nlev = t.shape
    try:
        p_mid, p_top, p_bot = _build_pressure_from_hybrid_apbp(ps, nlev)
    except ValueError:
        # Fallback keeps the pipeline robust if a non-GEOS-72 file is supplied.
        p_mid, p_top, p_bot = _build_pressure_from_sigma(ps, nlev, cfg.top_pressure_pa)

    tv = t * (1.0 + 0.61 * q)
    # Use the hydrostatic thickness relation to convert the temperature field
    # into a layer thickness that can be combined with the O2-O2 cross section.
    dz_m = (RD_DRY_AIR_J_KG_K * tv / GRAVITY_M_S2) * np.log(p_bot / p_top)

    n_air_cm3 = (p_mid / (BOLTZMANN_J_K * t)) / M3_TO_CM3
    dry_frac = 1.0 - q
    n_o2_cm3  = X_O2_DRY_AIR  * dry_frac * n_air_cm3
    n_n2_cm3  = X_N2_DRY_AIR  * dry_frac * n_air_cm3
    n_n2o_cm3 = X_N2O_DRY_AIR * n_air_cm3
    n_h2o_cm3 = q * (M_DRY_AIR_G_MOL / M_H2O_G_MOL) * n_air_cm3
    dz_cm = dz_m * M_TO_CM

    # NO2 and O3 from GEOS-CF (mol/mol VMR collocated to each pixel).
    no2_raw = geos["NO2"][0, :, ilat, ilon]
    o3_raw  = geos["O3"][0,  :, ilat, ilon]
    if no2_raw.shape[0] == nlev_ref:
        no2 = no2_raw.T
        o3  = o3_raw.T
    else:
        no2 = no2_raw
        o3  = o3_raw
    n_no2_cm3 = no2 * n_air_cm3
    n_o3_cm3  = o3  * n_air_cm3

    # Invalidate profiles for failed pixels to keep downstream masking explicit.
    for arr in (n_air_cm3, n_o2_cm3, n_n2_cm3, n_n2o_cm3, n_h2o_cm3,
                n_no2_cm3, n_o3_cm3, dz_cm):
        arr[~valid, :] = np.nan

    return {
        "lev": geos["lev"],
        "geos_ilat": ilat,
        "geos_ilon": ilon,
        "ps_pa": ps,
        "p_mid_pa": p_mid,
        "t_k": t,
        "q_kgkg": q,
        "n_air_cm3": n_air_cm3,
        "n_o2_cm3": n_o2_cm3,
        "n_n2_cm3": n_n2_cm3,
        "n_n2o_cm3": n_n2o_cm3,
        "n_h2o_cm3": n_h2o_cm3,
        "n_no2_cm3": n_no2_cm3,
        "n_o3_cm3": n_o3_cm3,
        "dz_cm": dz_cm,
        "valid_pixel": valid,
    }
