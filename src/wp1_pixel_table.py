"""WP1: Build a unified TEMPO pixel metadata table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from io_tempo import (
    corrected_wavelength_grid,
    find_wavelength_indices,
    read_cldo4_support,
    read_rad_core,
    wavecal_shift_nm,
)


@dataclass
class PixelTableConfig:
    wl_min_nm: float = 460.0
    wl_max_nm: float = 490.0
    quality_must_be_zero: bool = True
    require_wavecal_opt_status_zero: bool = True
    band: str = "band_290_490_nm"


def build_pixel_table(
    rad_file: str,
    cldo4_file: str,
    scan_start: int,
    scan_end: int,
    xtrack_start: int,
    xtrack_end: int,
    config: PixelTableConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Construct pixel metadata table for a scan/xtrack subset."""
    cfg = config or PixelTableConfig()

    rad = read_rad_core(rad_file, band=cfg.band)
    cld = read_cldo4_support(cldo4_file)

    sl_scan = slice(scan_start, scan_end)
    sl_xt = slice(xtrack_start, xtrack_end)

    lat = rad["latitude"][sl_scan, sl_xt]
    lon = rad["longitude"][sl_scan, sl_xt]
    sza = rad["sza"][sl_scan, sl_xt]
    vza = rad["vza"][sl_scan, sl_xt]
    terrain_height = rad["terrain_height"][sl_scan, sl_xt]

    wavecal_params = rad["wavecal_params"][sl_scan, sl_xt, ...]
    wavecal_opt_status = rad["wavecal_opt_status"][sl_scan, sl_xt]
    lambda_corr = corrected_wavelength_grid(
        rad["nominal_wavelength"][sl_xt, :],
        wavecal_params,
    )

    ch_mask = find_wavelength_indices(lambda_corr, cfg.wl_min_nm, cfg.wl_max_nm)
    if not np.any(ch_mask):
        raise ValueError("No spectral channels found in requested wavelength window.")

    pq = rad["pixel_quality_flag"][sl_scan, sl_xt, :][:, :, ch_mask]
    if cfg.quality_must_be_zero:
        quality_ok = np.all(pq == 0, axis=2)
    else:
        quality_ok = np.isfinite(np.sum(pq, axis=2))

    if cfg.require_wavecal_opt_status_zero:
        # TEMPO V03: 0=initial, 1=converged, 2=converged (outside bounds) → accept.
        # 3=failed, -1=fill → reject.
        wavecal_ok = (wavecal_opt_status >= 0) & (wavecal_opt_status <= 2)
    else:
        wavecal_ok = wavecal_opt_status != -1  # exclude fill/invalid only

    fitted_scd = cld["fitted_slant_column"][sl_scan, sl_xt]
    cloud_fraction = cld["cloud_fraction"][sl_scan, sl_xt]
    cloud_pressure = cld["cloud_pressure"][sl_scan, sl_xt]
    surface_pressure = cld["surface_pressure"][sl_scan, sl_xt]
    proc_qf = cld["processing_quality_flag"][sl_scan, sl_xt]

    scan_idx = np.arange(scan_start, scan_end)[:, None]
    xtrack_idx = np.arange(xtrack_start, xtrack_end)[None, :]

    out = pd.DataFrame(
        {
            "mirror_step": np.broadcast_to(scan_idx, lat.shape).ravel(),
            "xtrack": np.broadcast_to(xtrack_idx, lat.shape).ravel(),
            "latitude": lat.ravel(),
            "longitude": lon.ravel(),
            "sza_deg": sza.ravel(),
            "vza_deg": vza.ravel(),
            "terrain_height_m": terrain_height.ravel(),
            "quality_ok": quality_ok.ravel(),
            "wavecal_opt_status": wavecal_opt_status.ravel(),
            "wavecal_ok": wavecal_ok.ravel(),
            "delta_lambda_nm": wavecal_shift_nm(wavecal_params).ravel(),
            "cldo4_fitted_slant_column": fitted_scd.ravel(),
            "cldo4_cloud_fraction": cloud_fraction.ravel(),
            "cldo4_cloud_pressure_hpa": cloud_pressure.ravel(),
            "cldo4_surface_pressure_hpa": surface_pressure.ravel(),
            "cldo4_processing_quality_flag": proc_qf.ravel(),
        }
    )

    finite_geo = np.isfinite(out["latitude"]) & np.isfinite(out["longitude"])
    finite_geom = np.isfinite(out["sza_deg"]) & np.isfinite(out["vza_deg"])
    out["valid_pixel"] = finite_geo & finite_geom & out["quality_ok"] & out["wavecal_ok"]

    diag = {
        "lambda_corrected_nm": lambda_corr,
        "delta_lambda_nm": wavecal_shift_nm(wavecal_params),
        "channel_mask": ch_mask,
        "wavelength_window_nm": np.array([cfg.wl_min_nm, cfg.wl_max_nm], dtype=float),
    }

    return out, diag
