"""WP3: IRR slit-kernel reconstruction and QA.

This module reconstructs a normalized slit kernel from the IRR product and
records basic QA fields so later convolution uses the correct instrument shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from io_tempo import corrected_wavelength_grid, read_irr_core, wavecal_shift_nm


@dataclass
class SlitKernelConfig:
    """Settings that control slit-kernel construction and QA filtering."""

    wl_min_nm: float = 460.0
    wl_max_nm: float = 490.0
    n_kernel_points: int = 401
    sigma_clip_factor: float = 4.0
    require_wavecal_opt_status_zero: bool = True
    require_pixel_quality_zero: bool = True
    band: str = "band_290_490_nm"


def build_super_gaussian_kernel(
    offsets_nm: np.ndarray,
    hw1e_nm: float,
    shape: float,
    asym_nm: float,
) -> np.ndarray:
    """Build a normalized super-Gaussian slit kernel.

    Parameters are interpreted as:
    - hw1e_nm: half-width at 1/e
    - shape: super-Gaussian exponent
    - asym_nm: signed width adjustment applied by offset sign
    """
    offsets_nm = np.asarray(offsets_nm, dtype=float)
    width = hw1e_nm + np.sign(offsets_nm) * asym_nm
    width = np.where(np.abs(width) > 1e-12, width, np.nan)
    kernel = np.exp(-np.abs(offsets_nm / width) ** shape)
    kernel = np.where(np.isfinite(kernel), kernel, 0.0)
    total = np.sum(kernel)
    if total <= 0:
        raise ValueError("Kernel normalization failed: non-positive total weight")
    return kernel / total


def build_slit_kernel_table(
    irr_file: str,
    lambda_corrected_nm: np.ndarray,
    xtrack_start: int,
    xtrack_end: int,
    config: SlitKernelConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Construct per-xtrack slit-kernel diagnostics using corrected wavelengths.

    `lambda_corrected_nm` must have shape `(mirror_step, xtrack, spectral_channel)`.
    """
    cfg = config or SlitKernelConfig()
    irr = read_irr_core(irr_file, band=cfg.band)

    # IRR has one mirror step in this granule; use the first row explicitly.
    nominal = irr["nominal_wavelength"][xtrack_start:xtrack_end, :]
    sf_hw1e = irr["sf_hw1e"][0, xtrack_start:xtrack_end, :]
    sf_shape = irr["sf_shape"][0, xtrack_start:xtrack_end, :]
    sf_asym = irr["sf_asym"][0, xtrack_start:xtrack_end, :]
    wavecal_opt_status = irr["wavecal_opt_status"][0, xtrack_start:xtrack_end]
    pqf = irr["pixel_quality_flag"][0, xtrack_start:xtrack_end, :]

    if lambda_corrected_nm.ndim != 3:
        raise ValueError("lambda_corrected_nm must have shape (mirror_step, xtrack, spectral_channel)")

    # Use the first mirror step for the subset, consistent with the local test setup.
    lambda_row = lambda_corrected_nm[0, xtrack_start:xtrack_end, :]
    n_xt = lambda_row.shape[0]
    n_chan = lambda_row.shape[1]

    # Define a common offset grid in nm, centered on zero, so kernels share a
    # consistent support window across xtrack pixels.
    offset_max = np.nanmax(np.abs(lambda_row - np.nanmedian(lambda_row, axis=1, keepdims=True)))
    if not np.isfinite(offset_max) or offset_max <= 0:
        offset_max = 0.25
    offset_grid = np.linspace(-cfg.sigma_clip_factor * offset_max, cfg.sigma_clip_factor * offset_max, cfg.n_kernel_points)

    kernel_rows = []
    for i in range(n_xt):
        wl = lambda_row[i, :]
        center = np.nanmedian(wl)
        in_window = (wl >= cfg.wl_min_nm) & (wl <= cfg.wl_max_nm)
        if np.count_nonzero(in_window) < 3:
            continue

        if cfg.require_wavecal_opt_status_zero and int(wavecal_opt_status[i]) not in (0, 1, 2):
            # TEMPO V03: 0=initial, 1=converged, 2=converged (outside bounds)
            # are usable here; 3=failed and -1=fill are rejected.
            continue

        if cfg.require_pixel_quality_zero and not bool(np.any(pqf[i, in_window] == 0)):
            continue

        hw1e = float(np.nanmedian(sf_hw1e[i, in_window]))
        shape = float(np.nanmedian(sf_shape[i, in_window]))
        asym = float(np.nanmedian(sf_asym[i, in_window]))
        kernel = build_super_gaussian_kernel(offset_grid, hw1e, shape, asym)
        kernel_rows.append(
            {
                "xtrack": xtrack_start + i,
                "wavecal_opt_status": int(wavecal_opt_status[i]),
                "quality_ok": bool(np.any(pqf[i, in_window] == 0)),
                "kernel_sum": float(np.sum(kernel)),
                "kernel_peak": float(np.max(kernel)),
                "kernel_mean_offset_nm": float(np.sum(offset_grid * kernel)),
                "kernel_hw1e_nm": hw1e,
                "kernel_shape": shape,
                "kernel_asym_nm": asym,
                "n_window_channels": int(np.count_nonzero(in_window)),
                "center_wavelength_nm": float(center),
                "delta_lambda_nm": float(
                    np.ravel(irr["wavecal_params"][0, xtrack_start + i, ...])[0]
                ),
            }
        )

    kernel_df = pd.DataFrame(kernel_rows)
    diag = {
        "offset_grid_nm": offset_grid,
        "kernel_table": kernel_df,
        "lambda_corrected_nm": lambda_row,
        "nominal_wavelength_nm": nominal,
        "wavecal_opt_status": wavecal_opt_status,
    }
    return kernel_df, diag