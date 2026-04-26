"""I/O helpers for TEMPO and GEOS-CF files used in the O2-O2 workflow.

This module owns all file-reading logic for RAD, IRR, CLDO4, and GEOS-CF.
It also provides the wavelength-correction helpers used by WP1-WP4.

`h5py` is used for TEMPO products because some IRR files can fail with
`netCDF4` group parsing in certain environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import h5py
import numpy as np


@dataclass
class TempoPaths:
    """Container for the standard TEMPO input files used by the workflow."""

    rad_file: str
    irr_file: str
    cldo4_file: str
    geos_file: str


def _read_h5(path: str, key: str) -> np.ndarray:
    """Read a dataset from an HDF5-compatible TEMPO file as a NumPy array."""
    with h5py.File(path, "r") as f:
        return np.array(f[key])


def read_rad_core(rad_file: str, band: str = "band_290_490_nm") -> dict:
    """Read the RAD arrays needed to build the pixel table and wavelength grid."""
    base = f"{band}"
    out = {
        "time": _read_h5(rad_file, "time"),
        "latitude": _read_h5(rad_file, f"{base}/latitude"),
        "longitude": _read_h5(rad_file, f"{base}/longitude"),
        "sza": _read_h5(rad_file, f"{base}/solar_zenith_angle"),
        "vza": _read_h5(rad_file, f"{base}/viewing_zenith_angle"),
        "terrain_height": _read_h5(rad_file, f"{base}/terrain_height"),
        "nominal_wavelength": _read_h5(rad_file, f"{base}/nominal_wavelength"),
        "wavecal_params": _read_h5(rad_file, f"{base}/wavecal_params"),
        "wavecal_opt_status": _read_h5(rad_file, f"{base}/wavecal_opt_status"),
        "pixel_quality_flag": _read_h5(rad_file, f"{base}/pixel_quality_flag"),
    }
    return out


def wavecal_shift_nm(wavecal_params: np.ndarray) -> np.ndarray:
    """Return the 0th-order wavelength shift `delta_lambda(m, x)` in nm.

    Supports either shape `(m, x, p)` or `(m, x)`.
    """
    if wavecal_params.ndim == 3:
        return wavecal_params[:, :, 0]
    if wavecal_params.ndim == 2:
        return wavecal_params
    raise ValueError(f"Unexpected wavecal_params shape: {wavecal_params.shape}")


def corrected_wavelength_grid(
    nominal_wavelength: np.ndarray,
    wavecal_params: np.ndarray,
) -> np.ndarray:
    """Build the wavecal-corrected TEMPO wavelength cube for all pixels."""
    delta_lambda = wavecal_shift_nm(wavecal_params)
    return nominal_wavelength[np.newaxis, :, :] + delta_lambda[:, :, np.newaxis]


def read_cldo4_support(cldo4_file: str) -> dict:
    """Read CLDO4 fields used for validation and QC comparisons."""
    return {
        "fitted_slant_column": _read_h5(cldo4_file, "support_data/fitted_slant_column"),
        "cloud_fraction": _read_h5(cldo4_file, "product/cloud_fraction"),
        "cloud_pressure": _read_h5(cldo4_file, "product/cloud_pressure"),
        "surface_pressure": _read_h5(cldo4_file, "support_data/surface_pressure"),
        "processing_quality_flag": _read_h5(cldo4_file, "product/processing_quality_flag"),
    }


def read_irr_core(irr_file: str, band: str = "band_290_490_nm") -> dict:
    """Read the IRR arrays needed to reconstruct slit kernels and QA them."""
    base = f"{band}"
    return {
        "time": _read_h5(irr_file, "time"),
        "nominal_wavelength": _read_h5(irr_file, f"{base}/nominal_wavelength"),
        "sf_hw1e": _read_h5(irr_file, f"{base}/sf_hw1e"),
        "sf_shape": _read_h5(irr_file, f"{base}/sf_shape"),
        "sf_asym": _read_h5(irr_file, f"{base}/sf_asym"),
        "wavecal_params": _read_h5(irr_file, f"{base}/wavecal_params"),
        "wavecal_opt_status": _read_h5(irr_file, f"{base}/wavecal_opt_status"),
        "pixel_quality_flag": _read_h5(irr_file, f"{base}/pixel_quality_flag"),
    }


def read_geos_core(geos_file: str) -> dict:
    """Read the GEOS-CF arrays used for nearest-neighbor collocation and profiles."""
    with h5py.File(geos_file, "r") as f:
        return {
            "lat": np.array(f["lat"]),
            "lon": np.array(f["lon"]),
            "lev": np.array(f["lev"]),
            "time": np.array(f["time"]),
            "PS": np.array(f["PS"]),
            "T": np.array(f["T"]),
            "Q": np.array(f["Q"]),
            "NO2": np.array(f["NO2"]),  # mol/mol volume mixing ratio
            "O3": np.array(f["O3"]),    # mol/mol volume mixing ratio
        }


def find_wavelength_indices(
    wavelength_grid: np.ndarray,
    wl_min_nm: float,
    wl_max_nm: float,
) -> np.ndarray:
    """Return a spectral-channel mask for the target wavelength range.

    Uses median wavelength over non-spectral dimensions to build a global channel
    mask. Input can be `(x, s)` or `(m, x, s)`.
    """
    if wavelength_grid.ndim == 2:
        wl_1d = np.nanmedian(wavelength_grid, axis=0)
    elif wavelength_grid.ndim == 3:
        wl_1d = np.nanmedian(wavelength_grid, axis=(0, 1))
    else:
        raise ValueError(f"Unexpected wavelength grid shape: {wavelength_grid.shape}")
    return (wl_1d >= wl_min_nm) & (wl_1d <= wl_max_nm)


def nearest_latlon_indices(
    geos_lat: np.ndarray,
    geos_lon: np.ndarray,
    pix_lat: np.ndarray,
    pix_lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return nearest-neighbor GEOS lat/lon indices for each pixel location.

    Longitude input is normalized to [-180, 180] to match GEOS coordinates.
    """
    lon = ((pix_lon + 180.0) % 360.0) - 180.0
    ilat = np.abs(geos_lat[:, None] - pix_lat[None, :]).argmin(axis=0)
    ilon = np.abs(geos_lon[:, None] - lon[None, :]).argmin(axis=0)
    return ilat, ilon
