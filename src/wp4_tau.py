"""WP4: O2-O2 convolution and tau computation.

This module is the first end-to-end tau implementation: it loads the O2-O2
cross sections, interpolates them to the wavecal-corrected TEMPO grid, applies
the slit kernel, integrates layer optical depth, and writes native tau output.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from wp3_slit_kernel import build_super_gaussian_kernel


@dataclass
class TauConfig:
    """Settings for the tau integration and native-output step."""

    cross_section_dir: str = "data/crs/O2O2_FinkenzellerEtVolkamer_297-500nm(vac)"
    n_sigma_kernel: float = 4.0
    output_wavelength_min_nm: float = 460.0
    output_wavelength_max_nm: float = 490.0
    write_netcdf: bool = True


@lru_cache(maxsize=1)
def load_cross_section_tables(cross_section_dir: str) -> dict:
    """Load the three temperature tables and align them on a common wavelength grid."""
    base = Path(cross_section_dir)
    files = sorted(base.glob("*.xs"))
    if not files:
        raise FileNotFoundError(f"No .xs files found in {cross_section_dir}")

    temp_entries: list[tuple[float, np.ndarray, np.ndarray]] = []
    for fp in files:
        arr = np.loadtxt(fp, comments="//")
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Unexpected cross-section file shape for {fp.name}: {arr.shape}")
        wl_nm = arr[:, 0].astype(float)
        sigma = arr[:, 1].astype(float)

        stem = fp.stem
        temp_str = stem.split("_")[-1].replace("K", "")
        temp_k = float(temp_str)
        temp_entries.append((temp_k, wl_nm, sigma))

    temp_entries.sort(key=lambda item: item[0])
    temps = np.array([item[0] for item in temp_entries], dtype=float)
    ref_wl = temp_entries[0][1]
    sigma_stack = []
    for temp_k, wl_nm, sigma in temp_entries:
        if not np.allclose(wl_nm, ref_wl, rtol=0, atol=1e-8):
            raise ValueError("Cross-section wavelength grids do not match across temperature files.")
        sigma_stack.append(sigma)

    return {
        "temperatures_k": temps,
        "wavelength_nm": ref_wl,
        "sigma_table": np.vstack(sigma_stack),
    }


def interpolate_sigma_to_wavelengths(
    target_wavelength_nm: np.ndarray,
    xsec_tables: dict,
) -> np.ndarray:
    """Interpolate cross sections from the table grid onto the TEMPO wavelength grid.

    Returns an array with shape `(n_temp, n_wavelength)`.
    """
    wl_tab = xsec_tables["wavelength_nm"]
    sigma_tab = xsec_tables["sigma_table"]
    out = np.empty((sigma_tab.shape[0], target_wavelength_nm.size), dtype=float)
    for i in range(sigma_tab.shape[0]):
        out[i, :] = np.interp(target_wavelength_nm, wl_tab, sigma_tab[i], left=np.nan, right=np.nan)
    return out


def interpolate_sigma_temperature(
    layer_temperature_k: float,
    sigma_on_target_wl_by_temp: np.ndarray,
    temps_k: np.ndarray,
) -> np.ndarray:
    """Interpolate the cross section at one layer temperature by linear weighting."""
    if layer_temperature_k <= temps_k[0]:
        return sigma_on_target_wl_by_temp[0]
    if layer_temperature_k >= temps_k[-1]:
        return sigma_on_target_wl_by_temp[-1]
    hi = int(np.searchsorted(temps_k, layer_temperature_k, side="right"))
    lo = hi - 1
    w = (layer_temperature_k - temps_k[lo]) / (temps_k[hi] - temps_k[lo])
    return (1.0 - w) * sigma_on_target_wl_by_temp[lo] + w * sigma_on_target_wl_by_temp[hi]


def build_discrete_slit_kernel(
    dwl_nm: float,
    hw1e_nm: float,
    shape: float,
    asym_nm: float,
    n_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a discrete slit kernel on an approximate wavelength-spacing grid."""
    half_span_nm = max(n_sigma * hw1e_nm, 3.0 * dwl_nm)
    n_side = max(1, int(np.ceil(half_span_nm / dwl_nm)))
    offsets_nm = np.arange(-n_side, n_side + 1, dtype=float) * dwl_nm
    kernel = build_super_gaussian_kernel(offsets_nm, hw1e_nm, shape, asym_nm)
    return offsets_nm, kernel


def convolve_spectrum_with_kernel(
    spectrum: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """Convolve a 1D spectrum with a normalized slit kernel."""
    return np.convolve(spectrum, kernel, mode="same")


def rayleigh_cross_section_cm2(wavelength_nm: np.ndarray) -> np.ndarray:
    """Rayleigh scattering cross-section per molecule (cm²/molecule).

    Bodhaine et al. (1999), J. Atm. Ocean Technol., 16, 1854–1861, Eq. 29.
    Wavelength input in nm; converted internally to microns.
    """
    wv0 = np.asarray(wavelength_nm, dtype=float) / 1000.0  # nm → µm
    num = 1.0455996 - 341.29061 * wv0 ** (-2.0) - 0.90230850 * wv0 ** 2.0
    den = 1.0 + 0.0027059889 * wv0 ** (-2.0) - 85.968563 * wv0 ** 2.0
    return (num / den) * 1e-28  # cm²/molecule


def compute_tau_rayleigh(
    wavelength_nm: np.ndarray,
    n_air_cm3: np.ndarray,
    dz_cm: np.ndarray,
) -> np.ndarray:
    """Vertical Rayleigh optical depth for one pixel.

    Parameters
    ----------
    wavelength_nm : (n_wl,)
    n_air_cm3 : (n_lay,)  air number density in molecules cm⁻³
    dz_cm : (n_lay,)      layer thickness in cm

    Returns
    -------
    tau_vert : (n_wl,)
    """
    sigma = rayleigh_cross_section_cm2(wavelength_nm)           # (n_wl,)
    tau_layer = sigma[:, np.newaxis] * n_air_cm3 * dz_cm        # (n_wl, n_lay)
    return np.nansum(tau_layer, axis=1)                          # (n_wl,)


def compute_tau_subset(
    pixel_df: pd.DataFrame,
    profiles: dict,
    wavelength_diag: dict,
    slit_df: pd.DataFrame,
    scan_start: int,
    xtrack_start: int,
    cross_section_dir: str = "data/crs/O2O2_FinkenzellerEtVolkamer_297-500nm(vac)",
    n_sigma_kernel: float = 4.0,
) -> dict:
    """Compute wavelength-resolved tau for a subset of pixels.

    The output is organized per pixel so the runner can write a compact native
    tau file and a per-pixel summary for quick inspection.
    """
    if len(pixel_df) == 0:
        return {
            "pixel_index": np.array([], dtype=int),
            "wavelength_nm": np.empty((0, 0), dtype=float),
            "tau_vert": np.empty((0, 0), dtype=float),
            "tau_eff": np.empty((0, 0), dtype=float),
            "tau_rayleigh_vert": np.empty((0, 0), dtype=float),
            "tau_rayleigh_eff": np.empty((0, 0), dtype=float),
            "tau_band_mean": np.array([], dtype=float),
            "sec_airmass": np.array([], dtype=float),
        }

    xsec_tables = load_cross_section_tables(cross_section_dir)
    temps_k = xsec_tables["temperatures_k"]
    ch_mask = np.asarray(wavelength_diag["channel_mask"], dtype=bool)
    lambda_corr = np.asarray(wavelength_diag["lambda_corrected_nm"], dtype=float)

    slit_lookup = slit_df.set_index("xtrack")
    pixel_rows = []
    tau_vert_rows = []
    tau_eff_rows = []
    tau_rayleigh_vert_rows = []
    tau_rayleigh_eff_rows = []
    tau_band_mean_rows = []
    sec_airmass_rows = []
    wavelength_rows = []

    t_k = np.asarray(profiles["t_k"], dtype=float)
    n_air_cm3 = np.asarray(profiles["n_air_cm3"], dtype=float)
    n_o2_cm3 = np.asarray(profiles["n_o2_cm3"], dtype=float)
    dz_cm = np.asarray(profiles["dz_cm"], dtype=float)
    valid_profile = np.asarray(profiles["valid_pixel"], dtype=bool)

    for i, row in pixel_df.reset_index(drop=True).iterrows():
        if not bool(row["valid_pixel"]):
            continue
        rel_scan = int(row["mirror_step"] - scan_start)
        rel_xt = int(row["xtrack"] - xtrack_start)
        if rel_scan < 0 or rel_scan >= lambda_corr.shape[0]:
            continue
        if rel_xt < 0 or rel_xt >= lambda_corr.shape[1]:
            continue

        if not valid_profile[i]:
            continue

        wl_nm = lambda_corr[rel_scan, rel_xt, ch_mask]
        finite = np.isfinite(wl_nm)
        wl_nm = wl_nm[finite]
        if wl_nm.size < 3:
            continue

        dwl_nm = float(np.nanmedian(np.diff(wl_nm)))
        if not np.isfinite(dwl_nm) or dwl_nm <= 0:
            continue

        if row["xtrack"] not in slit_lookup.index:
            continue
        slit_row = slit_lookup.loc[int(row["xtrack"])]
        offsets_nm, kernel = build_discrete_slit_kernel(
            dwl_nm=dwl_nm,
            hw1e_nm=float(slit_row["kernel_hw1e_nm"]),
            shape=float(slit_row["kernel_shape"]),
            asym_nm=float(slit_row["kernel_asym_nm"]),
            n_sigma=n_sigma_kernel,
        )

        # First interpolate the tabulated O2-O2 cross sections to the exact TEMPO
        # wavelength points for this pixel, then interpolate between temperatures.
        sigma_on_wl_by_temp = interpolate_sigma_to_wavelengths(wl_nm, xsec_tables)
        tau_layer = np.zeros((n_o2_cm3.shape[1], wl_nm.size), dtype=float)
        for lev in range(n_o2_cm3.shape[1]):
            sigma_layer = interpolate_sigma_temperature(
                float(t_k[i, lev]),
                sigma_on_wl_by_temp,
                temps_k,
            )
            sigma_layer = np.nan_to_num(sigma_layer, nan=0.0, posinf=0.0, neginf=0.0)
            sigma_conv = convolve_spectrum_with_kernel(sigma_layer, kernel)
            tau_layer[lev, :] = sigma_conv * (n_o2_cm3[i, lev] ** 2) * dz_cm[i, lev]

        tau_vert = np.nansum(tau_layer, axis=0)
        # Use a simple two-way geometry factor for the current baseline.
        sec_airmass = 1.0 / np.cos(np.deg2rad(float(row["sza_deg"]))) + 1.0 / np.cos(np.deg2rad(float(row["vza_deg"])))
        tau_eff = tau_vert * sec_airmass

        # Rayleigh scattering — Bodhaine et al. (1999) Eq. 29
        tau_rayleigh_vert = compute_tau_rayleigh(wl_nm, n_air_cm3[i, :], dz_cm[i, :])
        tau_rayleigh_eff = tau_rayleigh_vert * sec_airmass

        pixel_rows.append(i)
        wavelength_rows.append(wl_nm)
        tau_vert_rows.append(tau_vert)
        tau_eff_rows.append(tau_eff)
        tau_rayleigh_vert_rows.append(tau_rayleigh_vert)
        tau_rayleigh_eff_rows.append(tau_rayleigh_eff)
        tau_band_mean_rows.append(float(np.nanmean(tau_eff)))
        sec_airmass_rows.append(float(sec_airmass))

    if not pixel_rows:
        return {
            "pixel_index": np.array([], dtype=int),
            "wavelength_nm": np.empty((0, 0), dtype=float),
            "tau_vert": np.empty((0, 0), dtype=float),
            "tau_eff": np.empty((0, 0), dtype=float),
            "tau_rayleigh_vert": np.empty((0, 0), dtype=float),
            "tau_rayleigh_eff": np.empty((0, 0), dtype=float),
            "tau_band_mean": np.array([], dtype=float),
            "sec_airmass": np.array([], dtype=float),
            "xtrack": np.array([], dtype=int),
            "mirror_step": np.array([], dtype=int),
        }

    return {
        "pixel_index": np.array(pixel_rows, dtype=int),
        "xtrack": pixel_df.iloc[pixel_rows]["xtrack"].to_numpy(dtype=int),
        "mirror_step": pixel_df.iloc[pixel_rows]["mirror_step"].to_numpy(dtype=int),
        "wavelength_nm": np.vstack(wavelength_rows),
        "tau_vert": np.vstack(tau_vert_rows),
        "tau_eff": np.vstack(tau_eff_rows),
        "tau_rayleigh_vert": np.vstack(tau_rayleigh_vert_rows),
        "tau_rayleigh_eff": np.vstack(tau_rayleigh_eff_rows),
        "tau_band_mean": np.array(tau_band_mean_rows, dtype=float),
        "sec_airmass": np.array(sec_airmass_rows, dtype=float),
    }


def write_tau_netcdf(out_path: str, tau_result: dict, pixel_df: pd.DataFrame) -> None:
    """Write a compact native tau file for the computed subset.

    The file is HDF5-backed but saved with a `.nc` extension to keep the
    production path familiar while avoiding netCDF4 stub issues in the editor.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_pix = int(tau_result["tau_eff"].shape[0])
    n_wl = int(tau_result["tau_eff"].shape[1]) if n_pix else 0

    with h5py.File(out_path, "w") as f:
        f.attrs["title"] = "TEMPO O2-O2 native tau subset"
        f.attrs["description"] = "Wavelength-resolved O2-O2 tau on wavecal-corrected TEMPO grid"
        f.create_dataset("pixel", data=np.arange(n_pix, dtype=np.int32))
        f.create_dataset("spectral_channel", data=np.arange(n_wl, dtype=np.int32))

        if n_pix:
            f.create_dataset("mirror_step", data=tau_result["mirror_step"].astype(np.int32))
            f.create_dataset("xtrack", data=tau_result["xtrack"].astype(np.int32))
            f.create_dataset("latitude", data=pixel_df.iloc[tau_result["pixel_index"]]["latitude"].to_numpy(dtype=float))
            f.create_dataset("longitude", data=pixel_df.iloc[tau_result["pixel_index"]]["longitude"].to_numpy(dtype=float))
            f.create_dataset("sec_airmass", data=tau_result["sec_airmass"].astype(float))
            f.create_dataset("tau_band_mean", data=tau_result["tau_band_mean"].astype(float))
            f.create_dataset("wavelength_nm", data=tau_result["wavelength_nm"].astype(float))
            f.create_dataset("tau_vert", data=tau_result["tau_vert"].astype(float))
            f.create_dataset("tau_eff", data=tau_result["tau_eff"].astype(float))
            if "tau_rayleigh_vert" in tau_result and tau_result["tau_rayleigh_vert"].size:
                f.create_dataset("tau_rayleigh_vert", data=tau_result["tau_rayleigh_vert"].astype(float))
                f.create_dataset("tau_rayleigh_eff", data=tau_result["tau_rayleigh_eff"].astype(float))


def write_o2o2_vertical_debug_plot(
    out_path: str,
    pixel_df: pd.DataFrame,
    profiles: dict,
    wavelength_diag: dict,
    slit_df: pd.DataFrame,
    tau_result: dict,
    scan_start: int,
    xtrack_start: int,
    cross_section_dir: str = "data/crs/O2O2_FinkenzellerEtVolkamer_297-500nm(vac)",
    n_sigma_kernel: float = 4.0,
    debug_row: int = 0,
) -> str:
    """Write a per-layer O2-O2 debug plot for one fitted pixel.

    Left x-axis: layer O2-O2 vertical optical-depth contribution at a
    representative wavelength (where total tau_vert is maximal).
    Right x-axis: O2 number density profile for the same pixel.
    Y-axis is pressure (hPa, log scale).
    """
    if tau_result["pixel_index"].size == 0:
        raise ValueError("No tau_result pixels available for debug plotting.")
    if debug_row < 0 or debug_row >= int(tau_result["pixel_index"].size):
        raise ValueError(f"debug_row out of range: {debug_row}")

    import matplotlib.pyplot as plt

    xsec_tables = load_cross_section_tables(cross_section_dir)
    temps_k = xsec_tables["temperatures_k"]
    ch_mask = np.asarray(wavelength_diag["channel_mask"], dtype=bool)
    lambda_corr = np.asarray(wavelength_diag["lambda_corrected_nm"], dtype=float)
    slit_lookup = slit_df.set_index("xtrack")

    src_row = int(np.asarray(tau_result["pixel_index"], dtype=int)[debug_row])
    row = pixel_df.iloc[src_row]

    rel_scan = int(row["mirror_step"] - scan_start)
    rel_xt = int(row["xtrack"] - xtrack_start)
    if rel_scan < 0 or rel_scan >= lambda_corr.shape[0] or rel_xt < 0 or rel_xt >= lambda_corr.shape[1]:
        raise ValueError("Selected debug pixel is outside wavelength_diag subset bounds.")
    if int(row["xtrack"]) not in slit_lookup.index:
        raise ValueError("Selected debug pixel xtrack missing in slit lookup.")

    wl_nm = lambda_corr[rel_scan, rel_xt, ch_mask]
    finite = np.isfinite(wl_nm)
    wl_nm = wl_nm[finite]
    if wl_nm.size < 3:
        raise ValueError("Insufficient finite wavelengths for debug pixel.")

    dwl_nm = float(np.nanmedian(np.diff(wl_nm)))
    slit_row = slit_lookup.loc[int(row["xtrack"])]
    _, kernel = build_discrete_slit_kernel(
        dwl_nm=dwl_nm,
        hw1e_nm=float(slit_row["kernel_hw1e_nm"]),
        shape=float(slit_row["kernel_shape"]),
        asym_nm=float(slit_row["kernel_asym_nm"]),
        n_sigma=n_sigma_kernel,
    )

    t_k = np.asarray(profiles["t_k"], dtype=float)
    n_o2_cm3 = np.asarray(profiles["n_o2_cm3"], dtype=float)
    dz_cm = np.asarray(profiles["dz_cm"], dtype=float)
    p_mid_hpa = np.asarray(profiles["p_mid_pa"], dtype=float)[src_row, :] * 0.01

    sigma_on_wl_by_temp = interpolate_sigma_to_wavelengths(wl_nm, xsec_tables)
    n_lev = n_o2_cm3.shape[1]
    tau_layer = np.zeros((n_lev, wl_nm.size), dtype=float)
    for lev in range(n_lev):
        sigma_layer = interpolate_sigma_temperature(
            float(t_k[src_row, lev]),
            sigma_on_wl_by_temp,
            temps_k,
        )
        sigma_layer = np.nan_to_num(sigma_layer, nan=0.0, posinf=0.0, neginf=0.0)
        sigma_conv = convolve_spectrum_with_kernel(sigma_layer, kernel)
        tau_layer[lev, :] = sigma_conv * (n_o2_cm3[src_row, lev] ** 2) * dz_cm[src_row, lev]

    tau_vert = np.nansum(tau_layer, axis=0)
    ref_idx = int(np.nanargmax(tau_vert))
    tau_layer_ref = tau_layer[:, ref_idx]
    n_o2_profile = n_o2_cm3[src_row, :]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax2 = ax1.twiny()

    ax1.plot(tau_layer_ref, p_mid_hpa, color="tab:blue", lw=2, label="O2-O2 dτ_layer")
    ax2.plot(n_o2_profile, p_mid_hpa, color="tab:red", lw=1.8, ls="--", label="n(O2)")

    ax1.set_yscale("log")
    ax1.invert_yaxis()
    ax1.set_xlabel(f"Layer O2-O2 optical depth at {wl_nm[ref_idx]:.2f} nm")
    ax2.set_xlabel("O2 number density (molec cm$^{-3}$)")
    ax1.set_ylabel("Pressure (hPa)")
    ax1.grid(alpha=0.25)

    title = (
        f"O2-O2 Vertical Debug: mirror_step={int(row['mirror_step'])}, xtrack={int(row['xtrack'])}\n"
        f"lat={float(row['latitude']):.3f}, lon={float(row['longitude']):.3f}"
    )
    ax1.set_title(title, fontsize=9)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)