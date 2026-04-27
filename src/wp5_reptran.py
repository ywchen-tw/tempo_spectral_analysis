"""WP5: REPTRAN-based gas optical depth computation for the TEMPO O2-O2 workflow.

Computes per-REPTRAN-band effective tau for six gas species:
  - H2O, O2, N2, N2O : k-distribution lookup tables
                        (reptran_solar_fine.lookup.<SPECIES>.cdf)
                        tau = xsec(P,T,[VMR]) × 1e-16 × n_species × dz
  - NO2, O3           : direct cross-sections × GEOS-CF number densities,
                        slit-convolved identically to the O2-O2 treatment in WP4

The lookup xsec values are stored in units of 10^{-20} m²/molecule; the factor
1e-16 converts them to cm²/molecule (libRadtran molecular.c line ~2970).

In the 460–490 nm window O2, N2, and N2O have no entries in their lookup tables
and therefore return zero tau; the implementation is otherwise general.

The original project_tau_to_reptran / write_reptran_outputs functions are
retained unchanged for backward compatibility with the O2-O2 tau pathway.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from wp4_tau import build_discrete_slit_kernel, convolve_spectrum_with_kernel

nc: Any = __import__("netCDF4")

_REPTRAN_DIR = "/Users/yuch8913/programming/er3t/libRadtran-2.0.6/data/correlated_k/reptran"
_CRS_DIR = "/Users/yuch8913/programming/er3t/libRadtran-2.0.6/data/crs"

# Converts REPTRAN xsec from 10^{-20} m²/molecule to cm²/molecule.
_XSEC_UNIT = 1e-16

_LOOKUP_SPECIES = ("H2O", "O2", "N2", "N2O")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ReptranConfig:
    """Settings for the REPTRAN projection and gas-tau computation steps."""

    reptran_file: str = f"{_REPTRAN_DIR}/reptran_solar_fine.cdf"
    wavelength_min_nm: float = 460.0
    wavelength_max_nm: float = 490.0
    no2_crs_file: str = f"{_CRS_DIR}/crs_NO2_UBremen_cf.dat"
    o3_crs_file: str = "/Users/yuch8913/programming/tempo/data/crs/SerdyuchenkoGorshelevVersionJuly2013.dat"


# ---------------------------------------------------------------------------
# Existing REPTRAN metadata loader (unchanged)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def load_reptran_metadata(
    reptran_file: str,
    wl_min: float = 460.0,
    wl_max: float = 490.0,
) -> dict:
    """Load REPTRAN band metadata and return the bands in the target window.

    Only bands whose weighted-mean representative wavelength falls strictly
    within [wl_min, wl_max] are returned, so downstream wavelength arrays
    never extend beyond the user's fitting range.
    """
    with nc.Dataset(reptran_file) as ds:
        wvl = np.asarray(ds.variables["wvl"][:], dtype=float)
        wvlmin = np.asarray(ds.variables["wvlmin"][:], dtype=float)
        wvlmax = np.asarray(ds.variables["wvlmax"][:], dtype=float)
        nwvl_in_band = np.asarray(ds.variables["nwvl_in_band"][:], dtype=int)
        iwvl = np.asarray(ds.variables["iwvl"][:], dtype=int)
        iwvl_weight = np.asarray(ds.variables["iwvl_weight"][:], dtype=float)
        band_name_raw = ds.variables["band_name"][:]

        species_name = None
        if "species_name" in ds.variables:
            species_name = ["".join(row.astype(str)).strip() for row in ds.variables["species_name"][:]]

        # Select bands that overlap [wl_min, wl_max]; edge bands are trimmed
        # below by the rep_wavelength containment check.
        band_mask = (wvlmin <= wl_max) & (wvlmax >= wl_min)
        band_indices = np.where(band_mask)[0]

        band_names, band_indices_kept = [], []
        rep_wavelengths, band_wmin, band_wmax, band_nwvl = [], [], [], []
        band_point_indices, band_point_wavelengths, band_point_weights = [], [], []

        for b in band_indices:
            n = int(nwvl_in_band[b])
            inds = iwvl[:n, b]
            weights = iwvl_weight[:n, b]
            valid = inds > 0
            inds = inds[valid]
            weights = weights[valid]
            if inds.size == 0:
                continue
            rep_wl = float(np.sum(wvl[inds - 1] * weights) / np.sum(weights))
            # Drop edge bands whose representative wavelength falls outside
            # the user's range — these arise because the overlap criterion
            # admits bands that straddle the boundary.
            if rep_wl < wl_min or rep_wl > wl_max:
                continue
            band_names.append("".join(band_name_raw[b].astype(str)).strip())
            band_indices_kept.append(int(b))
            rep_wavelengths.append(rep_wl)
            band_wmin.append(float(wvlmin[b]))
            band_wmax.append(float(wvlmax[b]))
            band_nwvl.append(int(inds.size))
            band_point_indices.append((inds - 1).astype(int))
            band_point_wavelengths.append(wvl[inds - 1].astype(float))
            band_point_weights.append(weights.astype(float))

    return {
        "band_index": np.array(band_indices_kept, dtype=int),
        "band_name": np.array(band_names, dtype=object),
        "rep_wavelength_nm": np.array(rep_wavelengths, dtype=float),
        "band_wmin_nm": np.array(band_wmin, dtype=float),
        "band_wmax_nm": np.array(band_wmax, dtype=float),
        "band_nwvl": np.array(band_nwvl, dtype=int),
        "band_point_indices": band_point_indices,
        "band_point_wavelengths": band_point_wavelengths,
        "band_point_weights": band_point_weights,
        "species_name": species_name,
    }


def interpolate_spectrum(wavelength_nm: np.ndarray, spectrum: np.ndarray, target_nm: np.ndarray) -> np.ndarray:
    """Interpolate a spectrum onto the target REPTRAN wavelengths."""
    return np.interp(target_nm, wavelength_nm, spectrum, left=np.nan, right=np.nan)


def _weighted_mean_finite(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean over finite values only; returns NaN if no valid points."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        return np.nan
    wsum = float(np.sum(w[valid]))
    if wsum <= 0:
        return np.nan
    return float(np.sum(v[valid] * w[valid]) / wsum)


def project_tau_to_reptran(tau_result: dict, reptran_cfg: ReptranConfig | None = None) -> dict:
    """Project per-pixel O2-O2 tau spectra onto REPTRAN bands in the target window."""
    cfg = reptran_cfg or ReptranConfig()
    meta = load_reptran_metadata(cfg.reptran_file, cfg.wavelength_min_nm, cfg.wavelength_max_nm)

    if tau_result["tau_eff"].size == 0:
        n_bands = meta["rep_wavelength_nm"].size
        return {
            "band_index": meta["band_index"],
            "band_name": meta["band_name"],
            "rep_wavelength_nm": meta["rep_wavelength_nm"],
            "band_wmin_nm": meta["band_wmin_nm"],
            "band_wmax_nm": meta["band_wmax_nm"],
            "band_nwvl": meta["band_nwvl"],
            "tau_reptran": np.empty((0, n_bands), dtype=float),
            "tau_reptran_mean": np.array([], dtype=float),
        }

    tau_rows, mean_rows = [], []
    for pix_idx in range(tau_result["tau_eff"].shape[0]):
        wl = np.asarray(tau_result["wavelength_nm"][pix_idx], dtype=float)
        tau_eff = np.asarray(tau_result["tau_eff"][pix_idx], dtype=float)
        band_tau = []
        for band_wl, band_weights in zip(meta["band_point_wavelengths"], meta["band_point_weights"]):
            band_wl = np.asarray(band_wl, dtype=float)
            if band_wl.size == 0:
                band_tau.append(np.nan)
                continue
            band_tau_pts = interpolate_spectrum(wl, tau_eff, band_wl)
            band_tau.append(_weighted_mean_finite(band_tau_pts, band_weights))
        band_tau = np.asarray(band_tau, dtype=float)
        tau_rows.append(band_tau)
        mean_rows.append(float(np.nanmean(band_tau)))

    return {
        "band_index": meta["band_index"],
        "band_name": meta["band_name"],
        "rep_wavelength_nm": meta["rep_wavelength_nm"],
        "band_wmin_nm": meta["band_wmin_nm"],
        "band_wmax_nm": meta["band_wmax_nm"],
        "band_nwvl": meta["band_nwvl"],
        "tau_reptran": np.vstack(tau_rows),
        "tau_reptran_mean": np.array(mean_rows, dtype=float),
    }


def write_reptran_outputs(out_prefix: str, reptran_result: dict, pixel_df: pd.DataFrame) -> tuple[str, str]:
    """Write REPTRAN O2-O2 outputs as HDF5-backed .nc and CSV summary."""
    out_prefix_path = Path(out_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    nc_path = str(out_prefix_path.with_suffix(".nc"))
    csv_path = str(out_prefix_path.with_suffix(".csv"))

    with h5py.File(nc_path, "w") as f:
        f.attrs["title"] = "TEMPO O2-O2 REPTRAN projection"
        f.attrs["description"] = "Tau projected onto REPTRAN representative wavelengths in the 460-490 nm window"
        f.create_dataset("band_index", data=np.asarray(reptran_result["band_index"], dtype=np.int32))
        f.create_dataset("rep_wavelength_nm", data=np.asarray(reptran_result["rep_wavelength_nm"], dtype=float))
        f.create_dataset("band_wmin_nm", data=np.asarray(reptran_result["band_wmin_nm"], dtype=float))
        f.create_dataset("band_wmax_nm", data=np.asarray(reptran_result["band_wmax_nm"], dtype=float))
        f.create_dataset("band_nwvl", data=np.asarray(reptran_result["band_nwvl"], dtype=np.int32))
        f.create_dataset("tau_reptran", data=np.asarray(reptran_result["tau_reptran"], dtype=float))
        f.create_dataset("tau_reptran_mean", data=np.asarray(reptran_result["tau_reptran_mean"], dtype=float))

    summary = pd.DataFrame(
        {
            "pixel_index": np.arange(len(pixel_df), dtype=int),
            "mirror_step": pixel_df["mirror_step"].to_numpy(dtype=int),
            "xtrack": pixel_df["xtrack"].to_numpy(dtype=int),
            "latitude": pixel_df["latitude"].to_numpy(dtype=float),
            "longitude": pixel_df["longitude"].to_numpy(dtype=float),
            "tau_reptran_mean": reptran_result["tau_reptran_mean"],
        }
    )
    summary.to_csv(csv_path, index=False)
    return nc_path, csv_path


# ---------------------------------------------------------------------------
# New: k-distribution lookup loaders
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def load_reptran_lookup(reptran_file: str, species: str) -> dict:
    """Load the k-distribution lookup table for one gas species.

    The companion file reptran_solar_fine.lookup.<species>.cdf stores xsec in
    units of 10^{-20} m²/molecule; apply _XSEC_UNIT (1e-16) to get cm²/molecule.
    """
    lookup_path = reptran_file.replace(".cdf", f".lookup.{species}.cdf")
    with nc.Dataset(lookup_path) as ds:
        return {
            "wvl_nm": np.array(ds["wvl"][:], dtype=float),
            "wvl_index": np.array(ds["wvl_index"][:], dtype=int),  # 1-based into global wvl
            "pressure_pa": np.array(ds["pressure"][:], dtype=float),  # decreasing, Pa
            "t_ref_k": np.array(ds["t_ref"][:], dtype=float),
            "t_pert_k": np.array(ds["t_pert"][:], dtype=float),
            "vmrs": np.array(ds["vmrs"][:], dtype=float),
            "xsec": np.array(ds["xsec"][:], dtype=np.float32),  # (n_t_pert, n_vmrs, nwvl, n_pressure)
        }


@lru_cache(maxsize=2)
def load_no2_cross_section(crs_file: str) -> dict:
    """Load the UBremen NO2 cross-section file (4-column polynomial coefficients).

    sigma = (C0 + C1*(T-273.15) + C2*(T-273.15)^2) * 1e-20 cm²/molecule
    Source: Bogumil et al., University of Bremen.
    """
    data = np.loadtxt(crs_file, comments="#")
    return {
        "wavelength_nm": data[:, 0],
        "c0": data[:, 1],
        "c1": data[:, 2],
        "c2": data[:, 3],
    }


def eval_no2_sigma(no2_xsec: dict, t_k: float) -> np.ndarray:
    """Evaluate NO2 cross section (cm²/molecule) at temperature t_k."""
    dt = t_k - 273.15
    return (no2_xsec["c0"] + no2_xsec["c1"] * dt + no2_xsec["c2"] * dt**2) * 1e-20


@lru_cache(maxsize=2)
def load_o3_cross_section(crs_file: str) -> dict:
    """Load the Serdyuchenko & Gorshelev O3 cross-section table.

    12-column file: wavelength (nm) + σ (cm²/molecule) at 11 temperatures
    293 K → 193 K in 10 K steps. File uses comma as decimal separator.
    The returned xsec_cm2 array is sorted by ascending temperature.
    """
    rows = []
    with open(crs_file) as fh:
        for line in fh:
            parts = line.replace(",", ".").split()
            if len(parts) < 12:
                continue
            try:
                vals = [float(x) for x in parts[:12]]
            except ValueError:
                continue
            if vals[0] > 100:   # wavelength sanity: skip non-data rows that parse
                rows.append(vals)
    data = np.array(rows, dtype=float)
    temps_k = np.array([293, 283, 273, 263, 253, 243, 233, 223, 213, 203, 193], dtype=float)
    sort_idx = np.argsort(temps_k)
    return {
        "wavelength_nm": data[:, 0],
        "xsec_cm2": data[:, 1:][:, sort_idx],  # (N, 11), ascending T
        "temperatures_k": temps_k[sort_idx],
    }


def eval_o3_sigma(o3_xsec: dict, t_k: float) -> np.ndarray:
    """Interpolate O3 cross section (cm²/molecule) at temperature t_k."""
    temps = o3_xsec["temperatures_k"]
    t_c = float(np.clip(t_k, temps[0], temps[-1]))
    ih = int(np.searchsorted(temps, t_c, side="right"))
    ih = max(1, min(ih, len(temps) - 1))
    il = ih - 1
    span = temps[ih] - temps[il]
    w = float((t_c - temps[il]) / span) if span > 0 else 0.0
    return (1.0 - w) * o3_xsec["xsec_cm2"][:, il] + w * o3_xsec["xsec_cm2"][:, ih]


# ---------------------------------------------------------------------------
# New: band-point → local lookup index map
# ---------------------------------------------------------------------------


def _build_band_loc_map(meta: dict, lookup: dict) -> list[np.ndarray]:
    """Map each REPTRAN band point's 0-based global index to its local index in the lookup.

    lookup['wvl_index'] is 1-based into the global wvl array;
    meta['band_point_indices'] is 0-based into the same array.
    Returns -1 for points absent from the lookup (species has no data there).
    """
    global_to_local: dict[int, int] = {int(g): i for i, g in enumerate(lookup["wvl_index"])}
    result = []
    for band_pts in meta["band_point_indices"]:
        locs = np.array([global_to_local.get(int(g) + 1, -1) for g in band_pts], dtype=int)
        result.append(locs)
    return result


# ---------------------------------------------------------------------------
# New: vectorised bracket helpers
# ---------------------------------------------------------------------------


def _bracket_dec(arr: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower/upper bracket indices and weight for a DECREASING array (pressure)."""
    n = len(arr)
    ih = np.searchsorted(-arr, -pts, side="right")
    ih = np.clip(ih, 1, n - 1)
    il = ih - 1
    span = arr[ih] - arr[il]                          # negative (decreasing)
    w = np.where(span != 0.0, (pts - arr[il]) / span, 0.0)
    w = np.clip(w, 0.0, 1.0)
    at_top = pts >= arr[0]
    at_bot = pts <= arr[-1]
    il = np.where(at_top, 0,     np.where(at_bot, n - 1, il)).astype(int)
    ih = np.where(at_top, 0,     np.where(at_bot, n - 1, ih)).astype(int)
    w  = np.where(at_top | at_bot, 0.0, w)
    return il, ih, w


def _bracket_inc(arr: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower/upper bracket indices and weight for an INCREASING array (t_pert, vmrs)."""
    n = len(arr)
    ih = np.searchsorted(arr, pts, side="right")
    ih = np.clip(ih, 1, n - 1)
    il = ih - 1
    span = arr[ih] - arr[il]
    w = np.where(span > 0.0, (pts - arr[il]) / span, 0.0)
    w = np.clip(w, 0.0, 1.0)
    at_lo = pts <= arr[0]
    at_hi = pts >= arr[-1]
    il = np.where(at_lo, 0,     np.where(at_hi, n - 1, il)).astype(int)
    ih = np.where(at_lo, 0,     np.where(at_hi, n - 1, ih)).astype(int)
    w  = np.where(at_lo | at_hi, 0.0, w)
    return il, ih, w


# ---------------------------------------------------------------------------
# New: per-pixel P/T/(VMR) interpolation state
# ---------------------------------------------------------------------------


def _precompute_layer_interp(
    lookup: dict,
    p_pa: np.ndarray,
    t_k: np.ndarray,
    h2o_vmr: np.ndarray | None = None,
) -> dict:
    """Pre-compute bracket indices and weights for all atmospheric layers.

    Call once per pixel per species; the result is reused across band points.
    """
    pressures = lookup["pressure_pa"]   # decreasing
    t_pert    = lookup["t_pert_k"]      # increasing
    t_ref     = lookup["t_ref_k"]       # (n_pressure,)
    vmrs      = lookup["vmrs"]          # increasing

    ip_lo, ip_hi, wp = _bracket_dec(pressures, p_pa)

    it_lo_lo, it_hi_lo, wt_lo = _bracket_inc(t_pert, t_k - t_ref[ip_lo])
    it_lo_hi, it_hi_hi, wt_hi = _bracket_inc(t_pert, t_k - t_ref[ip_hi])

    if h2o_vmr is not None and len(vmrs) > 1:
        iv_lo, iv_hi, wv = _bracket_inc(vmrs, h2o_vmr)
        single_vmr = False
    else:
        z = np.zeros(len(p_pa), dtype=int)
        iv_lo, iv_hi, wv = z, z, np.zeros(len(p_pa), dtype=float)
        single_vmr = True

    return {
        "ip_lo": ip_lo, "ip_hi": ip_hi, "wp": wp,
        "it_lo_lo": it_lo_lo, "it_hi_lo": it_hi_lo, "wt_lo": wt_lo,
        "it_lo_hi": it_lo_hi, "it_hi_hi": it_hi_hi, "wt_hi": wt_hi,
        "iv_lo": iv_lo, "iv_hi": iv_hi, "wv": wv,
        "single_vmr": single_vmr,
    }


def _interp_xsec_layers(lookup: dict, state: dict, loc_idx: int) -> np.ndarray:
    """Return k[n_layers] in cm²/molecule via NumPy advanced indexing (all layers at once).

    Trilinear in (P, T-perturbation, VMR) for H2O; bilinear for well-mixed gases.
    """
    xsec = lookup["xsec"]   # float32 (n_t_pert, n_vmrs, nwvl, n_pressure)
    il, ih, wp = state["ip_lo"], state["ip_hi"], state["wp"]
    it_ll, it_hl, wt_l = state["it_lo_lo"], state["it_hi_lo"], state["wt_lo"]
    it_lh, it_hh, wt_h = state["it_lo_hi"], state["it_hi_hi"], state["wt_hi"]

    if state["single_vmr"]:
        v = 0
        k_lo = (1 - wt_l) * xsec[it_ll, v, loc_idx, il] + wt_l * xsec[it_hl, v, loc_idx, il]
        k_hi = (1 - wt_h) * xsec[it_lh, v, loc_idx, ih] + wt_h * xsec[it_hh, v, loc_idx, ih]
    else:
        iv_lo, iv_hi, wv = state["iv_lo"], state["iv_hi"], state["wv"]

        def _bilin_t(it_lo, it_hi, wt, iv_, ip_):
            return (1 - wt) * xsec[it_lo, iv_, loc_idx, ip_] + wt * xsec[it_hi, iv_, loc_idx, ip_]

        k_lo = (1 - wv) * _bilin_t(it_ll, it_hl, wt_l, iv_lo, il) + wv * _bilin_t(it_ll, it_hl, wt_l, iv_hi, il)
        k_hi = (1 - wv) * _bilin_t(it_lh, it_hh, wt_h, iv_lo, ih) + wv * _bilin_t(it_lh, it_hh, wt_h, iv_hi, ih)

    return ((1 - wp) * k_lo + wp * k_hi).astype(float) * _XSEC_UNIT


# ---------------------------------------------------------------------------
# New: band-level tau for k-distribution gases
# ---------------------------------------------------------------------------


def _compute_tau_lookup_gas(
    meta: dict,
    lookup: dict,
    band_loc_maps: list[np.ndarray],
    p_pa: np.ndarray,
    t_k: np.ndarray,
    n_species: np.ndarray,
    dz: np.ndarray,
    h2o_vmr: np.ndarray | None = None,
) -> np.ndarray:
    """Band-level vertical tau for one pixel using the REPTRAN k-distribution.

    tau_vert_pt = sum_layers k(P,T,[VMR]) * n_species * dz
    Returns (n_bands,) with 0.0 where the lookup has no data for a band point.
    """
    state = _precompute_layer_interp(lookup, p_pa, t_k, h2o_vmr)
    col_factor = n_species * dz   # (n_lay,) molecules/cm²
    n_bands = len(meta["band_point_weights"])
    tau_bands = np.zeros(n_bands, dtype=float)

    # Unpack interpolation state once for the vectorised inner loop.
    xsec_arr = lookup["xsec"]  # (n_t_pert, n_vmrs, nwvl, n_pressure)
    il, ih, wp_p = state["ip_lo"], state["ip_hi"], state["wp"]
    it_ll, it_hl, wt_l = state["it_lo_lo"], state["it_hi_lo"], state["wt_lo"]
    it_lh, it_hh, wt_h = state["it_lo_hi"], state["it_hi_hi"], state["wt_hi"]
    single_vmr = state["single_vmr"]
    if not single_vmr:
        iv_lo_s, iv_hi_s, wv_s = state["iv_lo"], state["iv_hi"], state["wv"]

    for b, (loc_map, weights) in enumerate(zip(band_loc_maps, meta["band_point_weights"])):
        valid_mask = loc_map >= 0
        if not np.any(valid_mask):
            continue
        loc_valid = loc_map[valid_mask]              # (n_pts,)
        w_valid   = np.asarray(weights, dtype=float)[valid_mask]
        w_sum     = float(np.sum(w_valid))
        if w_sum <= 0:
            continue

        # Vectorised lookup over all band-points (n_pts) and layers (n_lay) at once.
        # xsec dims: (n_t_pert, n_vmrs, nwvl, n_pressure).
        # For single-VMR gases (O2/N2/N2O) dims 0,2,3 are advanced and dim 1 is a
        # basic scalar; NumPy places the broadcast advanced shape (n_lay, n_pts) first.
        if single_vmr:
            v = 0
            k_lo = (
                (1 - wt_l[:, None]) * xsec_arr[it_ll[:, None], v, loc_valid[None, :], il[:, None]] +
                wt_l[:, None]       * xsec_arr[it_hl[:, None], v, loc_valid[None, :], il[:, None]]
            )
            k_hi = (
                (1 - wt_h[:, None]) * xsec_arr[it_lh[:, None], v, loc_valid[None, :], ih[:, None]] +
                wt_h[:, None]       * xsec_arr[it_hh[:, None], v, loc_valid[None, :], ih[:, None]]
            )
        else:
            # H2O: all four dims are advanced → unambiguous (n_lay, n_pts) result.
            k_lo_lo = (
                (1 - wt_l[:, None]) * xsec_arr[it_ll[:, None], iv_lo_s[:, None], loc_valid[None, :], il[:, None]] +
                wt_l[:, None]       * xsec_arr[it_hl[:, None], iv_lo_s[:, None], loc_valid[None, :], il[:, None]]
            )
            k_lo_hi = (
                (1 - wt_l[:, None]) * xsec_arr[it_ll[:, None], iv_hi_s[:, None], loc_valid[None, :], il[:, None]] +
                wt_l[:, None]       * xsec_arr[it_hl[:, None], iv_hi_s[:, None], loc_valid[None, :], il[:, None]]
            )
            k_hi_lo = (
                (1 - wt_h[:, None]) * xsec_arr[it_lh[:, None], iv_lo_s[:, None], loc_valid[None, :], ih[:, None]] +
                wt_h[:, None]       * xsec_arr[it_hh[:, None], iv_lo_s[:, None], loc_valid[None, :], ih[:, None]]
            )
            k_hi_hi = (
                (1 - wt_h[:, None]) * xsec_arr[it_lh[:, None], iv_hi_s[:, None], loc_valid[None, :], ih[:, None]] +
                wt_h[:, None]       * xsec_arr[it_hh[:, None], iv_hi_s[:, None], loc_valid[None, :], ih[:, None]]
            )
            k_lo = (1 - wv_s[:, None]) * k_lo_lo + wv_s[:, None] * k_lo_hi
            k_hi = (1 - wv_s[:, None]) * k_hi_lo + wv_s[:, None] * k_hi_hi

        # k_interp: (n_lay, n_pts) in cm²/molecule
        k_interp  = ((1 - wp_p[:, None]) * k_lo + wp_p[:, None] * k_hi).astype(float) * _XSEC_UNIT
        tau_pts   = np.nansum(k_interp * col_factor[:, None], axis=0)    # (n_pts,)
        tau_bands[b] = float(np.nansum(tau_pts * w_valid) / w_sum)

    return tau_bands


# ---------------------------------------------------------------------------
# New: band-level tau for cross-section gases (NO2, O3) with slit convolution
# ---------------------------------------------------------------------------


def _compute_tau_crs_gas(
    meta: dict,
    crs_wl: np.ndarray,
    crs_sigma: np.ndarray,
    wl_nm: np.ndarray,
    n_species: np.ndarray,
    dz: np.ndarray,
    slit_row: "pd.Series",
    n_sigma_kernel: float,
    sec_airmass: float,
) -> np.ndarray:
    """Band-level tau_eff for one pixel via Beer-Lambert + slit convolution.

    sigma (already T-evaluated) is convolved with the per-xtrack TEMPO slit
    function — identical to the O2-O2 treatment in WP4 — then multiplied by
    the vertical column density and sec_airmass before projection onto bands.
    """
    sigma_on_tempo = np.interp(wl_nm, crs_wl, crs_sigma, left=0.0, right=0.0)

    dwl = float(np.nanmedian(np.diff(wl_nm)))
    if not np.isfinite(dwl) or dwl <= 0:
        return np.zeros(len(meta["band_point_weights"]), dtype=float)

    _, kernel = build_discrete_slit_kernel(
        dwl_nm=dwl,
        hw1e_nm=float(slit_row["kernel_hw1e_nm"]),
        shape=float(slit_row["kernel_shape"]),
        asym_nm=float(slit_row["kernel_asym_nm"]),
        n_sigma=n_sigma_kernel,
    )
    sigma_conv = convolve_spectrum_with_kernel(sigma_on_tempo, kernel)

    vcd = float(np.nansum(n_species * dz))   # molecules/cm²
    tau_eff = sigma_conv * vcd * sec_airmass  # (n_wl,)

    n_bands = len(meta["band_point_weights"])
    tau_bands = np.zeros(n_bands, dtype=float)
    for b, (band_wl, band_w) in enumerate(zip(meta["band_point_wavelengths"], meta["band_point_weights"])):
        pts = np.interp(np.asarray(band_wl, dtype=float), wl_nm, tau_eff, left=np.nan, right=np.nan)
        tau_bands[b] = _weighted_mean_finite(pts, band_w)
    return tau_bands


# ---------------------------------------------------------------------------
# New: main per-profile driver
# ---------------------------------------------------------------------------


def compute_tau_reptran_from_profiles(
    pixel_df: pd.DataFrame,
    profiles: dict,
    wavelength_diag: dict,
    slit_df: pd.DataFrame,
    reptran_cfg: ReptranConfig | None = None,
    scan_start: int = 0,
    xtrack_start: int = 0,
    n_sigma_kernel: float = 4.0,
) -> dict:
    """Compute per-REPTRAN-band effective tau for H2O, O2, N2, N2O, NO2, and O3.

    H2O/O2/N2/N2O use k-distribution lookup tables. In the 460–490 nm window
    O2/N2/N2O return zero (no lookup entries there); the code is general for
    other windows. NO2 and O3 use GEOS-CF number densities with slit-convolved
    cross sections, matching the WP4 O2-O2 spectral treatment.
    """
    cfg = reptran_cfg or ReptranConfig()
    meta = load_reptran_metadata(cfg.reptran_file, cfg.wavelength_min_nm, cfg.wavelength_max_nm)
    n_bands = meta["rep_wavelength_nm"].size

    lookups = {sp: load_reptran_lookup(cfg.reptran_file, sp) for sp in _LOOKUP_SPECIES}
    band_loc_maps = {sp: _build_band_loc_map(meta, lookups[sp]) for sp in _LOOKUP_SPECIES}
    no2_xsec = load_no2_cross_section(cfg.no2_crs_file)
    o3_xsec = load_o3_cross_section(cfg.o3_crs_file)

    _empty = np.empty((0, n_bands), dtype=float)
    _empty_ret: dict = {
        "pixel_index": np.array([], dtype=int),
        "band_index": meta["band_index"],
        "band_name": meta["band_name"],
        "rep_wavelength_nm": meta["rep_wavelength_nm"],
        "band_wmin_nm": meta["band_wmin_nm"],
        "band_wmax_nm": meta["band_wmax_nm"],
        "tau_h2o": _empty, "tau_o2": _empty, "tau_n2": _empty, "tau_n2o": _empty,
        "tau_no2": _empty, "tau_o3": _empty, "tau_total_gas": _empty,
        "sec_airmass": np.array([], dtype=float),
    }
    if len(pixel_df) == 0:
        return _empty_ret

    ch_mask = np.asarray(wavelength_diag["channel_mask"], dtype=bool)
    lambda_corr = np.asarray(wavelength_diag["lambda_corrected_nm"], dtype=float)
    slit_lookup = slit_df.set_index("xtrack")

    t_k_all   = np.asarray(profiles["t_k"],       dtype=float)
    n_air_all = np.asarray(profiles["n_air_cm3"],  dtype=float)
    n_o2_all  = np.asarray(profiles["n_o2_cm3"],   dtype=float)
    n_n2_all  = np.asarray(profiles["n_n2_cm3"],   dtype=float)
    n_n2o_all = np.asarray(profiles["n_n2o_cm3"],  dtype=float)
    n_h2o_all = np.asarray(profiles["n_h2o_cm3"],  dtype=float)
    n_no2_all = np.asarray(profiles["n_no2_cm3"],  dtype=float)
    n_o3_all  = np.asarray(profiles["n_o3_cm3"],   dtype=float)
    p_mid_all = np.asarray(profiles["p_mid_pa"],   dtype=float)
    dz_all    = np.asarray(profiles["dz_cm"],      dtype=float)
    valid_profile = np.asarray(profiles["valid_pixel"], dtype=bool)

    acc: dict[str, list] = {k: [] for k in ("h2o", "o2", "n2", "n2o", "no2", "o3")}
    pixel_rows: list[int] = []
    sec_rows: list[float] = []

    for i, row in pixel_df.reset_index(drop=True).iterrows():
        if not bool(row["valid_pixel"]) or not valid_profile[i]:
            continue
        rel_scan = int(row["mirror_step"] - scan_start)
        rel_xt   = int(row["xtrack"]      - xtrack_start)
        if not (0 <= rel_scan < lambda_corr.shape[0] and 0 <= rel_xt < lambda_corr.shape[1]):
            continue
        if row["xtrack"] not in slit_lookup.index:
            continue

        wl_nm = lambda_corr[rel_scan, rel_xt, ch_mask]
        wl_nm = wl_nm[np.isfinite(wl_nm)]
        if wl_nm.size < 3:
            continue

        p_pa  = p_mid_all[i, :]
        t_k   = t_k_all[i,   :]
        n_air = n_air_all[i, :]
        n_h2o = n_h2o_all[i, :]
        dz    = dz_all[i,    :]
        h2o_vmr = n_h2o / np.where(n_air > 0, n_air, np.nan)

        slit_row = slit_lookup.loc[int(row["xtrack"])]
        sec_airmass = (
            1.0 / np.cos(np.deg2rad(float(row["sza_deg"])))
            + 1.0 / np.cos(np.deg2rad(float(row["vza_deg"])))
        )

        # --- REPTRAN k-distribution gases ---
        acc["h2o"].append(
            _compute_tau_lookup_gas(
                meta, lookups["H2O"], band_loc_maps["H2O"],
                p_pa, t_k, n_h2o, dz, h2o_vmr=h2o_vmr,
            ) * sec_airmass
        )
        for key, sp, n_sp in (
            ("o2",  "O2",  n_o2_all[i,  :]),
            ("n2",  "N2",  n_n2_all[i,  :]),
            ("n2o", "N2O", n_n2o_all[i, :]),
        ):
            acc[key].append(
                _compute_tau_lookup_gas(
                    meta, lookups[sp], band_loc_maps[sp],
                    p_pa, t_k, n_sp, dz,
                ) * sec_airmass
            )

        # --- Cross-section gases: slit-convolve σ, then multiply by slant VCD ---
        # Use column-density-weighted mean T for both NO2 and O3 polynomial evaluation.
        col_w = n_air * dz
        t_eff = float(np.average(t_k, weights=np.where(col_w > 0, col_w, 1e-30)))

        acc["no2"].append(
            _compute_tau_crs_gas(
                meta,
                no2_xsec["wavelength_nm"], eval_no2_sigma(no2_xsec, t_eff),
                wl_nm, n_no2_all[i, :], dz, slit_row, n_sigma_kernel, sec_airmass,
            )
        )
        acc["o3"].append(
            _compute_tau_crs_gas(
                meta,
                o3_xsec["wavelength_nm"], eval_o3_sigma(o3_xsec, t_eff),
                wl_nm, n_o3_all[i, :], dz, slit_row, n_sigma_kernel, sec_airmass,
            )
        )

        pixel_rows.append(i)
        sec_rows.append(float(sec_airmass))

    if not pixel_rows:
        return _empty_ret

    tau_2d = {f"tau_{k}": np.vstack(v) for k, v in acc.items()}
    tau_total = sum(tau_2d.values())

    return {
        "pixel_index": np.array(pixel_rows, dtype=int),
        "xtrack": pixel_df.iloc[pixel_rows]["xtrack"].to_numpy(dtype=int),
        "mirror_step": pixel_df.iloc[pixel_rows]["mirror_step"].to_numpy(dtype=int),
        "band_index": meta["band_index"],
        "band_name": meta["band_name"],
        "rep_wavelength_nm": meta["rep_wavelength_nm"],
        "band_wmin_nm": meta["band_wmin_nm"],
        "band_wmax_nm": meta["band_wmax_nm"],
        **tau_2d,
        "tau_total_gas": tau_total,
        "sec_airmass": np.array(sec_rows, dtype=float),
    }


# ---------------------------------------------------------------------------
# New: output writer for gas tau
# ---------------------------------------------------------------------------


def write_reptran_gas_outputs(
    out_prefix: str,
    result: dict,
    pixel_df: pd.DataFrame,
) -> tuple[str, str]:
    """Write per-species REPTRAN band tau to HDF5 (.nc) and CSV summary."""
    out_path = Path(out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nc_path  = str(out_path.with_suffix(".nc"))
    csv_path = str(out_path.with_suffix(".csv"))

    gas_keys = ["tau_h2o", "tau_o2", "tau_n2", "tau_n2o", "tau_no2", "tau_o3", "tau_total_gas"]

    with h5py.File(nc_path, "w") as f:
        f.attrs["title"] = "TEMPO gas optical depth on REPTRAN bands"
        f.attrs["description"] = (
            "H2O/O2/N2/N2O from REPTRAN k-distribution (xsec×1e-16 cm²/mol); "
            "NO2/O3 from GEOS-CF VMR × slit-convolved cross sections"
        )
        for k in ("band_index", "rep_wavelength_nm", "band_wmin_nm", "band_wmax_nm"):
            f.create_dataset(k, data=np.asarray(result[k]))
        if result["pixel_index"].size:
            pix_sel = pixel_df.iloc[result["pixel_index"]]
            f.create_dataset("pixel_index",  data=np.asarray(result["pixel_index"],  dtype=np.int32))
            f.create_dataset("xtrack",       data=np.asarray(result["xtrack"],       dtype=np.int32))
            f.create_dataset("mirror_step",  data=np.asarray(result["mirror_step"],  dtype=np.int32))
            f.create_dataset("sec_airmass",  data=np.asarray(result["sec_airmass"]))
            f.create_dataset("latitude",     data=pix_sel["latitude"].to_numpy(dtype=float))
            f.create_dataset("longitude",    data=pix_sel["longitude"].to_numpy(dtype=float))
            for k in gas_keys:
                f.create_dataset(k, data=np.asarray(result[k]))

    if result["pixel_index"].size:
        pix_sel = pixel_df.iloc[result["pixel_index"]]
        rows: dict = {
            "pixel_index": result["pixel_index"],
            "mirror_step": result["mirror_step"],
            "xtrack":      result["xtrack"],
            "latitude":    pix_sel["latitude"].to_numpy(dtype=float),
            "longitude":   pix_sel["longitude"].to_numpy(dtype=float),
            "sec_airmass": result["sec_airmass"],
        }
        for k in gas_keys:
            rows[k + "_mean"] = np.nanmean(np.asarray(result[k]), axis=1)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    else:
        pd.DataFrame().to_csv(csv_path, index=False)

    return nc_path, csv_path
