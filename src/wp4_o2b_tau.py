"""WP4 (O2-B band): HITRAN line-by-line tau computation for ~685–695 nm.

Computes optical depth for two absorbers on the native TEMPO wavelength grid:

  O2 B-band (b¹Σg⁺ ← X³Σg⁻):  τ = σ_O2(λ,T) × n_O2 × dz   (linear in n_O2)
  H2O continuum/lines:           τ = σ_H2O(λ,T) × n_H2O × dz  (linear in n_H2O)

Both cross-sections are computed line-by-line from HITRAN .par files using
Voigt profiles (scipy.special.wofz).  The slit convolution is done at high
resolution (_HIRES_DWL_NM = 2×10⁻⁴ nm) before sampling at the pixel's
wavecal-corrected TEMPO wavelengths, so individual lines are handled
correctly even though they are narrower than one TEMPO channel.

Key differences from wp4_tau.py (O2-O2 CIA):
  - Cross-sections from HITRAN, not pre-measured .xs tables
  - τ ∝ n_species × dz  (monomer Beer-Lambert), not n_O2² × dz
  - Per-pixel FFT slit convolution on the high-res grid before resampling
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.special import wofz

from wp3_slit_kernel import build_super_gaussian_kernel
from wp4_tau import compute_tau_rayleigh

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_T_REF_K  = 296.0       # HITRAN reference temperature (K)
_HC_K_CM  = 1.4387769   # hc / k_B in cm·K
_P_REP_HPA = 500.0      # representative pressure for cross-section precomputation
_M_O2_AMU  = 31.9988    # ¹⁶O₂ molecular mass (amu)
_M_H2O_AMU = 18.015     # ¹H₂¹⁶O molecular mass (amu)

_HIRES_DWL_NM    = 2e-4    # high-res wavelength step for Voigt convolution (nm)
_HIRES_PAD_NM    = 2.0     # padding on each side of the fitting window (nm)
_LINE_CUTOFF_HWHM = 100.0  # Voigt profile set to zero beyond this many HWHM

# Representative self-broadening VMR for precomputation (only affects line width,
# not the per-layer tau magnitude which uses actual n_h2o from profiles).
_H2O_SELF_VMR_REP = 0.005   # 0.5 % — lower-troposphere representative

# Partition-function power-law exponents Q(T)/Q(T_ref) ≈ (T/T_ref)^q_exp:
#   O2  — linear rigid rotor:      q = 1.0
#   H2O — nonlinear asymmetric top: q = 1.5
_Q_EXP_O2  = 1.0
_Q_EXP_H2O = 1.5


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TauO2BConfig:
    """Settings for the O2-B + H2O HITRAN tau computation."""

    hitran_o2b_file: str  = "data/crs/hitran_o2b.par"
    hitran_h2o_file: str  = "data/crs/hitran_h2o_o2b_range.par"
    n_sigma_kernel: float = 4.0
    output_wavelength_min_nm: float = 683.0
    output_wavelength_max_nm: float = 697.0
    temperatures_k: tuple[float, ...] = (180.0, 210.0, 240.0, 270.0, 300.0)


# ---------------------------------------------------------------------------
# HITRAN parser  (shared by O2 and H2O)
# ---------------------------------------------------------------------------


def parse_hitran_par(par_file: str) -> dict:
    """Parse a HITRAN 160-character .par file.

    Column layout (0-based Python slicing):
      [0:2]   molecule id
      [2:3]   isotopologue id
      [3:15]  vacuum wavenumber ν₀ (cm⁻¹)
      [15:25] intensity S at 296 K  (cm⁻¹/(mol·cm⁻²))
      [35:40] air-broadening HWHM γ_air at 296 K, 1 atm (cm⁻¹/atm)
      [40:45] self-broadening HWHM γ_self at 296 K (cm⁻¹/atm)
      [45:55] lower-state energy E″ (cm⁻¹)
      [55:59] temperature exponent n for γ_air
      [59:67] pressure shift δ at 296 K, 1 atm (cm⁻¹/atm)

    Lines with missing or non-positive ν₀/S are silently skipped.
    """
    nu0_l, S_l, gair_l, gself_l, E_l, nair_l, delta_l = [], [], [], [], [], [], []
    with open(par_file) as fh:
        for line in fh:
            if len(line) < 67:
                continue
            try:
                nu0        = float(line[3:15])
                S_296      = float(line[15:25])
                gamma_air  = float(line[35:40])
                gamma_self = float(line[40:45])
                E_lower    = float(line[45:55])
                n_air      = float(line[55:59])
                delta      = float(line[59:67])
            except ValueError:
                continue
            if not (np.isfinite(nu0) and nu0 > 0 and np.isfinite(S_296) and S_296 > 0):
                continue
            nu0_l.append(nu0);       S_l.append(S_296)
            gair_l.append(gamma_air); gself_l.append(gamma_self)
            E_l.append(E_lower);     nair_l.append(n_air);  delta_l.append(delta)

    return {
        "nu0_cm1":       np.array(nu0_l,    dtype=float),
        "S_296K":        np.array(S_l,      dtype=float),
        "gamma_air":     np.array(gair_l,   dtype=float),
        "gamma_self":    np.array(gself_l,  dtype=float),
        "E_lower_cm1":   np.array(E_l,      dtype=float),
        "n_air":         np.array(nair_l,   dtype=float),
        "delta_cm1_atm": np.array(delta_l,  dtype=float),
    }


# ---------------------------------------------------------------------------
# Line-by-line cross-section  (generalised for any HITRAN species)
# ---------------------------------------------------------------------------


def _intensity_at_t(lines: dict, t_k: float, q_exponent: float = 1.0) -> np.ndarray:
    """Temperature-correct HITRAN intensities from T_ref = 296 K to t_k.

    Standard HITRAN formula:
        S(T) = S(296) × [Q(296)/Q(T)] × exp[−c₂E″(1/T − 1/296)]
                       × [1 − exp(−c₂ν₀/T)] / [1 − exp(−c₂ν₀/296)]

    Q(T)/Q(296) ≈ (T/296)^q_exponent
        q_exponent = 1.0  →  O2  (linear rigid rotor)
        q_exponent = 1.5  →  H2O (nonlinear asymmetric top)
    """
    c2    = _HC_K_CM
    nu0   = lines["nu0_cm1"]
    E_pp  = lines["E_lower_cm1"]
    S_ref = lines["S_296K"]

    q_ratio  = (_T_REF_K / t_k) ** q_exponent
    boltz    = np.exp(-c2 * E_pp * (1.0 / t_k - 1.0 / _T_REF_K))
    stim_t   = 1.0 - np.exp(-c2 * nu0 / t_k)
    stim_ref = 1.0 - np.exp(-c2 * nu0 / _T_REF_K)
    stim_ref = np.where(np.abs(stim_ref) > 1e-30, stim_ref, np.nan)
    return S_ref * q_ratio * boltz * stim_t / stim_ref


def _compute_sigma_on_nu_grid(
    nu_grid_cm1: np.ndarray,
    lines: dict,
    t_k: float,
    p_hpa: float,
    m_amu: float,
    q_exponent: float = 1.0,
    self_vmr: float = 0.0,
) -> np.ndarray:
    """Compute cross-section σ(ν) in cm²/molecule on a wavenumber grid.

    Parameters
    ----------
    m_amu     : molecular mass in amu (sets Doppler width)
    q_exponent: partition-function power law exponent (1.0 for O2, 1.5 for H2O)
    self_vmr  : mole fraction of the absorber (used for self-broadening term)

    Lorentz HWHM: γ_L = [γ_air(1−self_vmr) + γ_self·self_vmr] × P_atm × (T_ref/T)^n
    """
    p_atm = p_hpa / 1013.25

    S_t        = _intensity_at_t(lines, t_k, q_exponent)
    nu0        = lines["nu0_cm1"]
    gamma_air  = lines["gamma_air"]
    gamma_self = lines["gamma_self"]
    n_air      = lines["n_air"]
    delta      = lines["delta_cm1_atm"]

    # Doppler (Gaussian) HWHM in cm⁻¹
    alpha_D = nu0 * 3.5812e-7 * np.sqrt(t_k / m_amu)
    # Lorentz (pressure) HWHM in cm⁻¹, with self-broadening
    gamma_mix = gamma_air * (1.0 - self_vmr) + gamma_self * self_vmr
    gamma_L   = gamma_mix * p_atm * (_T_REF_K / t_k) ** n_air
    # Pressure-shifted line centres
    nu0_eff = nu0 + delta * p_atm

    sigma = np.zeros(len(nu_grid_cm1), dtype=float)
    dnu   = float(nu_grid_cm1[1] - nu_grid_cm1[0]) if len(nu_grid_cm1) > 1 else 0.01

    for i in range(len(nu0)):
        hwhm     = max(float(alpha_D[i]), float(gamma_L[i]))
        half_win = _LINE_CUTOFF_HWHM * hwhm
        lo_idx = max(0,                int((nu0_eff[i] - half_win - nu_grid_cm1[0]) / dnu) - 1)
        hi_idx = min(len(nu_grid_cm1), int((nu0_eff[i] + half_win - nu_grid_cm1[0]) / dnu) + 2)
        if lo_idx >= hi_idx:
            continue
        nu_win = nu_grid_cm1[lo_idx:hi_idx]
        # Faddeeva argument: z = sqrt(ln2) × [(ν−ν₀) + iγ_L] / α_D
        z     = np.sqrt(np.log(2.0)) * ((nu_win - nu0_eff[i]) + 1j * float(gamma_L[i])) / float(alpha_D[i])
        voigt = np.real(wofz(z)) * (np.sqrt(np.log(2.0) / np.pi) / float(alpha_D[i]))
        sigma[lo_idx:hi_idx] += float(S_t[i]) * voigt

    return sigma   # cm²/molecule at each ν grid point


# ---------------------------------------------------------------------------
# Generic precomputed cross-section table  (cached, shared by O2 and H2O)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _load_hitran_xsec_tables(
    hitran_file: str,
    wl_min_nm: float,
    wl_max_nm: float,
    m_amu: float,
    q_exponent: float,
    self_vmr: float,
    temperatures_k: tuple[float, ...],
) -> dict:
    """Precompute cross-section table for any HITRAN species.

    Cross-sections are computed at _P_REP_HPA = 500 hPa for all requested
    temperatures on a uniform high-res wavelength grid that spans
    [wl_min - pad, wl_max + pad].  The table is cached (lru_cache) so the
    slow line-by-line computation runs only once per unique set of arguments.

    Returns
    -------
    wavelength_nm  : (n_wl,)  uniform grid at _HIRES_DWL_NM = 2×10⁻⁴ nm spacing
    temperatures_k : (n_T,)
    sigma_table    : (n_T, n_wl)  cross-section in cm²/molecule
    """
    wl_lo = wl_min_nm - _HIRES_PAD_NM
    wl_hi = wl_max_nm + _HIRES_PAD_NM

    n_wl     = int(np.ceil((wl_hi - wl_lo) / _HIRES_DWL_NM)) + 1
    wl_hires = np.linspace(wl_lo, wl_hi, n_wl)

    # Fine wavenumber grid sized by _HIRES_DWL_NM at the short-λ edge
    nu_lo  = 1e7 / wl_hi
    nu_hi  = 1e7 / wl_lo
    dnu    = 1e7 * _HIRES_DWL_NM / wl_lo ** 2
    n_nu   = int(np.ceil((nu_hi - nu_lo) / dnu)) + 1
    nu_grid = np.linspace(nu_lo, nu_hi, n_nu)

    lines = parse_hitran_par(hitran_file)

    sigma_table = np.zeros((len(temperatures_k), n_wl), dtype=float)
    for k, t_k in enumerate(temperatures_k):
        sigma_nu = _compute_sigma_on_nu_grid(
            nu_grid, lines, t_k, _P_REP_HPA, m_amu, q_exponent, self_vmr,
        )
        # ν is increasing → λ = 1e7/ν is decreasing; sort before interp
        wl_from_nu = 1e7 / nu_grid
        sort_idx   = np.argsort(wl_from_nu)
        sigma_table[k] = np.interp(
            wl_hires, wl_from_nu[sort_idx], sigma_nu[sort_idx],
            left=0.0, right=0.0,
        )

    return {
        "wavelength_nm":  wl_hires,
        "temperatures_k": np.array(temperatures_k, dtype=float),
        "sigma_table":    sigma_table,
    }


def load_o2b_xsec_tables(
    hitran_file: str,
    wl_min_nm: float,
    wl_max_nm: float,
    temperatures_k: tuple[float, ...] = (180.0, 210.0, 240.0, 270.0, 300.0),
) -> dict:
    """Precompute O2 B-band cross-section table (thin wrapper around _load_hitran_xsec_tables)."""
    return _load_hitran_xsec_tables(
        hitran_file, wl_min_nm, wl_max_nm,
        _M_O2_AMU, _Q_EXP_O2, 0.0,   # O2: no self-broadening in the air-broadened table
        temperatures_k,
    )


def load_h2o_xsec_tables(
    hitran_file: str,
    wl_min_nm: float,
    wl_max_nm: float,
    temperatures_k: tuple[float, ...] = (180.0, 210.0, 240.0, 270.0, 300.0),
) -> dict:
    """Precompute H2O cross-section table (thin wrapper around _load_hitran_xsec_tables).

    Uses _H2O_SELF_VMR_REP = 0.005 for the representative self-broadening VMR
    and q_exponent = 1.5 (nonlinear asymmetric-top partition function).
    The actual per-layer tau uses the correct n_h2o profile from GEOS-CF.
    """
    return _load_hitran_xsec_tables(
        hitran_file, wl_min_nm, wl_max_nm,
        _M_H2O_AMU, _Q_EXP_H2O, _H2O_SELF_VMR_REP,
        temperatures_k,
    )


# ---------------------------------------------------------------------------
# High-resolution slit kernel  (shared)
# ---------------------------------------------------------------------------


def _build_hires_slit_kernel(
    hw1e_nm: float,
    shape: float,
    asym_nm: float,
    n_sigma: float = 4.0,
) -> np.ndarray:
    """Super-Gaussian slit kernel sampled at _HIRES_DWL_NM spacing."""
    half_span = max(n_sigma * hw1e_nm, 3.0 * _HIRES_DWL_NM)
    n_side    = max(1, int(np.ceil(half_span / _HIRES_DWL_NM)))
    offsets   = np.arange(-n_side, n_side + 1, dtype=float) * _HIRES_DWL_NM
    return build_super_gaussian_kernel(offsets, hw1e_nm, shape, asym_nm)


def _interpolate_sigma_temperature(
    t_k: float,
    sigma_on_tempo: np.ndarray,
    temperatures_k: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a (n_T, n_wl) σ table to a single temperature."""
    if t_k <= temperatures_k[0]:
        return sigma_on_tempo[0]
    if t_k >= temperatures_k[-1]:
        return sigma_on_tempo[-1]
    hi = int(np.searchsorted(temperatures_k, t_k, side="right"))
    lo = hi - 1
    w  = (t_k - temperatures_k[lo]) / (temperatures_k[hi] - temperatures_k[lo])
    return (1.0 - w) * sigma_on_tempo[lo] + w * sigma_on_tempo[hi]


def _slit_convolve_and_sample(
    sigma_hi: np.ndarray,
    wl_hires: np.ndarray,
    kernel_hi: np.ndarray,
    wl_tempo: np.ndarray,
) -> np.ndarray:
    """Convolve each temperature row of sigma_hi with the slit, sample at wl_tempo.

    Parameters
    ----------
    sigma_hi  : (n_T, n_wl_hi) precomputed high-res σ table
    wl_hires  : (n_wl_hi,) uniform high-res wavelength grid (nm)
    kernel_hi : (n_k,) slit kernel at _HIRES_DWL_NM spacing
    wl_tempo  : (n_wl_tempo,) TEMPO pixel wavelengths (nm)

    Returns
    -------
    sigma_on_tempo : (n_T, n_wl_tempo)
    """
    n_T = sigma_hi.shape[0]
    sigma_on_tempo = np.zeros((n_T, wl_tempo.size), dtype=float)
    for k in range(n_T):
        conv = fftconvolve(sigma_hi[k], kernel_hi, mode="same")
        sigma_on_tempo[k] = np.interp(wl_tempo, wl_hires, conv, left=0.0, right=0.0)
    return sigma_on_tempo


def _slit_convolve_hi(
    sigma_hi: np.ndarray,
    kernel_hi: np.ndarray,
) -> np.ndarray:
    """FFT-convolve each temperature row with the slit kernel on the high-res grid.

    Returns (n_T, n_wl_hi) — does not yet sample at TEMPO wavelengths.
    Call once per unique xtrack; cache the result and sample per-pixel via
    _sample_sigma_at_tempo (cheap np.interp) rather than repeating the FFT.
    """
    return np.vstack([
        fftconvolve(sigma_hi[k], kernel_hi, mode="same")
        for k in range(sigma_hi.shape[0])
    ])


def _sample_sigma_at_tempo(
    conv_hi: np.ndarray,
    wl_hires: np.ndarray,
    wl_tempo: np.ndarray,
) -> np.ndarray:
    """Sample already-convolved high-res sigma rows at TEMPO wavelengths.

    Parameters
    ----------
    conv_hi  : (n_T, n_wl_hi)  output of _slit_convolve_hi
    wl_hires : (n_wl_hi,)      uniform high-res wavelength grid (nm)
    wl_tempo : (n_wl_tempo,)   per-pixel TEMPO wavelengths (nm)

    Returns
    -------
    (n_T, n_wl_tempo)
    """
    out = np.empty((conv_hi.shape[0], wl_tempo.size), dtype=float)
    for k in range(conv_hi.shape[0]):
        out[k] = np.interp(wl_tempo, wl_hires, conv_hi[k], left=0.0, right=0.0)
    return out


# ---------------------------------------------------------------------------
# Main tau computation
# ---------------------------------------------------------------------------


def compute_tau_subset_o2b(
    pixel_df: pd.DataFrame,
    profiles: dict,
    wavelength_diag: dict,
    slit_df: pd.DataFrame,
    scan_start: int,
    xtrack_start: int,
    config: TauO2BConfig | None = None,
) -> dict:
    """Compute wavelength-resolved O2-B and H2O tau for a pixel subset.

    For each valid pixel the function:
      1. Loads precomputed high-res σ tables for O2-B and H2O (cached).
      2. Convolves each temperature row with the per-pixel slit kernel using
         FFT (n_T FFTs per species per pixel, not per layer).
      3. Samples the convolved σ at the pixel's wavecal-corrected TEMPO grid.
      4. Per layer: interpolates σ(T) and multiplies by the layer number
         density (n_O2 or n_H2O) × dz.

    The returned dict extends the wp4_tau.compute_tau_subset keys with two
    extra fields for H2O, so all downstream WP5–WP7 functions work unchanged
    when using tau_eff (O2-B) as the primary tau.

    Extra keys compared with wp4_tau output
    ----------------------------------------
    tau_h2o_vert : (n_pix, n_wl)  H2O vertical optical depth
    tau_h2o_eff  : (n_pix, n_wl)  H2O slant optical depth (× sec_airmass)
    """
    cfg = config or TauO2BConfig()

    _empty: dict = {
        "pixel_index":       np.array([], dtype=int),
        "xtrack":            np.array([], dtype=int),
        "mirror_step":       np.array([], dtype=int),
        "wavelength_nm":     np.empty((0, 0), dtype=float),
        "tau_vert":          np.empty((0, 0), dtype=float),
        "tau_eff":           np.empty((0, 0), dtype=float),
        "tau_rayleigh_vert": np.empty((0, 0), dtype=float),
        "tau_rayleigh_eff":  np.empty((0, 0), dtype=float),
        "tau_h2o_vert":      np.empty((0, 0), dtype=float),
        "tau_h2o_eff":       np.empty((0, 0), dtype=float),
        "tau_band_mean":     np.array([], dtype=float),
        "sec_airmass":       np.array([], dtype=float),
    }
    if len(pixel_df) == 0:
        return _empty

    # Load precomputed cross-section tables (cached after first call)
    xsec_o2b = load_o2b_xsec_tables(
        cfg.hitran_o2b_file,
        cfg.output_wavelength_min_nm,
        cfg.output_wavelength_max_nm,
        cfg.temperatures_k,
    )
    xsec_h2o = load_h2o_xsec_tables(
        cfg.hitran_h2o_file,
        cfg.output_wavelength_min_nm,
        cfg.output_wavelength_max_nm,
        cfg.temperatures_k,
    )

    wl_hires_o2b = xsec_o2b["wavelength_nm"]   # same grid for both (same wl range)
    wl_hires_h2o = xsec_h2o["wavelength_nm"]
    temps_k      = xsec_o2b["temperatures_k"]  # same T grid for both
    sigma_hi_o2b = xsec_o2b["sigma_table"]     # (n_T, n_wl_hi)
    sigma_hi_h2o = xsec_h2o["sigma_table"]     # (n_T, n_wl_hi)

    ch_mask     = np.asarray(wavelength_diag["channel_mask"], dtype=bool)
    lambda_corr = np.asarray(wavelength_diag["lambda_corrected_nm"], dtype=float)
    slit_lookup = slit_df.set_index("xtrack")

    t_k_all       = np.asarray(profiles["t_k"],       dtype=float)
    n_air_cm3_all = np.asarray(profiles["n_air_cm3"], dtype=float)
    n_o2_cm3_all  = np.asarray(profiles["n_o2_cm3"],  dtype=float)
    n_h2o_cm3_all = np.asarray(profiles["n_h2o_cm3"], dtype=float)
    dz_cm_all     = np.asarray(profiles["dz_cm"],     dtype=float)
    valid_profile = np.asarray(profiles["valid_pixel"], dtype=bool)

    pixel_rows: list[int]          = []
    wavelength_rows: list          = []
    tau_vert_rows: list            = []
    tau_eff_rows: list             = []
    tau_rayleigh_vert_rows: list   = []
    tau_rayleigh_eff_rows: list    = []
    tau_h2o_vert_rows: list        = []
    tau_h2o_eff_rows: list         = []
    tau_band_mean_rows: list[float] = []
    sec_airmass_rows: list[float]  = []

    # Pre-compute FFT slit convolution on the high-res grid for each unique xtrack.
    # The kernel depends only on xtrack, not on scan position, so this amortises
    # n_T FFTs per xtrack across every scan row sharing that column.
    # Per-pixel work is then a cheap np.interp sample at the TEMPO wavelengths.
    xtrack_conv_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    n_layers = n_o2_cm3_all.shape[1]
    for xt in pixel_df["xtrack"].unique():
        xt = int(xt)
        if xt not in slit_lookup.index:
            continue
        slit_row  = slit_lookup.loc[xt]
        kernel_hi = _build_hires_slit_kernel(
            hw1e_nm = float(slit_row["kernel_hw1e_nm"]),
            shape   = float(slit_row["kernel_shape"]),
            asym_nm = float(slit_row["kernel_asym_nm"]),
            n_sigma = cfg.n_sigma_kernel,
        )
        xtrack_conv_cache[xt] = (
            _slit_convolve_hi(sigma_hi_o2b, kernel_hi),
            _slit_convolve_hi(sigma_hi_h2o, kernel_hi),
        )

    for i, row in pixel_df.reset_index(drop=True).iterrows():
        if not bool(row["valid_pixel"]):
            continue
        rel_scan = int(row["mirror_step"] - scan_start)
        rel_xt   = int(row["xtrack"]      - xtrack_start)
        if rel_scan < 0 or rel_scan >= lambda_corr.shape[0]:
            continue
        if rel_xt  < 0 or rel_xt  >= lambda_corr.shape[1]:
            continue
        if not valid_profile[i]:
            continue

        wl_nm = lambda_corr[rel_scan, rel_xt, ch_mask]
        wl_nm = wl_nm[np.isfinite(wl_nm)]
        if wl_nm.size < 3:
            continue
        xt = int(row["xtrack"])
        if xt not in xtrack_conv_cache:
            continue
        conv_o2b_hi, conv_h2o_hi = xtrack_conv_cache[xt]

        # Sample the cached convolved sigma at this pixel's wavecal-corrected TEMPO grid
        sigma_o2b_tempo = _sample_sigma_at_tempo(conv_o2b_hi, wl_hires_o2b, wl_nm)
        sigma_h2o_tempo = _sample_sigma_at_tempo(conv_h2o_hi, wl_hires_h2o, wl_nm)

        # Vectorised temperature interpolation over all layers at once.
        # sigma_*_tempo: (n_T, n_wl);  index with (n_layers,) → (n_layers, n_wl)
        t_layers = t_k_all[i]                                              # (n_layers,)
        t_clip   = np.clip(t_layers, temps_k[0], temps_k[-1])
        hi_t     = np.searchsorted(temps_k, t_clip, side="right").clip(1, len(temps_k) - 1)
        lo_t     = hi_t - 1
        span_t   = temps_k[hi_t] - temps_k[lo_t]
        w_t      = np.where(span_t > 0, (t_clip - temps_k[lo_t]) / span_t, 0.0)

        sig_o2b_layers = (1 - w_t[:, None]) * sigma_o2b_tempo[lo_t] + w_t[:, None] * sigma_o2b_tempo[hi_t]
        sig_h2o_layers = (1 - w_t[:, None]) * sigma_h2o_tempo[lo_t] + w_t[:, None] * sigma_h2o_tempo[hi_t]
        np.nan_to_num(sig_o2b_layers, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(sig_h2o_layers, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        col_o2b = n_o2_cm3_all[i] * dz_cm_all[i]   # (n_layers,)
        col_h2o = n_h2o_cm3_all[i] * dz_cm_all[i]
        tau_vert     = np.sum(sig_o2b_layers * col_o2b[:, None], axis=0)  # (n_wl,)
        tau_h2o_vert = np.sum(sig_h2o_layers * col_h2o[:, None], axis=0)

        sec_airmass = (
            1.0 / np.cos(np.deg2rad(float(row["sza_deg"])))
            + 1.0 / np.cos(np.deg2rad(float(row["vza_deg"])))
        )
        tau_eff      = tau_vert    * sec_airmass
        tau_h2o_eff_ = tau_h2o_vert * sec_airmass

        tau_rayleigh_vert = compute_tau_rayleigh(wl_nm, n_air_cm3_all[i, :], dz_cm_all[i, :])
        tau_rayleigh_eff  = tau_rayleigh_vert * sec_airmass

        pixel_rows.append(i)
        wavelength_rows.append(wl_nm)
        tau_vert_rows.append(tau_vert)
        tau_eff_rows.append(tau_eff)
        tau_rayleigh_vert_rows.append(tau_rayleigh_vert)
        tau_rayleigh_eff_rows.append(tau_rayleigh_eff)
        tau_h2o_vert_rows.append(tau_h2o_vert)
        tau_h2o_eff_rows.append(tau_h2o_eff_)
        tau_band_mean_rows.append(float(np.nanmean(tau_eff)))
        sec_airmass_rows.append(float(sec_airmass))

    if not pixel_rows:
        return _empty

    return {
        "pixel_index":       np.array(pixel_rows, dtype=int),
        "xtrack":            pixel_df.iloc[pixel_rows]["xtrack"].to_numpy(dtype=int),
        "mirror_step":       pixel_df.iloc[pixel_rows]["mirror_step"].to_numpy(dtype=int),
        "wavelength_nm":     np.vstack(wavelength_rows),
        "tau_vert":          np.vstack(tau_vert_rows),
        "tau_eff":           np.vstack(tau_eff_rows),
        "tau_rayleigh_vert": np.vstack(tau_rayleigh_vert_rows),
        "tau_rayleigh_eff":  np.vstack(tau_rayleigh_eff_rows),
        "tau_h2o_vert":      np.vstack(tau_h2o_vert_rows),
        "tau_h2o_eff":       np.vstack(tau_h2o_eff_rows),
        "tau_band_mean":     np.array(tau_band_mean_rows, dtype=float),
        "sec_airmass":       np.array(sec_airmass_rows, dtype=float),
    }


# ---------------------------------------------------------------------------
# NetCDF output
# ---------------------------------------------------------------------------


def write_o2b_tau_netcdf(out_path: str, tau_result: dict, pixel_df: pd.DataFrame) -> None:
    """Write O2-B + H2O native tau to an HDF5-backed .nc file."""
    import h5py
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_pix = int(tau_result["tau_eff"].shape[0])
    n_wl  = int(tau_result["tau_eff"].shape[1]) if n_pix else 0

    with h5py.File(out_path, "w") as f:
        f.attrs["title"]       = "TEMPO O2-B native tau subset"
        f.attrs["description"] = "O2 B-band and H2O HITRAN tau on wavecal-corrected TEMPO grid"
        f.create_dataset("pixel",            data=np.arange(n_pix, dtype=np.int32))
        f.create_dataset("spectral_channel", data=np.arange(n_wl,  dtype=np.int32))
        if n_pix:
            pix_sel = pixel_df.iloc[tau_result["pixel_index"]]
            f.create_dataset("mirror_step",       data=tau_result["mirror_step"].astype(np.int32))
            f.create_dataset("xtrack",            data=tau_result["xtrack"].astype(np.int32))
            f.create_dataset("latitude",          data=pix_sel["latitude"].to_numpy(dtype=float))
            f.create_dataset("longitude",         data=pix_sel["longitude"].to_numpy(dtype=float))
            f.create_dataset("sec_airmass",       data=tau_result["sec_airmass"].astype(float))
            f.create_dataset("tau_band_mean",     data=tau_result["tau_band_mean"].astype(float))
            f.create_dataset("wavelength_nm",     data=tau_result["wavelength_nm"].astype(float))
            f.create_dataset("tau_vert",          data=tau_result["tau_vert"].astype(float))
            f.create_dataset("tau_eff",           data=tau_result["tau_eff"].astype(float))
            f.create_dataset("tau_h2o_vert",      data=tau_result["tau_h2o_vert"].astype(float))
            f.create_dataset("tau_h2o_eff",       data=tau_result["tau_h2o_eff"].astype(float))
            if tau_result["tau_rayleigh_vert"].size:
                f.create_dataset("tau_rayleigh_vert", data=tau_result["tau_rayleigh_vert"].astype(float))
                f.create_dataset("tau_rayleigh_eff",  data=tau_result["tau_rayleigh_eff"].astype(float))


# ---------------------------------------------------------------------------
# Vertical debug plot
# ---------------------------------------------------------------------------


def write_o2b_vertical_debug_plot(
    out_path: str,
    pixel_df: pd.DataFrame,
    profiles: dict,
    wavelength_diag: dict,
    slit_df: pd.DataFrame,
    tau_result: dict,
    scan_start: int,
    xtrack_start: int,
    config: TauO2BConfig | None = None,
    debug_row: int = 0,
) -> str:
    """Write a 3-panel per-layer tau debug plot for one pixel.

    Panels: (1) layer τ_O₂B at peak-tau channel, (2) layer τ_H₂O at that
    channel, (3) n_O₂ and n_H₂O number density profiles.
    """
    import matplotlib.pyplot as plt

    cfg = config or TauO2BConfig()
    if tau_result["pixel_index"].size == 0:
        raise ValueError("No tau_result pixels available for debug plotting.")

    xsec_o2b = load_o2b_xsec_tables(
        cfg.hitran_o2b_file, cfg.output_wavelength_min_nm,
        cfg.output_wavelength_max_nm, cfg.temperatures_k,
    )
    xsec_h2o = load_h2o_xsec_tables(
        cfg.hitran_h2o_file, cfg.output_wavelength_min_nm,
        cfg.output_wavelength_max_nm, cfg.temperatures_k,
    )
    temps_k = xsec_o2b["temperatures_k"]

    ch_mask     = np.asarray(wavelength_diag["channel_mask"], dtype=bool)
    lambda_corr = np.asarray(wavelength_diag["lambda_corrected_nm"], dtype=float)
    slit_lookup = slit_df.set_index("xtrack")

    src_row  = int(np.asarray(tau_result["pixel_index"], dtype=int)[debug_row])
    row      = pixel_df.iloc[src_row]
    rel_scan = int(row["mirror_step"] - scan_start)
    rel_xt   = int(row["xtrack"]      - xtrack_start)
    wl_nm    = lambda_corr[rel_scan, rel_xt, ch_mask]
    wl_nm    = wl_nm[np.isfinite(wl_nm)]

    slit_row  = slit_lookup.loc[int(row["xtrack"])]
    kernel_hi = _build_hires_slit_kernel(
        float(slit_row["kernel_hw1e_nm"]), float(slit_row["kernel_shape"]),
        float(slit_row["kernel_asym_nm"]), cfg.n_sigma_kernel,
    )
    sig_o2b_tempo = _slit_convolve_and_sample(
        xsec_o2b["sigma_table"], xsec_o2b["wavelength_nm"], kernel_hi, wl_nm,
    )
    sig_h2o_tempo = _slit_convolve_and_sample(
        xsec_h2o["sigma_table"], xsec_h2o["wavelength_nm"], kernel_hi, wl_nm,
    )

    t_k_all       = np.asarray(profiles["t_k"],       dtype=float)
    n_o2_cm3_all  = np.asarray(profiles["n_o2_cm3"],  dtype=float)
    n_h2o_cm3_all = np.asarray(profiles["n_h2o_cm3"], dtype=float)
    p_mid_all     = np.asarray(profiles["p_mid_pa"],  dtype=float)
    dz_cm_all     = np.asarray(profiles["dz_cm"],     dtype=float)

    n_layers = n_o2_cm3_all.shape[1]
    p_hpa    = p_mid_all[src_row, :] * 0.01

    tau_vert_pix = np.asarray(tau_result["tau_vert"][debug_row], dtype=float)
    peak_ch      = int(np.nanargmax(tau_vert_pix))

    layer_tau_o2b = np.zeros(n_layers)
    layer_tau_h2o = np.zeros(n_layers)
    for lev in range(n_layers):
        t = float(t_k_all[src_row, lev])
        dz = float(dz_cm_all[src_row, lev])
        s_o = np.nan_to_num(_interpolate_sigma_temperature(t, sig_o2b_tempo, temps_k))
        s_h = np.nan_to_num(_interpolate_sigma_temperature(t, sig_h2o_tempo, temps_k))
        layer_tau_o2b[lev] = s_o[peak_ch] * n_o2_cm3_all[src_row, lev] * dz
        layer_tau_h2o[lev] = s_h[peak_ch] * n_h2o_cm3_all[src_row, lev] * dz

    fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    ax1, ax2, ax3 = axes

    ax1.plot(layer_tau_o2b, p_hpa, "b-o", markersize=3)
    ax1.set_xlabel("Layer τ_O₂B (peak λ)")
    ax1.set_ylabel("Pressure (hPa)")
    ax1.set_title(f"O₂-B vertical τ  pixel {src_row}")
    ax1.invert_yaxis(); ax1.set_yscale("log"); ax1.grid(True, alpha=0.4)

    ax2.plot(layer_tau_h2o, p_hpa, "g-o", markersize=3)
    ax2.set_xlabel("Layer τ_H₂O (peak λ)")
    ax2.set_title("H₂O HITRAN vertical τ")
    ax2.set_yscale("log"); ax2.grid(True, alpha=0.4)

    ax3.plot(n_o2_cm3_all[src_row], p_hpa, "b-o", markersize=3, label="n_O₂")
    ax3.plot(n_h2o_cm3_all[src_row], p_hpa, "g-o", markersize=3, label="n_H₂O")
    ax3.set_xlabel("Number density (cm⁻³)")
    ax3.set_title("Number density profiles")
    ax3.legend(fontsize=8); ax3.set_yscale("log"); ax3.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
