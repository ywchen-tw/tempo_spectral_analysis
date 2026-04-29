"""
WP7: Spectral Fitting Module — Cumulant Expansion Tau-Transmittance Modeling

Purpose:
--------
After WP4 computes wavelength-resolved optical depths (tau), this module fits
a cumulant expansion model to the log(transmittance) vs tau relationship:

    ln(T) = -k1*τ + ½k2*τ² - ⅓k3*τ³ + ... + intercept

Fitted kappa coefficients (k1, k2, ...) summarize the spectral shape and are
stored per pixel for validation and retrieval diagnostics.

References:
-----------
- OCO-FP analysis: /Users/yuch8913/programming/oco_fp_analysis/src/oco_fp_spec_anal.py
- Knuteson et al. (2004): Cumulant expansions in radiative transfer
"""

import warnings

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


# ─── Cumulant Expansion Model Functions ────────────────────────────────────
# Each order k corresponds to truncation at the k-th cumulant.
# Formula: ln(T) = -k1*τ + (1/2)*k2*τ² - (1/3)*k3*τ³ + ... + intercept

def log_transmittance_model_1(tau, k1, intercept):
    """Order 2: linear + quadratic cumulants."""
    return -k1 * tau + intercept

def log_transmittance_model_2(tau, k1, k2, intercept):
    """Order 2: linear + quadratic cumulants."""
    return -k1 * tau + 0.5 * k2 * tau**2 + intercept


def log_transmittance_model_3(tau, k1, k2, k3, intercept):
    """Order 3: linear + quadratic + cubic."""
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + intercept)


def log_transmittance_model_4(tau, k1, k2, k3, k4, intercept):
    """Order 4: up to 4th order cumulant."""
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + (1/4) * k4 * tau**4
            + intercept)


def log_transmittance_model_5(tau, k1, k2, k3, k4, k5, intercept):
    """Order 5: up to 5th order cumulant."""
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + (1/4) * k4 * tau**4
            - (1/5) * k5 * tau**5
            + intercept)


def log_transmittance_model_7(tau, k1, k2, k3, k4, k5, k6, k7, intercept):
    """Order 7: up to 7th order cumulant."""
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + (1/4) * k4 * tau**4
            - (1/5) * k5 * tau**5
            + (1/6) * k6 * tau**6
            - (1/7) * k7 * tau**7
            + intercept)


def log_transmittance_model_9(tau, k1, k2, k3, k4, k5, k6, k7, k8, k9, intercept):
    """Order 9: up to 9th order cumulant."""
    return (-k1 * tau
            + 0.5   * k2 * tau**2
            - (1/3) * k3 * tau**3
            + (1/4) * k4 * tau**4
            - (1/5) * k5 * tau**5
            + (1/6) * k6 * tau**6
            - (1/7) * k7 * tau**7
            + (1/8) * k8 * tau**8
            - (1/9) * k9 * tau**9
            + intercept)


# Map fit_order integer → model function (start with order 2)
LOG_TRANSMITTANCE_MODELS = {
    1: log_transmittance_model_1,
    2: log_transmittance_model_2,
    3: log_transmittance_model_3,
    4: log_transmittance_model_4,
    5: log_transmittance_model_5,
    7: log_transmittance_model_7,
    9: log_transmittance_model_9,
}


# ─── Ring Effect — Rotational Raman Scattering (Chance & Spurr 1997) ──────
#
# Ring spectrum: R(λ) = ln[I_Raman(λ)] − ln[I₀(λ)]
#
# I_Raman(λ) = weighted average of the solar spectrum at incident wavelengths
# that Raman-scatter to λ, summed over all N2 + O2 rotational transitions.
#
# References:
#   Chance & Spurr (1997), Appl. Opt. 36(21):5224-5230
#   Chance & Martin (2017), Spectroscopy & Radiative Transfer, OUP
#     companion data: https://global.oup.com/booksites/content/9780199662104/data/
#   skdoas Ring effect API: https://arg.usask.ca/docs/skdoas/ring_effect.html
#
# Implementation note — analytic vs. tabulated cross-sections:
#   skdoas (RingSpectrum class) reads pre-tabulated N2/O2 rotational Raman
#   cross-sections from the Chance & Martin (2017) OUP companion tables
#   (Tables 1–4: partition functions + cross-sections at fixed temperature).
#   Those files were NOT downloaded here.  Instead, the cross-sections are
#   derived analytically at runtime from spectroscopic constants (B0, D0, γ²)
#   using the same Chance & Spurr (1997) formulas that underlie the tables.
#   The two approaches give identical line positions and equivalent strengths;
#   the only neglected terms are higher-order Herman-Wallis centrifugal
#   corrections, which are negligible for DOAS fitting purposes.
#   To switch to the tabulated values, download Tables 2 & 4 from the OUP
#   site, parse them into (shift_cm1, weight) arrays, and pass them directly
#   to _compute_ring_spectrum in place of the _raman_lines() output.

_HC_OVER_KB = 1.4387769   # h·c / k_B [cm·K]

# N2 X¹Σg⁺ spectroscopic constants (Chance & Spurr 1997, Table A1)
# B0, D0: rotational and centrifugal distortion constants [cm⁻¹]
# gamma2: square of polarizability anisotropy [Å⁶] (Penney et al. 1974)
# Nuclear spin ¹⁴N: I=1 (boson) → symmetric nuclear states: g_even=6, g_odd=3
_N2_RAMAN_PARAMS = dict(
    B0=1.99824, D0=5.76e-6, gamma2=0.510,
    g_even=6, g_odd=3, J_min=0, J_max=30,
)

# O2 X³Σg⁻ spectroscopic constants (Chance & Spurr 1997, Table A1)
# Nuclear spin ¹⁶O: I=0 (boson, Pauli exclusion on X³Σg⁻) → only odd J exists
# g_even=0 causes even-J transitions to be skipped in _raman_lines()
_O2_RAMAN_PARAMS = dict(
    B0=1.43768, D0=4.84e-6, gamma2=1.27,
    g_even=0, g_odd=1, J_min=1, J_max=40,
)


def _raman_lines(params: dict, T_ref: float):
    """Compute rotational Raman line shifts and relative weights for one molecule.

    Implements Chance & Spurr (1997) Eqs. (A1)–(A4) for both Stokes (S-branch,
    ΔJ = +2) and anti-Stokes (O-branch, ΔJ = −2) transitions.

    Parameters
    ----------
    params : dict
        Molecular constants (B0, D0, gamma2, g_even, g_odd, J_min, J_max).
    T_ref : float
        Reference temperature [K] for Boltzmann population weights.

    Returns
    -------
    shifts : ndarray [cm⁻¹]
        Wavenumber shifts; positive = Stokes (energy loss by photon),
        negative = anti-Stokes (energy gain).
    weights : ndarray
        Relative scattering weight ∝ Placzek-Teller coeff × population × γ².
    """
    B0, D0, gamma2 = params["B0"], params["D0"], params["gamma2"]
    g_even, g_odd = params["g_even"], params["g_odd"]
    J_min, J_max = params["J_min"], params["J_max"]

    # Partition function
    Q = 0.0
    for J in range(J_min, J_max + 1):
        g_J = g_even if J % 2 == 0 else g_odd
        if g_J == 0:
            continue
        Q += g_J * (2 * J + 1) * np.exp(-_HC_OVER_KB * B0 * J * (J + 1) / T_ref)

    shifts, weights = [], []
    for J in range(J_min, J_max + 1):
        g_J = g_even if J % 2 == 0 else g_odd
        if g_J == 0:
            continue
        pop = g_J * (2 * J + 1) * np.exp(-_HC_OVER_KB * B0 * J * (J + 1) / T_ref) / Q

        # Stokes S-branch: J → J+2
        b_S = 3 * (J + 1) * (J + 2) / (2 * (2 * J + 1) * (2 * J + 3))
        shift_S = 4 * B0 * (J + 1.5) - 8 * D0 * (J + 1.5) ** 3
        shifts.append(shift_S)
        weights.append(b_S * pop * gamma2)

        # Anti-Stokes O-branch: J → J-2  (requires J ≥ 2)
        if J >= 2:
            b_A = 3 * J * (J - 1) / (2 * (2 * J - 1) * (2 * J + 1))
            shift_A = -(4 * B0 * (J - 0.5) - 8 * D0 * (J - 0.5) ** 3)
            shifts.append(shift_A)
            weights.append(b_A * pop * gamma2)

    return np.array(shifts), np.array(weights)


def _compute_ring_spectrum(
    irr_wl_nm: np.ndarray,
    irr_1d: np.ndarray,
    out_wl_nm: np.ndarray,
    T_ref: float = 250.0,
) -> np.ndarray:
    """Core Ring spectrum computation on an arbitrary output wavelength grid.

    R(λ) = ln[I_Raman(λ)] − ln[I₀(λ)]  (mean-subtracted)

    where I_Raman(λ) = Σ_J w_J · I₀(λ_inc,J) / Σ_J w_J  sums over all N2
    and O2 Stokes + anti-Stokes lines.  λ_inc,J is the incident wavelength
    whose photons Raman-scatter to the detection wavelength λ.

    Parameters
    ----------
    irr_wl_nm, irr_1d : sorted, valid irradiance grid (no NaN, irr > 0).
    out_wl_nm : detection wavelength grid [nm].
    T_ref : rotational temperature [K] for Boltzmann populations (default 250 K).

    Returns
    -------
    ring : ndarray, same shape as out_wl_nm
        NaN where out_wl_nm falls outside the irradiance grid.
    """
    # Combined N2 + O2 Raman lines
    sh_n2, wt_n2 = _raman_lines(_N2_RAMAN_PARAMS, T_ref)
    sh_o2, wt_o2 = _raman_lines(_O2_RAMAN_PARAMS, T_ref)
    shifts = np.concatenate([sh_n2, sh_o2])   # [cm⁻¹], positive=Stokes
    weights = np.concatenate([wt_n2, wt_o2])
    W = weights.sum()

    # At each detection wavelength ν̃_det, the incident wavenumber for line j is:
    #   ν̃_inc = ν̃_det + shift_j   (positive shift → shorter λ_inc, i.e. Stokes)
    nu_det = 1e7 / out_wl_nm                          # shape (n_out,)
    nu_inc = nu_det[:, None] + shifts[None, :]         # (n_out, n_lines)
    lam_inc = 1e7 / nu_inc                             # (n_out, n_lines) [nm]

    # Interpolate irradiance at incident wavelengths (vectorised)
    I_inc = np.interp(
        lam_inc.ravel(), irr_wl_nm, irr_1d, left=np.nan, right=np.nan,
    ).reshape(lam_inc.shape)                           # (n_out, n_lines)

    # Raman-weighted average solar spectrum at each detection wavelength
    I_raman = np.nansum(weights[None, :] * I_inc, axis=1) / W  # (n_out,)

    # Reference irradiance at detection wavelengths
    I0 = np.interp(out_wl_nm, irr_wl_nm, irr_1d, left=np.nan, right=np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ring = np.log(I_raman) - np.log(I0)

    finite = np.isfinite(ring)
    if finite.sum() >= 2:
        ring = np.where(finite, ring - np.nanmean(ring[finite]), np.nan)

    return ring


def build_ring_template_from_irr(
    irr_file: str,
    rep_wavelength_nm: np.ndarray,
    band: str = "band_290_490_nm",
    T_ref: float = 250.0,
) -> np.ndarray:
    """Build a Ring-effect spectrum template on the fitting wavelength grid.

    Computes the rotational Raman scattering Ring spectrum following
    Chance & Spurr (1997) using N2 and O2 molecular constants and the
    TEMPO solar irradiance as the incident solar reference.

    R(λ) = ln[I_Raman(λ)] − ln[I₀(λ)]  (mean-subtracted)

    Parameters
    ----------
    irr_file : str
        Path to TEMPO IRR L1 HDF5 file.
    rep_wavelength_nm : 1-D array
        Output wavelength grid [nm].
    band : str
        HDF5 group name (default: "band_290_490_nm").
    T_ref : float
        Rotational temperature [K] for Boltzmann populations (default 250 K).

    Returns
    -------
    ring : ndarray, same shape as rep_wavelength_nm
        NaN where outside the irradiance grid or computation fails.
    """
    rep_wavelength_nm = np.asarray(rep_wavelength_nm, dtype=float)
    if rep_wavelength_nm.ndim != 1 or rep_wavelength_nm.size < 5:
        return np.full_like(rep_wavelength_nm, np.nan, dtype=float)

    with h5py.File(irr_file, "r") as f:
        irr = np.asarray(f[f"{band}/irradiance"], dtype=float)
        nominal_wl = np.asarray(f[f"{band}/nominal_wavelength"], dtype=float)

    wl_1d = np.nanmedian(nominal_wl, axis=0)
    irr_1d = np.nanmedian(irr[0, :, :], axis=0)
    valid = np.isfinite(wl_1d) & np.isfinite(irr_1d) & (irr_1d > 0)
    if np.count_nonzero(valid) < 11:
        return np.full_like(rep_wavelength_nm, np.nan, dtype=float)

    # Sort so np.interp works correctly
    sort_idx = np.argsort(wl_1d[valid])
    wl_v = wl_1d[valid][sort_idx]
    irr_v = irr_1d[valid][sort_idx]

    return _compute_ring_spectrum(wl_v, irr_v, rep_wavelength_nm, T_ref=T_ref)


def compute_ring_optical_depth(
    tau_result: dict,
    irr_file: str,
    band: str = "band_290_490_nm",
    T_ref: float = 250.0,
) -> np.ndarray:
    """Compute Ring-effect optical depth for every pixel using rotational Raman scattering.

    Reads the TEMPO irradiance once, then for each pixel interpolates the
    Ring spectrum R(λ) = ln[I_Raman(λ)] − ln[I₀(λ)] onto the pixel's
    wavelength grid.  Ready to store as tau_result["ring_basis"].

    Physical basis: rotational Raman scattering of N2 and O2 partially fills
    Fraunhofer absorption lines, creating a spectrally structured additive term
    in ln(T).  Computed per Chance & Spurr (1997), Appl. Opt. 36(21):5224-5230.

    Parameters
    ----------
    tau_result : dict
        Output of compute_tau_subset; provides per-pixel wavelength grids.
    irr_file : str
        Path to TEMPO IRR L1 HDF5 file.
    band : str
        HDF5 group name (default: "band_290_490_nm").
    T_ref : float
        Rotational temperature [K] for Boltzmann populations (default 250 K).

    Returns
    -------
    ring_tau : ndarray, shape (n_pix, n_ch)
        Ring spectrum R(λ) per pixel and channel [dimensionless, mean-subtracted].
        NaN where wavelength is outside the irradiance grid.
    """
    tau_eff = np.asarray(tau_result["tau_eff"], dtype=float)
    if tau_eff.ndim != 2:
        return np.full_like(tau_eff, np.nan, dtype=float)
    n_pix, n_ch = tau_eff.shape
    ring_tau = np.full((n_pix, n_ch), np.nan)

    # Read irradiance once
    with h5py.File(irr_file, "r") as f:
        irr = np.asarray(f[f"{band}/irradiance"], dtype=float)
        nominal_wl = np.asarray(f[f"{band}/nominal_wavelength"], dtype=float)

    wl_1d = np.nanmedian(nominal_wl, axis=0)
    irr_1d = np.nanmedian(irr[0, :, :], axis=0)
    valid = np.isfinite(wl_1d) & np.isfinite(irr_1d) & (irr_1d > 0)
    if np.count_nonzero(valid) < 11:
        return ring_tau

    sort_idx = np.argsort(wl_1d[valid])
    wl_v = wl_1d[valid][sort_idx]
    irr_v = irr_1d[valid][sort_idx]

    # Pre-compute Raman lines once — shared across all pixels
    sh_n2, wt_n2 = _raman_lines(_N2_RAMAN_PARAMS, T_ref)
    sh_o2, wt_o2 = _raman_lines(_O2_RAMAN_PARAMS, T_ref)
    shifts = np.concatenate([sh_n2, sh_o2])
    weights = np.concatenate([wt_n2, wt_o2])
    W = weights.sum()

    for pix_idx in range(n_pix):
        wl = np.asarray(tau_result["wavelength_nm"][pix_idx], dtype=float)
        finite_wl = np.isfinite(wl)
        if finite_wl.sum() < 5:
            continue

        out_wl = wl[finite_wl]
        nu_det = 1e7 / out_wl
        nu_inc = nu_det[:, None] + shifts[None, :]
        lam_inc = 1e7 / nu_inc

        I_inc = np.interp(
            lam_inc.ravel(), wl_v, irr_v, left=np.nan, right=np.nan,
        ).reshape(lam_inc.shape)
        I_raman = np.nansum(weights[None, :] * I_inc, axis=1) / W
        I0 = np.interp(out_wl, wl_v, irr_v, left=np.nan, right=np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            ring = np.log(I_raman) - np.log(I0)

        finite_r = np.isfinite(ring)
        if finite_r.sum() >= 2:
            ring = np.where(finite_r, ring - np.nanmean(ring[finite_r]), np.nan)

        ring_tau[pix_idx, finite_wl] = ring

    return ring_tau


# ─── Transmittance Calculation ─────────────────────────────────────────────

def compute_transmittance(radiance, toa_solar_irradiance):
    """
    Compute transmittance from radiance and TOA solar irradiance.

    Parameters
    ----------
    radiance : float or array
        Spectral radiance (in standard L1B units).
    toa_solar_irradiance : float or array, same shape as radiance
        TOA solar irradiance (matching L1B scaling).

    Returns
    -------
    transmittance : float or array
        T = (radiance / toa_sol_irr) * π, masked where T > 1 or invalid.
        Note: π factor converts from radiance to irradiance ratio convention.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        T = radiance / toa_solar_irradiance * np.pi
    T = np.where(T > 1.0, np.nan, T)
    return T


def extract_observed_lnT(
    tau_result: dict,
    pixel_table: pd.DataFrame,
    rad_file: str,
    irr_file: str,
    wavelength_diag: dict,
    scan_start: int = 0,
    xtrack_start: int = 0,
    band: str = "band_290_490_nm",
) -> np.ndarray:
    """Compute observed ln(T) = ln(π * RAD / (IRR * cos(SZA))) per pixel and channel.

    Uses the same channel mask and per-pixel finite-wavelength filter as WP4,
    so the returned array aligns exactly with tau_result["tau_eff"].

    Parameters
    ----------
    tau_result : dict
        Output of ``compute_tau_subset``.
    pixel_table : DataFrame
        WP1 pixel metadata (must contain mirror_step, xtrack, sza_deg).
    rad_file : str
        Path to the TEMPO RAD L1 file (HDF5/netCDF).
    irr_file : str
        Path to the TEMPO IRR L1 file (HDF5/netCDF).
    wavelength_diag : dict
        Output of WP1 ``build_pixel_table`` containing
        ``channel_mask`` and ``lambda_corrected_nm``.
    scan_start : int
        First mirror step of the wavelength_diag subset (to convert absolute
        mirror_step to a relative index into lambda_corrected_nm).
    xtrack_start : int
        First xtrack of the wavelength_diag subset.
    band : str
        HDF5 group name for the spectral band (default: ``band_290_490_nm``).

    Returns
    -------
    obs_ln_T : ndarray, shape (n_pix, n_ch)
        NaN where T <= 0, fill value, or wavelength is undefined.
    """
    n_pix = int(tau_result["tau_eff"].shape[0]) if tau_result["tau_eff"].ndim == 2 else 0
    n_ch = int(tau_result["tau_eff"].shape[1]) if n_pix > 0 else 0
    obs_ln_T = np.full((n_pix, n_ch), np.nan)
    if n_pix == 0:
        return obs_ln_T

    ch_mask = np.asarray(wavelength_diag["channel_mask"], dtype=bool)
    lambda_corr = np.asarray(wavelength_diag["lambda_corrected_nm"], dtype=float)

    with h5py.File(rad_file, "r") as rf:
        radiance = np.asarray(rf[f"{band}/radiance"], dtype=float)
    with h5py.File(irr_file, "r") as irf:
        irradiance = np.asarray(irf[f"{band}/irradiance"], dtype=float)  # (1, xt, ch)

    pixel_idx_arr = np.asarray(tau_result["pixel_index"], dtype=int)
    pixel_subset = pixel_table.iloc[pixel_idx_arr].reset_index(drop=True)

    for pix_idx in range(n_pix):
        row = pixel_subset.iloc[pix_idx]
        ms = int(row["mirror_step"])
        xt = int(row["xtrack"])
        sza = float(row["sza_deg"])
        mu0 = float(np.cos(np.deg2rad(sza)))
        if mu0 <= 0:
            continue

        rel_scan = ms - scan_start
        rel_xt = xt - xtrack_start
        if rel_scan < 0 or rel_scan >= lambda_corr.shape[0]:
            continue
        if rel_xt < 0 or rel_xt >= lambda_corr.shape[1]:
            continue
        if ms < 0 or ms >= radiance.shape[0]:
            continue
        if xt < 0 or xt >= radiance.shape[1]:
            continue

        # Apply the same channel mask and finite-wavelength filter used in WP4
        wl_all = lambda_corr[rel_scan, rel_xt, ch_mask]
        finite = np.isfinite(wl_all)
        if finite.sum() < 3:
            continue

        rad_sel = radiance[ms, xt, ch_mask][finite]
        irr_sel = irradiance[0, xt, ch_mask][finite]

        # Filter fill values (TEMPO fill ~9.97e36) and compute T = π·RAD / (IRR·cos(SZA))
        valid = (
            np.isfinite(rad_sel) & np.isfinite(irr_sel)
            & (rad_sel > 0) & (irr_sel > 0)
            & (rad_sel < 1e30) & (irr_sel < 1e30)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.where(valid, rad_sel * np.pi / (irr_sel * mu0), np.nan)
        lnT = np.where(np.isfinite(T) & (T > 0), np.log(T), np.nan)

        n_finite = int(finite.sum())
        if n_finite == n_ch:
            obs_ln_T[pix_idx, :] = lnT
        else:
            # Trim or pad to match n_ch (should rarely differ)
            n_copy = min(n_finite, n_ch)
            obs_ln_T[pix_idx, :n_copy] = lnT[:n_copy]

    return obs_ln_T


# ─── Fitting Engine ───────────────────────────────────────────────────────

def _build_composite_model(fit_order: int, use_ring: bool, poly_order: int):
    """Return a composite model closure and its total parameter count.

    The model:
        ln(T) = Σ_{n=1}^{fit_order} (-1)^n/n · k_n · τ^n
                + [alpha_ring · ring]
                + polyval(poly_c, wl_norm)
                + intercept

    xdata tuple passed to curve_fit:
        (tau, ring, wl_norm)  when use_ring is True
        (tau, wl_norm)        when use_ring is False

    wl_norm = (wavelength - wl_center) / wl_scale  (caller's responsibility).

    Parameter order:
        k1, …, k_{fit_order}, [alpha_ring], p_{poly_order}, …, p_0, intercept
    """
    n_poly = poly_order + 1

    def model(X, *params):
        tau = X[0]
        idx = 0

        absorption = np.zeros_like(tau, dtype=float)
        for n in range(1, fit_order + 1):
            absorption += (-1) ** n / n * params[idx] * tau ** n
            idx += 1

        if use_ring:
            ring_term = params[idx] * X[1]
            wl_norm = X[2]
            idx += 1
        else:
            ring_term = 0.0
            wl_norm = X[1]

        baseline = np.polyval(np.asarray(params[idx: idx + n_poly]), wl_norm)
        idx += n_poly
        intercept = params[idx]

        return absorption + ring_term + baseline + intercept

    n_params = fit_order + (1 if use_ring else 0) + n_poly + 1
    return model, n_params


def _eval_absorption_only(tau: np.ndarray, kappas) -> np.ndarray:
    """Evaluate cumulant absorption terms only (no ring / baseline / intercept)."""
    result = np.zeros_like(tau, dtype=float)
    for n, k in enumerate(kappas, start=1):
        result += (-1) ** n / n * float(k) * tau ** n
    return result


def _fit_simple_model(tau: np.ndarray, ln_T: np.ndarray, fit_order: int):
    """Fit a simple cumulant model using LOG_TRANSMITTANCE_MODELS (no ring / polynomial).

    Parameters
    ----------
    tau : 1-D array
        Effective optical depth (caller ensures no NaN, tau > 0).
    ln_T : 1-D array
        Observed log-transmittance, same length.
    fit_order : int
        Must be a key in LOG_TRANSMITTANCE_MODELS (2, 3, 4, 5, 7, 9).

    Returns
    -------
    kappas : ndarray, shape (fit_order,)
    intercept : float

    Raises
    ------
    KeyError
        If fit_order is not in LOG_TRANSMITTANCE_MODELS.
    RuntimeError
        If curve_fit does not converge.
    """
    if fit_order not in LOG_TRANSMITTANCE_MODELS:
        raise KeyError(
            f"fit_order {fit_order} not in LOG_TRANSMITTANCE_MODELS; "
            f"available orders: {sorted(LOG_TRANSMITTANCE_MODELS)}"
        )

    # Sort by tau and SG-smooth (mirrors fit_spectral_model)
    sort_idx = np.argsort(tau)
    tau_s = tau[sort_idx]
    ln_T_s = ln_T[sort_idx]

    n = len(tau_s)
    window_len = min(51, n if n % 2 == 1 else n - 1)
    if window_len < 5:
        window_len = min(5, n)
    sg_polyorder = min(3, window_len - 2)
    ln_T_smooth = savgol_filter(ln_T_s, window_length=window_len, polyorder=sg_polyorder)

    model_func = LOG_TRANSMITTANCE_MODELS[fit_order]
    p0 = [1.0] + [0.1] * (fit_order - 1) + [0.0]   # k1 ~ 1, rest small, intercept ~ 0
    # k1 and k2 must be positive; higher kappas and intercept are unconstrained
    n_constrained = min(2, fit_order)
    lb = [0.0] * n_constrained + [-np.inf] * (fit_order - n_constrained) + [-np.inf]
    ub = [np.inf] * (fit_order + 1)
    popt, _ = curve_fit(model_func, tau_s, ln_T_smooth, p0=p0,
                        bounds=(lb, ub), maxfev=10000)
    return popt[: fit_order], float(popt[fit_order])


def fit_spectral_model(tau, ln_T, fit_order, wavelength, ring_basis=None, poly_order=2):
    """Fit the composite spectral model to log(transmittance) vs optical depth.

    Model (see _build_composite_model for full expression):
        ln(T) = absorption(tau, k1…k_order)
                + alpha_ring · ring(λ)
                + polynomial(λ, poly_order)
                + intercept

    The absorption term uses cumulant expansion of arbitrary order.
    The Ring term is included only when ring_basis is supplied.
    The polynomial baseline captures smooth Rayleigh/aerosol continuum.
    Wavelength is normalized internally: wl_norm = (λ - median(λ)) / range(λ).

    Parameters
    ----------
    tau : 1-D array [n_channels]
        O2-O2 effective optical depth (no NaN expected; caller filters).
    ln_T : 1-D array [n_channels]
        log(transmittance) corresponding to tau.
    fit_order : int
        Cumulant truncation order (any positive integer).
    wavelength : 1-D array [n_channels]
        Channel centre wavelengths in nm, same ordering as tau / ln_T.
    ring_basis : 1-D array [n_channels], optional
        Normalized Ring-effect template (from compute_ring_optical_depth).
    poly_order : int
        Degree of the wavelength polynomial baseline (default 2).

    Returns
    -------
    popt : 1-D array
        [k1, …, k_order, (alpha_ring,) p_N, …, p_0, intercept]
    wl_center : float
        Wavelength normalization centre (median of input wavelengths).
    wl_scale : float
        Wavelength normalization scale (peak-to-peak range, or 1 if flat).

    Raises
    ------
    ValueError
        If fit_order < 1 or insufficient data points.
    RuntimeError
        If curve_fit does not converge.
    """
    if fit_order < 1:
        raise ValueError(f"fit_order must be >= 1, got {fit_order}")

    use_ring = ring_basis is not None

    sort_idx = np.argsort(tau)
    tau_s = tau[sort_idx]
    ln_T_s = ln_T[sort_idx]
    wl_s = np.asarray(wavelength, dtype=float)[sort_idx]

    # Savitzky-Golay smoothing of ln_T before fitting
    window_len = min(51, len(tau_s) if len(tau_s) % 2 == 1 else len(tau_s) - 1)
    if window_len < 5:
        window_len = min(5, len(tau_s))
    sg_polyorder = min(3, window_len - 2)
    ln_T_smooth = savgol_filter(ln_T_s, window_length=window_len, polyorder=sg_polyorder)

    # Normalize wavelength so polynomial coefficients are O(1)
    wl_center = float(np.median(wl_s))
    wl_range = float(np.ptp(wl_s))
    wl_scale = wl_range if wl_range > 0 else 1.0
    wl_norm = (wl_s - wl_center) / wl_scale

    model_func, n_params = _build_composite_model(fit_order, use_ring, poly_order)
    if len(tau_s) < n_params + 2:
        raise ValueError(
            f"Insufficient data: {len(tau_s)} points < {n_params + 2} required"
        )

    p0 = [1.0] + [0.1] * (fit_order - 1)   # kappas: k1 ~ 1, rest small
    if use_ring:
        p0.append(0.0)                       # alpha_ring
    p0.extend([0.0] * (poly_order + 1))      # polynomial coefficients
    p0.append(0.0)                           # intercept

    # k1 and k2 must be positive; ring, polynomial, higher kappas, intercept unconstrained
    n_constrained = min(2, fit_order)
    lb = [0.0] * n_constrained + [-np.inf] * (fit_order - n_constrained)
    if use_ring:
        lb.append(-np.inf)
    lb.extend([-np.inf] * (poly_order + 1 + 1))   # poly coeffs + intercept
    ub = [np.inf] * n_params

    if use_ring:
        ring_s = np.asarray(ring_basis, dtype=float)[sort_idx]
        xdata = (tau_s, ring_s, wl_norm)
    else:
        xdata = (tau_s, wl_norm)

    popt, _ = curve_fit(model_func, xdata, ln_T_smooth, p0=p0,
                        bounds=(lb, ub), maxfev=10000)
    return popt, wl_center, wl_scale


def fit_pixel_ensemble(
    tau_result,
    pixel_table,
    fit_order=2,
    obs_ln_T=None,
    poly_order=2,
    is_ocean=None,
    ocean_poly_order=None,
    fit_mode="composite",
    tau_simple=None,
):
    """Fit the composite spectral model for all pixels in a WP4 output ensemble.

    Three fitting modes are supported via ``fit_mode``:

    ``'composite'`` (default)
        Full model: cumulant absorption + Ring-effect term + polynomial baseline.
        Requires ``obs_ln_T`` and uses ``tau_result['tau_eff']`` as the O2-O2
        optical depth.  Ring and polynomial terms are optional.

    ``'simple'`` with ``tau_simple`` = total gas tau  →  **mode 3 (total gas tau)**
        Uses ``LOG_TRANSMITTANCE_MODELS[fit_order]`` (pure cumulant, no Ring /
        polynomial) fitted against total gas optical depth:
        τ_gas = τ(O₂-O₂) + τ(NO₂) + τ(O₃) + τ(H₂O).
        ``tau_simple`` must have shape (n_pix, n_ch).

    ``'simple'`` with ``tau_simple`` = total tau  →  **mode 2 (total tau)**
        Same simple cumulant model, but fitted against total optical depth
        including all components: τ = τ(O₂-O₂) + τ(NO₂) + τ(O₃) + τ(H₂O) +
        τ(Rayleigh) + τ(Ring).  ``tau_simple`` must have shape (n_pix, n_ch).

    Parameters
    ----------
    tau_result : dict from wp4_tau.compute_tau_subset()
        Keys: 'pixel_index', 'wavelength_nm', 'tau_eff', and optionally 'ring_basis'.
    pixel_table : DataFrame
        WP1 pixel metadata with columns: 'pixel_index', 'longitude', 'latitude', etc.
    fit_order : int
        Cumulant truncation order.  For ``fit_mode='simple'`` must be a key in
        ``LOG_TRANSMITTANCE_MODELS`` (2, 3, 4, 5, 7, 9).
    obs_ln_T : ndarray, shape (n_pix, n_ch)
        Observed log-transmittance from ``extract_observed_lnT``.
    poly_order : int
        Degree of the wavelength polynomial baseline for land pixels (default 2).
        Only used in ``fit_mode='composite'``.
    is_ocean : bool array of shape (n_pix,), optional
        True for water/coastal pixels; they use ``ocean_poly_order`` instead of
        ``poly_order`` to absorb broad water Raman signal.
        Only used in ``fit_mode='composite'``.
    ocean_poly_order : int, optional
        Polynomial degree for ocean pixels (default: ``poly_order + 2``).
    fit_mode : {'composite', 'simple'}
        Selects the model family (see above).
    tau_simple : ndarray, shape (n_pix, n_ch), optional
        Effective optical depth used when ``fit_mode='simple'``.
        **Mode 2**: τ(O₂-O₂) + τ(NO₂) + τ(O₃) + τ(H₂O) + τ(Rayleigh) + τ(Ring).
        **Mode 3**: τ(O₂-O₂) + τ(NO₂) + τ(O₃) + τ(H₂O) (gas only).
        If None, falls back to ``tau_result['tau_eff']`` (O₂-O₂ only).

    Returns
    -------
    fitting_result : dict
        'pixel_index'    : int   [npix]
        'lon'            : float [npix]
        'lat'            : float [npix]
        'fit_mode'       : str               — 'composite' or 'simple'
        'kappas'         : float [npix, fit_order]       — k1 … k_order
        'ring_coefficient': float [npix]                 — alpha_ring (NaN in simple mode)
        'poly_coeffs'    : float [npix, max_poly_order+1]— NaN in simple mode
        'poly_wl_center' : float [npix]                  — NaN in simple mode
        'poly_wl_scale'  : float [npix]                  — NaN in simple mode
        'is_ocean'       : bool  [npix]
        'intercept'      : float [npix]
        'n_valid_channels': int  [npix]
        'fit_success'    : bool  [npix]
    """
    if obs_ln_T is None:
        raise ValueError("obs_ln_T must be provided for meaningful fitting.")
    if fit_mode not in ("composite", "simple"):
        raise ValueError(f"fit_mode must be 'composite' or 'simple', got {fit_mode!r}")
    if fit_mode == "simple" and fit_order not in LOG_TRANSMITTANCE_MODELS:
        raise KeyError(
            f"fit_order {fit_order} not in LOG_TRANSMITTANCE_MODELS for simple mode; "
            f"available: {sorted(LOG_TRANSMITTANCE_MODELS)}"
        )

    npix = len(tau_result["tau_eff"])

    # Resolve effective poly orders (composite mode only)
    if ocean_poly_order is None:
        ocean_poly_order = poly_order + 2
    max_poly_order = max(poly_order, ocean_poly_order)

    ocean_mask = np.zeros(npix, dtype=bool)
    if is_ocean is not None:
        ocean_mask[:] = np.asarray(is_ocean, dtype=bool)[:npix]

    kappas = np.full((npix, fit_order), np.nan)
    ring_coeff = np.full(npix, np.nan)
    poly_coeffs = np.full((npix, max_poly_order + 1), np.nan)
    poly_wl_center = np.full(npix, np.nan)
    poly_wl_scale = np.full(npix, np.nan)
    intercept = np.full(npix, np.nan)
    n_valid = np.full(npix, 0, dtype=int)
    fit_success = np.full(npix, False, dtype=bool)

    pixel_idx_arr = np.asarray(tau_result["pixel_index"], dtype=int)
    pixel_subset = pixel_table.iloc[pixel_idx_arr].reset_index(drop=True)

    for pix_idx in range(npix):
        wl = np.asarray(tau_result["wavelength_nm"][pix_idx], dtype=float)
        ln_T_row = obs_ln_T[pix_idx, :]

        if fit_mode == "simple":
            # Use caller-supplied total tau, or fall back to gas-only tau
            if tau_simple is not None:
                tau_eff = np.asarray(tau_simple[pix_idx], dtype=float)
            else:
                tau_eff = np.asarray(tau_result["tau_eff"][pix_idx], dtype=float)

            valid_mask = ~np.isnan(tau_eff) & (tau_eff > 0) & np.isfinite(ln_T_row)
            n_valid[pix_idx] = int(np.sum(valid_mask))
            if n_valid[pix_idx] < fit_order + 2:
                continue

            try:
                kaps, intc = _fit_simple_model(
                    tau_eff[valid_mask], ln_T_row[valid_mask], fit_order
                )
                kappas[pix_idx, :] = kaps
                intercept[pix_idx] = intc
                fit_success[pix_idx] = True
            except (RuntimeError, ValueError, KeyError):
                pass

        else:  # composite
            pix_poly = ocean_poly_order if ocean_mask[pix_idx] else poly_order
            tau_eff = np.asarray(tau_result["tau_eff"][pix_idx], dtype=float)
            ring_basis = None
            if "ring_basis" in tau_result:
                ring_basis = np.asarray(tau_result["ring_basis"][pix_idx], dtype=float)

            valid_mask = ~(np.isnan(wl) | np.isnan(tau_eff)) & (tau_eff > 0)
            valid_mask = valid_mask & np.isfinite(ln_T_row)
            if ring_basis is not None:
                valid_mask = valid_mask & np.isfinite(ring_basis)

            n_valid[pix_idx] = int(np.sum(valid_mask))
            if n_valid[pix_idx] < fit_order + pix_poly + 3:
                continue

            tau_fit = tau_eff[valid_mask]
            wl_fit = wl[valid_mask]
            ln_T_fit = ln_T_row[valid_mask]
            ring_fit = ring_basis[valid_mask] if ring_basis is not None else None

            try:
                popt, wl_c, wl_sc = fit_spectral_model(
                    tau_fit, ln_T_fit, fit_order, wl_fit,
                    ring_basis=ring_fit, poly_order=pix_poly,
                )
                # popt layout: k1..k_order, [alpha_ring,] p_N..p_0, intercept
                idx = 0
                kappas[pix_idx, :] = popt[idx: idx + fit_order];  idx += fit_order
                if ring_fit is not None:
                    ring_coeff[pix_idx] = popt[idx];               idx += 1
                # Store at leading slots; higher slots remain NaN for land pixels
                poly_coeffs[pix_idx, :pix_poly + 1] = popt[idx: idx + pix_poly + 1]
                idx += pix_poly + 1
                intercept[pix_idx] = popt[idx]
                poly_wl_center[pix_idx] = wl_c
                poly_wl_scale[pix_idx] = wl_sc
                fit_success[pix_idx] = True
            except (RuntimeError, ValueError):
                pass

    return {
        "pixel_index": pixel_idx_arr,
        "lon": pixel_subset["longitude"].to_numpy(dtype=float),
        "lat": pixel_subset["latitude"].to_numpy(dtype=float),
        "fit_mode": fit_mode,
        "kappas": kappas,
        "ring_coefficient": ring_coeff,
        "poly_coeffs": poly_coeffs,
        "poly_wl_center": poly_wl_center,
        "poly_wl_scale": poly_wl_scale,
        "is_ocean": ocean_mask,
        "intercept": intercept,
        "n_valid_channels": n_valid,
        "fit_success": fit_success,
    }


# ─── ln(T) vs tau example plots ───────────────────────────────────────────

def _prepare_pixel_fit_data(
    tau_result: dict,
    pix_idx: int,
    obs_ln_T: np.ndarray,
    tau_override: np.ndarray = None,
):
    """Return (tau_sorted, wl_sorted, ln_T_sorted, ln_T_smooth, ring_sorted) for one pixel.

    Reproduces the same sort + SG-smooth steps used in fit_spectral_model so
    the example plots show exactly what curve_fit received.

    Parameters
    ----------
    obs_ln_T : ndarray, shape (n_pix, n_ch)
        Observed ln(T) from ``extract_observed_lnT``; uses row pix_idx.
    tau_override : 1-D array, optional
        Replace ``tau_result['tau_eff'][pix_idx]`` with this tau vector.
        Used when ``fit_mode='simple'`` was run with a custom ``tau_simple``
        (e.g. total tau) so the plot x-axis matches what was actually fitted.
    """
    wl = np.asarray(tau_result["wavelength_nm"][pix_idx], dtype=float)
    tau_eff = (
        np.asarray(tau_override, dtype=float)
        if tau_override is not None
        else np.asarray(tau_result["tau_eff"][pix_idx], dtype=float)
    )
    obs = obs_ln_T[pix_idx, :]
    ring = None
    if "ring_basis" in tau_result:
        ring = np.asarray(tau_result["ring_basis"][pix_idx], dtype=float)

    valid = ~(np.isnan(wl) | np.isnan(tau_eff) | np.isnan(obs)) & (tau_eff > 0)
    if ring is not None:
        valid = valid & np.isfinite(ring)

    wl_v = wl[valid]
    tau_v = tau_eff[valid]
    ln_T_v = obs[valid]
    ring_v = ring[valid] if ring is not None else None

    sort_idx = np.argsort(tau_v)
    tau_s = tau_v[sort_idx]
    wl_s = wl_v[sort_idx]
    ln_T_s = ln_T_v[sort_idx]
    ring_s = ring_v[sort_idx] if ring_v is not None else None

    n = len(tau_s)
    if n < 5:
        return tau_s, wl_s, ln_T_s, ln_T_s.copy(), ring_s
    window_len = min(51, n if n % 2 == 1 else n - 1)
    if window_len < 5:
        window_len = 5
    polyorder = min(3, window_len - 2)
    ln_T_smooth = savgol_filter(ln_T_s, window_length=window_len, polyorder=polyorder)

    return tau_s, wl_s, ln_T_s, ln_T_smooth, ring_s


def write_lnT_tau_examples(
    tau_result: dict,
    fitting_result: dict,
    fit_order: int,
    output_dir,
    n_examples: int = 4,
    pixel_indices=None,
    tag: str = "",
    obs_ln_T=None,
    tau_simple=None,
    tau_x_label: str = None,
) -> Path:
    """Plot ln(T) vs tau for a few example pixels showing raw data, SG-smoothed
    data, and the fitted cumulant model curve.

    Parameters
    ----------
    tau_result : dict
        Output of ``compute_tau_subset``.
    fitting_result : dict
        Output of ``fit_pixel_ensemble``.
    fit_order : int
        Cumulant order used for fitting.
    n_examples : int
        Number of example pixels to plot (ignored if pixel_indices is given).
    pixel_indices : list of int, optional
        Indices into the fitting_result arrays (0-based). If None, ``n_examples``
        pixels are chosen evenly from the successfully fitted set.
    tag : str
        Suffix appended to the output filename.
    obs_ln_T : ndarray, shape (n_pix, n_ch), optional
        Observed ln(T) from ``extract_observed_lnT``. If None a synthetic
        fallback (ln_T = −τ) is used with a warning.
    tau_simple : ndarray, shape (n_pix, n_ch), optional
        The total tau array passed to ``fit_pixel_ensemble`` when
        ``fit_mode='simple'`` with a custom tau.  When provided, this is used
        as the x-axis so the plot matches what was actually fitted.
        Ignored in composite mode.
    tau_x_label : str, optional
        Override for the x-axis label.  Useful to distinguish mode 2
        (τ_total including Rayleigh + Ring) from mode 3 (τ_gas only).
        If None, a default label is chosen from the fit mode.

    Returns
    -------
    Path to the saved PNG.
    """
    n_pix_tau = int(tau_result["tau_eff"].shape[0]) if tau_result["tau_eff"].ndim == 2 else 0
    n_ch_tau = int(tau_result["tau_eff"].shape[1]) if n_pix_tau > 0 else 0

    if obs_ln_T is None:
        warnings.warn(
            "write_lnT_tau_examples: obs_ln_T not provided; using synthetic "
            "ln(T) = −τ. Pass extract_observed_lnT() result for physically "
            "correct plots.",
            UserWarning,
            stacklevel=2,
        )
        obs_ln_T = np.full((n_pix_tau, n_ch_tau), np.nan)
        for i in range(n_pix_tau):
            obs_ln_T[i, :] = -np.asarray(tau_result["tau_eff"][i], dtype=float)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_mode = fitting_result.get("fit_mode", "composite")
    stem = f"lnT_vs_tau_order{fit_order}_{fit_mode}{tag}"
    out_path = output_dir / f"{stem}.png"

    success = np.asarray(fitting_result["fit_success"], dtype=bool)
    ok_indices = np.flatnonzero(success)

    if ok_indices.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No successfully fitted pixels", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    if pixel_indices is not None:
        chosen = [int(i) for i in pixel_indices if 0 <= int(i) < len(success)]
    else:
        n = min(n_examples, len(ok_indices))
        chosen = ok_indices[np.round(np.linspace(0, len(ok_indices) - 1, n)).astype(int)].tolist()

    n_panels = len(chosen)
    # squeeze=False keeps return type uniform: always (1, n_panels) array of Axes
    fig, ax_grid = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4),
                                constrained_layout=True, squeeze=False)

    for col, pix_idx in enumerate(chosen):
        ax = ax_grid[0, col]
        tau_row_override = (
            np.asarray(tau_simple[pix_idx], dtype=float)
            if (fit_mode == "simple" and tau_simple is not None)
            else None
        )
        tau_s, wl_s, ln_T_s, ln_T_smooth, ring_s = _prepare_pixel_fit_data(
            tau_result, pix_idx, obs_ln_T, tau_override=tau_row_override
        )

        kappas = fitting_result["kappas"][pix_idx]
        intercept_val = fitting_result["intercept"][pix_idx]
        if tau_s.size == 0:
            ax.text(0.5, 0.5, "no valid data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        # Full model at actual data points — differs by fit_mode
        if fit_mode == "composite":
            # Absorption-only curve on dense tau grid (no ring / baseline / intercept)
            tau_dense = np.linspace(tau_s.min(), tau_s.max(), 300)
            ln_T_absorption = _eval_absorption_only(tau_dense, kappas)

            wl_c = float(fitting_result["poly_wl_center"][pix_idx])
            wl_sc = float(fitting_result["poly_wl_scale"][pix_idx])
            wl_norm_s = (wl_s - wl_c) / wl_sc
            poly_c = fitting_result["poly_coeffs"][pix_idx]
            baseline_pts = np.polyval(poly_c, wl_norm_s)
            ln_T_model_pts = _eval_absorption_only(tau_s, kappas) + baseline_pts + intercept_val
            if ring_s is not None and np.isfinite(fitting_result["ring_coefficient"][pix_idx]):
                a_ring = float(fitting_result["ring_coefficient"][pix_idx])
                ln_T_model_pts = ln_T_model_pts + a_ring * ring_s
            full_model_label = "full model (ring + baseline)"
        else:
            # simple mode: cumulant + intercept only — no absorption-only line
            tau_dense = None
            ln_T_absorption = None
            ln_T_model_pts = _eval_absorption_only(tau_s, kappas) + intercept_val
            full_model_label = "simple cumulant model"

        # Scatter: observed ln_T coloured by wavelength
        sc = ax.scatter(tau_s, ln_T_s, c=wl_s, cmap="Spectral_r", s=12, alpha=0.7,
                        zorder=2, label="observed ln(T)")
        plt.colorbar(sc, ax=ax, label="wavelength (nm)", fraction=0.046, pad=0.04)

        # SG-smoothed data
        ax.plot(tau_s, ln_T_smooth, color="black", lw=1.5, ls="--",
                zorder=3, label="SG smoothed (input to fit)")

        # Absorption-only curve — composite mode only
        if fit_mode == "composite":
            ax.plot(tau_dense, ln_T_absorption, color="red", lw=2,
                zorder=4, label=f"order-{fit_order} absorption only")
        ax.plot(tau_s, ln_T_model_pts, color="tab:orange", lw=1.2,
            zorder=5, label=full_model_label)

        lon = fitting_result["lon"][pix_idx]
        lat = fitting_result["lat"][pix_idx]
        k_str = "  ".join(f"k{i+1}={kappas[i]:.3f}" for i in range(len(kappas)))
        if tau_x_label is not None:
            x_label = tau_x_label
        elif fit_mode == "simple" and tau_simple is not None:
            x_label = "τ_eff total (all components)"
        else:
            x_label = "τ_eff O₂-O₂ gas"
        ax.set_title(
            f"pixel {pix_idx}  ({lat:.2f}°N, {lon:.2f}°E)  [{fit_mode}]\n{k_str}",
            fontsize=8,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("ln(T) observed")
        ax.legend(fontsize=7)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def compute_residual_lnT(
    obs_ln_T: np.ndarray,
    wavelength_nm: np.ndarray,
) -> np.ndarray:
    """Remove a per-pixel linear trend from ln(T) vs wavelength.

    For each pixel, fits a linear model ln(T)(λ) ≈ a·λ + b and returns
    the residual ln(T_res)(λ) = ln(T)(λ) − (a·λ + b), isolating the
    structured gas absorption signal from the broadband continuum.

    Parameters
    ----------
    obs_ln_T : ndarray, shape (n_pix, n_ch)
        Observed log-transmittance.
    wavelength_nm : ndarray, shape (n_pix, n_ch)
        Wavelengths [nm] corresponding to each channel.

    Returns
    -------
    residual_ln_T : ndarray, shape (n_pix, n_ch)
        Residual log-transmittance after linear-trend removal.
        NaN where obs_ln_T or wavelength_nm are NaN.
    """
    residual = np.full_like(obs_ln_T, np.nan, dtype=float)
    n_pix = obs_ln_T.shape[0]
    for pix_idx in range(n_pix):
        wl = np.asarray(wavelength_nm[pix_idx], dtype=float)
        ln_T = np.asarray(obs_ln_T[pix_idx], dtype=float)
        valid = np.isfinite(wl) & np.isfinite(ln_T)
        if valid.sum() < 2:
            continue
        wl_v = wl[valid]
        ln_T_v = ln_T[valid]
        a, b = np.polyfit(wl_v, ln_T_v, 1)
        residual[pix_idx, valid] = ln_T_v - (a * wl_v + b)
    return residual


def write_transmittance_wavelength_examples(
    tau_result: dict,
    fitting_result: dict,
    output_dir,
    n_examples: int = 4,
    pixel_indices=None,
    tag: str = "",
    obs_ln_T: np.ndarray = None,
) -> Path:
    """Plot observed transmittance vs wavelength for a few example pixels.

    Parameters
    ----------
    tau_result : dict
        Fitting assembly output; provides ``wavelength_nm`` on the native
        TEMPO grid (used as the x-axis).
    fitting_result : dict
        Output of ``fit_pixel_ensemble``; used for pixel selection and
        lon/lat labels.
    obs_ln_T : ndarray, shape (n_pix, n_ch)
        Observed log-transmittance from ``extract_observed_lnT``.

    Returns
    -------
    Path to the saved PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"transmittance_vs_wavelength{tag}.png"

    success = np.asarray(fitting_result["fit_success"], dtype=bool)
    ok_indices = np.flatnonzero(success)

    if ok_indices.size == 0 or obs_ln_T is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    if pixel_indices is not None:
        chosen = [int(i) for i in pixel_indices if 0 <= int(i) < len(success)]
    else:
        n = min(n_examples, len(ok_indices))
        chosen = ok_indices[np.round(np.linspace(0, len(ok_indices) - 1, n)).astype(int)].tolist()

    n_panels = len(chosen)
    fig, ax_grid = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4),
                                constrained_layout=True, squeeze=False)

    for col, pix_idx in enumerate(chosen):
        ax = ax_grid[0, col]

        wl     = np.asarray(tau_result["wavelength_nm"][pix_idx], dtype=float)
        ln_T   = obs_ln_T[pix_idx, :]

        valid = np.isfinite(wl) & np.isfinite(ln_T)
        if valid.sum() < 3:
            ax.text(0.5, 0.5, "no valid data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        wl_v   = wl[valid]
        ln_T_v = ln_T[valid]
        sort_idx = np.argsort(wl_v)
        wl_s   = wl_v[sort_idx]
        T_s    = np.exp(ln_T_v[sort_idx])

        ax.plot(wl_s, T_s, color="steelblue", lw=1.2)

        lon = fitting_result["lon"][pix_idx]
        lat = fitting_result["lat"][pix_idx]
        ax.set_title(f"pixel {pix_idx}  ({lat:.2f}°N, {lon:.2f}°E)", fontsize=9)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmittance")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_tau_component_examples(
    fitting_result: dict,
    reptran_o2o2_result: dict,
    reptran_rayleigh_result: dict,
    reptran_gas_result: dict,
    output_dir,
    reptran_o2o2_pixel_index=None,
    reptran_rayleigh_pixel_index=None,
    n_examples: int = 4,
    pixel_indices=None,
    tag: str = "",
    include_rayleigh: bool = True,
    ring_tau: np.ndarray = None,
    tau_result: dict = None,
) -> Path:
    """Plot optical-depth component spectra for a few example pixels.

    Components shown per panel:
      - O2-O2 (REPTRAN-projected)
      - Rayleigh (REPTRAN-projected, if include_rayleigh)
      - NO2 (REPTRAN gas product)
      - O3 (REPTRAN gas product)
      - Ring effect (if ring_tau is provided)

    AMF (sec_airmass) is added to each panel title when tau_result is supplied.

    Parameters
    ----------
    ring_tau : 1-D array, optional
        Ring-effect spectrum at the REPTRAN representative wavelengths.
        When provided, plotted as an additional component per panel.
    tau_result : dict, optional
        Output of ``compute_tau_subset``; used to look up ``sec_airmass``
        per pixel for display in the panel title.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if include_rayleigh else "_no_rayleigh"
    out_path = output_dir / f"tau_components_examples{tag}{suffix}.png"

    rep_wl = np.asarray(reptran_gas_result["rep_wavelength_nm"], dtype=float)
    fit_success = np.asarray(fitting_result.get("fit_success", []), dtype=bool)
    ok_indices = np.flatnonzero(fit_success)

    if ok_indices.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No successfully fitted pixels", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    if pixel_indices is not None:
        chosen = [int(i) for i in pixel_indices if 0 <= int(i) < len(fit_success)]
    else:
        n = min(n_examples, len(ok_indices))
        chosen = ok_indices[np.round(np.linspace(0, len(ok_indices) - 1, n)).astype(int)].tolist()

    if reptran_o2o2_pixel_index is None:
        o2_pix = np.arange(np.asarray(reptran_o2o2_result["tau_reptran"]).shape[0], dtype=int)
    else:
        o2_pix = np.asarray(reptran_o2o2_pixel_index, dtype=int)

    if reptran_rayleigh_pixel_index is None:
        ray_pix = np.arange(np.asarray(reptran_rayleigh_result["tau_reptran"]).shape[0], dtype=int)
    else:
        ray_pix = np.asarray(reptran_rayleigh_pixel_index, dtype=int)

    gas_pix = np.asarray(reptran_gas_result["pixel_index"], dtype=int)
    idx_o2 = {int(p): i for i, p in enumerate(o2_pix)}
    idx_ray = {int(p): i for i, p in enumerate(ray_pix)}
    idx_gas = {int(p): i for i, p in enumerate(gas_pix)}

    amf_map: dict = {}
    if tau_result is not None:
        tr_pix = np.asarray(tau_result["pixel_index"], dtype=int)
        tr_amf = np.asarray(tau_result["sec_airmass"], dtype=float)
        amf_map = {int(p): float(a) for p, a in zip(tr_pix, tr_amf)}

    n_panels = len(chosen)
    fig, ax_grid = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), constrained_layout=True, squeeze=False)

    for col, fit_row in enumerate(chosen):
        ax = ax_grid[0, col]
        pix_id = int(fitting_result["pixel_index"][fit_row])

        o2_row = idx_o2.get(pix_id)
        ray_row = idx_ray.get(pix_id)
        gas_row = idx_gas.get(pix_id)
        if o2_row is None or ray_row is None or gas_row is None:
            ax.text(0.5, 0.5, f"pixel {pix_id}\nmissing component rows", ha="center", va="center")
            ax.axis("off")
            continue

        tau_o2o2 = np.asarray(reptran_o2o2_result["tau_reptran"][o2_row], dtype=float)
        tau_no2 = np.asarray(reptran_gas_result["tau_no2"][gas_row], dtype=float)
        tau_o3 = np.asarray(reptran_gas_result["tau_o3"][gas_row], dtype=float)

        ax.plot(rep_wl, tau_no2, lw=1.6, label="NO2")
        ax.plot(rep_wl, tau_o3, lw=1.6, label="O3")
        ax.plot(rep_wl, tau_o2o2, lw=1.6, label="O2-O2")
        if include_rayleigh:
            tau_ray = np.asarray(reptran_rayleigh_result["tau_reptran"][ray_row], dtype=float)
            ax.plot(rep_wl, tau_ray, lw=1.6, label="Rayleigh")
        if ring_tau is not None:
            ax.plot(rep_wl, np.asarray(ring_tau, dtype=float), lw=1.6, ls="--", label="Ring")

        lon = float(fitting_result["lon"][fit_row])
        lat = float(fitting_result["lat"][fit_row])
        amf = amf_map.get(pix_id)
        amf_str = f"  AMF={amf:.3f}" if amf is not None else ""
        ax.set_title(f"pixel {pix_id} ({lat:.2f}°N, {lon:.2f}°E){amf_str}", fontsize=9)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Optical depth")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─── Output Writing ───────────────────────────────────────────────────────

def write_spectral_fitting_outputs(fitting_result, fit_order, output_dir, tag=""):
    """
    Write spectral fitting results to CSV and HDF5.

    Parameters
    ----------
    fitting_result : dict from fit_pixel_ensemble()
    fit_order : int
    output_dir : str or Path
    tag : str
        Optional suffix appended to output filenames before the extension
        (e.g. "_scan_0_1_xt_0_128") so successive runs do not overwrite each other.

    Returns
    -------
    output_paths : dict with keys 'csv', 'h5', 'md'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"spectral_fitting_order{fit_order}{tag}"

    # Build DataFrame for CSV output
    data_dict = {
        "pixel_index": fitting_result["pixel_index"],
        "lon": fitting_result["lon"],
        "lat": fitting_result["lat"],
        "n_valid_channels": fitting_result["n_valid_channels"],
        "fit_success": fitting_result["fit_success"],
        "intercept": fitting_result["intercept"],
    }
    if "is_ocean" in fitting_result:
        data_dict["is_ocean"] = fitting_result["is_ocean"].astype(int)
    if "ring_coefficient" in fitting_result:
        data_dict["a_ring"] = fitting_result["ring_coefficient"]

    # Add kappas as separate columns
    for i in range(fit_order):
        data_dict[f"k{i+1}"] = fitting_result["kappas"][:, i]

    # Add polynomial baseline coefficients
    if "poly_coeffs" in fitting_result:
        n_poly_coeffs = fitting_result["poly_coeffs"].shape[1]
        poly_order_stored = n_poly_coeffs - 1
        for i in range(n_poly_coeffs):
            data_dict[f"poly_p{poly_order_stored - i}"] = fitting_result["poly_coeffs"][:, i]
        data_dict["poly_wl_center"] = fitting_result.get("poly_wl_center", np.full(npix := len(fitting_result["pixel_index"]), np.nan))
        data_dict["poly_wl_scale"] = fitting_result.get("poly_wl_scale", np.full(npix, np.nan))

    df = pd.DataFrame(data_dict)

    # CSV output
    csv_path = output_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)

    # HDF5 output (for efficient storage and retrieval)
    h5_path = output_dir / f"{stem}.h5"
    try:
        import h5py
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("pixel_index", data=fitting_result["pixel_index"])
            f.create_dataset("lon", data=fitting_result["lon"])
            f.create_dataset("lat", data=fitting_result["lat"])
            f.create_dataset("kappas", data=fitting_result["kappas"])
            f.create_dataset("intercept", data=fitting_result["intercept"])
            if "ring_coefficient" in fitting_result:
                f.create_dataset("ring_coefficient", data=fitting_result["ring_coefficient"])
            if "poly_coeffs" in fitting_result:
                f.create_dataset("poly_coeffs", data=fitting_result["poly_coeffs"])
                f.create_dataset("poly_wl_center", data=fitting_result["poly_wl_center"])
                f.create_dataset("poly_wl_scale", data=fitting_result["poly_wl_scale"])
            f.create_dataset("n_valid_channels", data=fitting_result["n_valid_channels"])
            f.create_dataset("fit_success", data=fitting_result["fit_success"])
            if "is_ocean" in fitting_result:
                f.create_dataset("is_ocean", data=fitting_result["is_ocean"])
            f.attrs["fit_order"] = fit_order
            f.attrs["fit_mode"] = fitting_result.get("fit_mode", "composite")
    except ImportError:
        print(f"h5py not available; skipping HDF5 output at {h5_path}")

    # Markdown summary
    n_successful = np.sum(fitting_result["fit_success"])
    n_ocean = int(np.sum(fitting_result["is_ocean"])) if "is_ocean" in fitting_result else 0
    mode_label = fitting_result.get("fit_mode", "composite")
    md_text = (
        f"# Spectral Fitting Summary (Order {fit_order}, mode={mode_label})\n\n"
        f"- Total pixels: {len(fitting_result['pixel_index'])}\n"
        f"- Ocean pixels (higher poly_order): {n_ocean}\n"
        f"- Successfully fitted: {n_successful}\n"
        f"- Fit success rate: {100*n_successful/len(fitting_result['pixel_index']):.1f}%\n"
        f"- Data range:\n"
        f"  - Longitude: {fitting_result['lon'].min():.2f} to {fitting_result['lon'].max():.2f}°\n"
        f"  - Latitude: {fitting_result['lat'].min():.2f} to {fitting_result['lat'].max():.2f}°\n"
        f"- K1 (dominant opacity): {np.nanmean(fitting_result['kappas'][:, 0]):.3e} ±"
        f" {np.nanstd(fitting_result['kappas'][:, 0]):.3e}\n"
    )
    if fit_order >= 2:
        md_text += (
            f"- K2 (scattering/non-linear effects): "
            f"{np.nanmean(fitting_result['kappas'][:, 1]):.3e} ±"
            f" {np.nanstd(fitting_result['kappas'][:, 1]):.3e}\n"
        )
    if "ring_coefficient" in fitting_result:
        md_text += (
            f"- Ring coefficient (a_ring): "
            f"{np.nanmean(fitting_result['ring_coefficient']):.3e} ±"
            f" {np.nanstd(fitting_result['ring_coefficient']):.3e}\n"
        )

    md_path = output_dir / f"{stem}.md"
    md_path.write_text(md_text)

    return {
        "csv": csv_path,
        "h5": h5_path,
        "md": md_path,
    }


def _read_swath_panel_values(file_path, dataset_path, mirror_step, xtrack):
    """Read a swath variable and sample it at the requested pixel indices."""
    with h5py.File(file_path, "r") as f:
        ds = f[dataset_path]
        values = np.asarray(ds, dtype=float)
        fill_value = ds.attrs.get("_FillValue")
        if fill_value is not None:
            fill_value = float(np.asarray(fill_value).reshape(-1)[0])
            # Use relative tolerance to handle float32→float64 promotion of large fill values
            values = np.where(np.abs(values - fill_value) <= 1e-6 * abs(fill_value), np.nan, values)

    return values[mirror_step, xtrack]


def write_spectral_fitting_3panel_plot(
    fitting_result,
    pixel_table,
    cldo4_file,
    no2_file,
    fit_order,
    wl_min_nm,
    wl_max_nm,
    output_dir,
    tag="",
    cf_threshold=0.2,
    rad_file=None,
    fontsize=18,
    dpi=200,
):
    """Write a geographic scatter figure: optional GOES RGB + cloud fraction + k1 + k2.

    When *rad_file* is provided the UTC scan time is parsed from its filename
    and a GOES ABI true-colour panel is prepended on the left.  The GOES PNG
    is cached in *output_dir* so repeated calls reuse the same download.

    Each scatter panel plots pixels coloured by the panel value.  Pixels
    without a successful fit are excluded from k1/k2 panels but still appear
    in the cloud-fraction panel for spatial context.

    Parameters
    ----------
    fontsize : int
        Base font size.  Titles use fontsize, axis labels use fontsize,
        tick labels and colorbar annotations use fontsize-2.
        Use 18 for poster output, 12 for inline/paper figures.
    dpi : int
        Output PNG resolution.  200 is good for posters; 150 for screen.
    """
    import re
    import sys
    from pathlib import Path as _Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"spectral_fitting_order{fit_order}{tag}_4panel"
    out_path = output_dir / f"{stem}.png"

    pixel_idx = np.asarray(fitting_result["pixel_index"], dtype=int)
    if pixel_idx.size == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No fitted pixels", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    fit_lon = fitting_result["lon"]
    fit_lat = fitting_result["lat"]

    subset_mirror_step = pixel_table["mirror_step"].to_numpy(dtype=int)
    subset_xtrack = pixel_table["xtrack"].to_numpy(dtype=int)
    subset_lon = pixel_table["longitude"].to_numpy(dtype=float)
    subset_lat = pixel_table["latitude"].to_numpy(dtype=float)

    no2_proxy = _read_swath_panel_values(
        no2_file,
        "support_data/vertical_column_total",
        subset_mirror_step,
        subset_xtrack,
    )
    cloud_fraction = _read_swath_panel_values(
        cldo4_file,
        "product/cloud_fraction",
        subset_mirror_step,
        subset_xtrack,
    )
    success = np.asarray(fitting_result["fit_success"], dtype=bool)

    # Mask fitted kappas for cloudy or unknown-CF pixels.
    # NaN cloud fraction (CLDO4 retrieval failure) usually indicates heavy cloud cover.
    cloud_fraction_fit = np.full(pixel_idx.shape, np.nan, dtype=float)
    valid_pix = (pixel_idx >= 0) & (pixel_idx < cloud_fraction.size)
    cloud_fraction_fit[valid_pix] = cloud_fraction[pixel_idx[valid_pix]]
    cloud_mask = ~np.isfinite(cloud_fraction_fit) | (cloud_fraction_fit > cf_threshold)

    k1 = np.where(success, fitting_result["kappas"][:, 0], np.nan)
    k2 = (
        np.where(success, fitting_result["kappas"][:, 1], np.nan)
        if fitting_result["kappas"].shape[1] >= 2
        else np.full_like(k1, np.nan)
    )
    k1 = np.where(cloud_mask, np.nan, k1)
    k2 = np.where(cloud_mask, np.nan, k2)

    # ── Optional GOES panel ───────────────────────────────────────────────────
    goes_img    = None
    goes_extent = None
    goes_title  = "GOES ABI RGB"

    if rad_file is not None:
        m = re.search(r'(\d{8}T\d{6}Z)', str(rad_file))
        if m:
            utc_time = pd.Timestamp(m.group(1), tz='UTC')

            fin_lon = subset_lon[np.isfinite(subset_lon)]
            fin_lat = subset_lat[np.isfinite(subset_lat)]
            if fin_lon.size > 0 and fin_lat.size > 0:
                pad = 0.5
                goes_extent = [
                    float(fin_lon.min()) - pad, float(fin_lon.max()) + pad,
                    float(fin_lat.min()) - pad, float(fin_lat.max()) + pad,
                ]
                try:
                    _src = str(_Path(__file__).parent)
                    if _src not in sys.path:
                        sys.path.insert(0, _src)
                    from goes_abi_rgb import download_goes_abi_rgb
                    png_path, sat_label, scan_time = download_goes_abi_rgb(
                        utc_time  = utc_time,
                        extent    = goes_extent,
                        fdir      = str(output_dir),
                        run       = True,
                    )
                    goes_img   = plt.imread(png_path)
                    dt_min     = abs((scan_time - utc_time).total_seconds()) / 60
                    goes_title = (f"GOES-{sat_label.upper()} ABI RGB\n"
                                  f"{scan_time.strftime('%H:%M UTC')}  "
                                  f"(Δt={dt_min:.1f} min)")
                except Exception as exc:
                    print(f"  Warning: GOES download skipped ({exc})", flush=True)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fs       = fontsize          # title / axis label size
    fs_small = max(fontsize - 2, 8)   # tick labels and colorbar annotations
    dot_size = max(int(fontsize * 1.8), 20)   # scatter marker size

    n_panels   = 4 if goes_img is not None else 3
    fig_width  = 6 * n_panels
    fig_height = max(5, fontsize * 0.15 + 0.5)   # ~7" at fontsize=18, ~5" at fontsize=12
    fig, axes  = plt.subplots(1, n_panels, figsize=(fig_width, fig_height),
                               constrained_layout=True)

    scatter_axes = axes[1:] if goes_img is not None else axes

    panels = [
        (subset_lon, subset_lat, cloud_fraction, "Cloud fraction"),
        (fit_lon, fit_lat, k1, f"Fitted <l'> (order {fit_order})"),
        (fit_lon, fit_lat, k2, f"Fitted var(l') (order {fit_order})"),
    ]

    for ax, (lon, lat, values, title) in zip(scatter_axes, panels):
        finite = np.isfinite(values) & np.isfinite(lon) & np.isfinite(lat)
        if title == "Cloud fraction":
            sc = ax.scatter(
                lon[finite], lat[finite], c=values[finite],
                s=dot_size, cmap="Blues_r", vmin=0, vmax=1,
            )
        elif title.startswith("NO2"):
            sc = ax.scatter(lon[finite], lat[finite], c=values[finite],
                            s=dot_size, cmap="Reds")
        else:
            vmin_val = np.nanpercentile(values[finite], 5) if finite.any() else 0
            vmax_val = np.nanpercentile(values[finite], 95) if finite.any() else 1
            sc = ax.scatter(lon[finite], lat[finite], c=values[finite], s=dot_size,
                            cmap="viridis", vmin=vmin_val, vmax=vmax_val)
        ax.set_title(title, fontsize=fs)
        ax.set_xlabel("Longitude", fontsize=fs)
        ax.set_ylabel("Latitude", fontsize=fs)
        cbar_label = ("molecules/cm$^2$" if title.startswith("NO2")
                      else "fraction" if title == "Cloud fraction"
                      else "value")
        cb = fig.colorbar(sc, ax=ax, fraction=0.060, pad=0.04)
        cb.ax.tick_params(labelsize=fs_small)
        cb.set_label(cbar_label, fontsize=fs_small)

    # Align all scatter panels to the same lon/lat range
    ref_ax = scatter_axes[-1]
    xmin, xmax = ref_ax.get_xlim()
    ymin, ymax = ref_ax.get_ylim()
    for ax in scatter_axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis='both', which='major', labelsize=fs_small-2)

    # ── GOES panel (leftmost) ─────────────────────────────────────────────────
    if goes_img is not None:
        ax0 = axes[0]
        ax0.imshow(goes_img,
                   extent=[goes_extent[0], goes_extent[1],
                           goes_extent[2], goes_extent[3]],
                   aspect='auto', origin='upper')
        # Overlay TEMPO pixel footprint
        # fp_finite = np.isfinite(subset_lon) & np.isfinite(subset_lat)
        # ax0.scatter(subset_lon[fp_finite], subset_lat[fp_finite],
        #             s=4, c='yellow', alpha=0.4, linewidths=0)
        ax0.set_xlim(xmin, xmax)
        ax0.set_ylim(ymin, ymax)
        ax0.set_title(goes_title, fontsize=fs)
        ax0.set_xlabel("Longitude", fontsize=fs)
        ax0.set_ylabel("Latitude", fontsize=fs)
        ax0.tick_params(axis='both', which='major', labelsize=fs_small)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
