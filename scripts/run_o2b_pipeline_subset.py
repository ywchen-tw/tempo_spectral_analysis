#!/usr/bin/env python
"""Run the full TEMPO O2-B retrieval pipeline on a scan/xtrack subset.

Executes the complete staged workflow (WP1–WP7) on a user-defined spatial
and spectral subset centred on the O2 B-band (~685–695 nm): pixel table →
GEOS profiles → slit kernels → tau computation → REPTRAN projection →
validation → spectral fitting.

Key difference from run_o2o2_pipeline_subset.py:
  - WP4 uses HITRAN line-by-line O2 B-band cross-sections (wp4_o2b_tau)
  - tau is linear in n_O2 (monomer), not quadratic (CIA)
  - REPTRAN gas tau includes O2 via k-distribution; it is excluded from the
    composite tau to avoid double-counting with our WP4 O2-B tau

Example:
python scripts/run_o2b_pipeline_subset.py \\
    --lat-min 31.6 --lat-max 33.6 \\
    --lon-min -123.1 --lon-max -121.1 \\
    --wl-min 683 --wl-max 697 --spectral-fit-order 2 \\
    --rad-file data/TEMPO/TEMPO_RAD_L1_V03_20240708T160926Z_S007G09.nc \\
    --cldo4-file data/TEMPO/TEMPO_CLDO4_L2_V03_20240708T160926Z_S007G09.nc \\
    --no2-file data/TEMPO/TEMPO_NO2_L2_V03_20240708T160926Z_S007G09.nc \\
    --irr-file data/TEMPO/TEMPO_IRR_L1_V03_20240711T042711Z.nc \\
    --geos-file data/TEMPO/GEOS-CF.v01.rpl.sat_inst_1hr_r721x361_v72.20240708_1600z.nc4 \\
    --tag S007G09_160926_o2b
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from wp1_pixel_table import PixelTableConfig, build_pixel_table
from wp2_profiles import ProfileConfig, build_profiles_for_pixels
from wp3_slit_kernel import SlitKernelConfig, build_slit_kernel_table
from wp4_o2b_tau import (
    TauO2BConfig,
    compute_tau_subset_o2b,
    write_o2b_tau_netcdf,
    write_o2b_vertical_debug_plot,
)
from wp4_tau import write_tau_netcdf  # reused for Rayleigh-only NC if needed
from wp5_reptran import (
    ReptranConfig,
    compute_tau_reptran_from_profiles,
    project_tau_to_reptran,
    write_reptran_gas_outputs,
    write_reptran_outputs,
)
from wp6_validation import build_validation_table, write_validation_outputs
from wp7_spectral_fitting import (
    build_ring_template_from_irr,
    compute_residual_lnT,
    compute_ring_optical_depth,
    extract_observed_lnT,
    fit_pixel_ensemble,
    write_tau_component_examples,
    write_lnT_tau_examples,
    write_spectral_fitting_outputs,
    write_spectral_fitting_3panel_plot,
    write_transmittance_wavelength_examples,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rad-file",   default="data/TEMPO/TEMPO_RAD_L1_V03_20240708T190926Z_S010G09.nc")
    p.add_argument("--cldo4-file", default="data/TEMPO/TEMPO_CLDO4_L2_V03_20240708T190926Z_S010G09.nc")
    p.add_argument("--no2-file",   default="data/TEMPO/TEMPO_NO2_L2_V03_20240708T190926Z_S010G09.nc")
    p.add_argument("--irr-file",   default="data/TEMPO/TEMPO_IRR_L1_V03_20240711T042711Z.nc")
    p.add_argument("--geos-file",  default="data/TEMPO/GEOS-CF.v01.rpl.sat_inst_1hr_r721x361_v72.20240708_1900z.nc4")
    p.add_argument("--scan-start",  type=int, default=0)
    p.add_argument("--scan-end",    type=int, default=131)
    p.add_argument("--xtrack-start", type=int, default=0)
    p.add_argument("--xtrack-end",   type=int, default=2048)
    p.add_argument("--wl-min", type=float, default=683.0)
    p.add_argument("--wl-max", type=float, default=697.0)
    p.add_argument("--lat-min", type=float, default=None)
    p.add_argument("--lat-max", type=float, default=None)
    p.add_argument("--lon-min", type=float, default=None)
    p.add_argument("--lon-max", type=float, default=None)
    p.add_argument(
        "--use-interest-region",
        action="store_true",
        help="Apply the preferred test bbox (lat 29.6-33.6, lon -123.8 to -120.5) if no bbox supplied.",
    )
    p.add_argument(
        "--allow-wavecal-nonzero",
        action="store_true",
        help="Do not require wavecal_opt_status==0 in valid-pixel mask.",
    )
    p.add_argument("--hitran-o2b-file", default="data/crs/hitran_o2b.par",
                   help="HITRAN .par file for O2 B-band cross-sections.")
    p.add_argument("--hitran-h2o-file", default="data/crs/hitran_h2o_o2b_range.par",
                   help="HITRAN .par file for H2O cross-sections in the O2-B range.")
    p.add_argument("--out-dir", default="outputs/wp1_wp2")
    p.add_argument("--tag", default="",
                   help="Optional run tag; outputs go into tag-specific subfolders.")
    p.add_argument("--spectral-fit-order", type=int, default=2)
    p.add_argument("--cf-threshold", type=float, default=0.2,
                   help="Maximum cloud fraction for pixels to be included in spectral fitting.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    run_tag = str(args.tag).strip().replace(os.sep, "_")
    if os.altsep:
        run_tag = run_tag.replace(os.altsep, "_")

    run_out_dir    = os.path.join(args.out_dir, run_tag) if run_tag else args.out_dir
    tau_native_dir = os.path.join(ROOT, "outputs", "tau_o2b_native",  run_tag or "")
    tau_reptran_dir= os.path.join(ROOT, "outputs", "tau_o2b_reptran", run_tag or "")
    qc_dir         = os.path.join(ROOT, "outputs", "qc",              run_tag or "")
    for d in (run_out_dir, tau_native_dir, tau_reptran_dir, qc_dir):
        os.makedirs(d, exist_ok=True)

    if args.use_interest_region and all(
        v is None for v in (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    ):
        args.lat_min = 29.6;  args.lat_max = 33.6
        args.lon_min = -123.8; args.lon_max = -120.5

    # ── WP1: pixel table ─────────────────────────────────────────────────────
    pix, wl_diag = build_pixel_table(
        rad_file=args.rad_file,
        cldo4_file=args.cldo4_file,
        scan_start=args.scan_start,
        scan_end=args.scan_end,
        xtrack_start=args.xtrack_start,
        xtrack_end=args.xtrack_end,
        config=PixelTableConfig(
            wl_min_nm=args.wl_min,
            wl_max_nm=args.wl_max,
            require_wavecal_opt_status_zero=not args.allow_wavecal_nonzero,
            band="band_540_740_nm",
        ),
    )

    base_valid_n = int(pix["valid_pixel"].sum())
    spatial_mask = pix["valid_pixel"].to_numpy(dtype=bool).copy()
    if args.lat_min is not None: spatial_mask &= pix["latitude"] >= args.lat_min
    if args.lat_max is not None: spatial_mask &= pix["latitude"] <= args.lat_max
    if args.lon_min is not None: spatial_mask &= pix["longitude"] >= args.lon_min
    if args.lon_max is not None: spatial_mask &= pix["longitude"] <= args.lon_max

    if base_valid_n == 0:
        print("Warning: no valid pixels found in the requested scan/xtrack subset before bbox filtering.")

    if not np.all(spatial_mask):
        pix = pd.DataFrame(pix.iloc[np.flatnonzero(spatial_mask)].copy().reset_index(drop=True))

    if len(pix) == 0:
        if base_valid_n > 0 and any(v is not None for v in (args.lat_min, args.lat_max, args.lon_min, args.lon_max)):
            print("Warning: bbox constraints removed all valid pixels.")
        else:
            print("Warning: requested constraints produced no pixels.")
        print("No valid pixels to process — exiting.")
        return 1

    # ── WP2: GEOS-CF profiles ─────────────────────────────────────────────────
    prof = build_profiles_for_pixels(
        pixel_df=pix,
        geos_file=args.geos_file,
        config=ProfileConfig(),
    )

    # ── WP3: slit kernel ──────────────────────────────────────────────────────
    slit_df, slit_diag = build_slit_kernel_table(
        irr_file=args.irr_file,
        lambda_corrected_nm=wl_diag["lambda_corrected_nm"],
        xtrack_start=args.xtrack_start,
        xtrack_end=args.xtrack_end,
        config=SlitKernelConfig(
            wl_min_nm=args.wl_min,
            wl_max_nm=args.wl_max,
            require_wavecal_opt_status_zero=not args.allow_wavecal_nonzero,
            band="band_540_740_nm",
        ),
    )

    # ── WP4: O2-B tau (HITRAN line-by-line, linear in n_O2) ──────────────────
    tau_result = compute_tau_subset_o2b(
        pixel_df=pix,
        profiles=prof,
        wavelength_diag=wl_diag,
        slit_df=slit_df,
        scan_start=args.scan_start,
        xtrack_start=args.xtrack_start,
        config=TauO2BConfig(
            hitran_o2b_file=args.hitran_o2b_file,
            hitran_h2o_file=args.hitran_h2o_file,
            output_wavelength_min_nm=args.wl_min,
            output_wavelength_max_nm=args.wl_max,
        ),
    )

    # ── WP5: REPTRAN gas tau ──────────────────────────────────────────────────
    # REPTRAN computes O2 tau (non-zero at 685–695 nm) along with H2O/N2/N2O/NO2/O3.
    # We use only NO2 + O3 + H2O from REPTRAN in the composite tau (see below)
    # to avoid double-counting O2 with the WP4 HITRAN result.
    reptran_cfg = ReptranConfig(wavelength_min_nm=args.wl_min, wavelength_max_nm=args.wl_max)

    reptran_result = project_tau_to_reptran(
        tau_result=tau_result,
        reptran_cfg=reptran_cfg,
    )
    reptran_gas_result = compute_tau_reptran_from_profiles(
        pixel_df=pix,
        profiles=prof,
        wavelength_diag=wl_diag,
        slit_df=slit_df,
        reptran_cfg=reptran_cfg,
        scan_start=args.scan_start,
        xtrack_start=args.xtrack_start,
    )

    validation_table = build_validation_table(
        pixel_df=pix,
        tau_result=tau_result,
        reptran_result=reptran_result,
    )

    # ── Output paths ─────────────────────────────────────────────────────────
    chunk_tag = f"scan_{args.scan_start}_{args.scan_end}_xt_{args.xtrack_start}_{args.xtrack_end}"
    if any(v is not None for v in (args.lat_min, args.lat_max, args.lon_min, args.lon_max)):
        chunk_tag += "_bbox"

    pixel_csv        = os.path.join(run_out_dir,    f"pixel_table_{chunk_tag}.csv")
    profile_npz      = os.path.join(run_out_dir,    f"profiles_{chunk_tag}.npz")
    wavelength_npz   = os.path.join(run_out_dir,    f"wavelength_diag_{chunk_tag}.npz")
    slit_csv         = os.path.join(run_out_dir,    f"slit_kernel_{chunk_tag}.csv")
    slit_npz         = os.path.join(run_out_dir,    f"slit_kernel_{chunk_tag}.npz")
    tau_nc           = os.path.join(tau_native_dir,  f"tau_o2b_native_{chunk_tag}.nc")
    tau_npz          = os.path.join(run_out_dir,    f"tau_o2b_native_{chunk_tag}.npz")
    tau_summary_csv  = os.path.join(run_out_dir,    f"tau_o2b_summary_{chunk_tag}.csv")
    reptran_prefix       = os.path.join(tau_reptran_dir, f"tau_o2b_reptran_{chunk_tag}")
    reptran_gas_prefix   = os.path.join(tau_reptran_dir, f"tau_gas_reptran_{chunk_tag}")
    reptran_summary_csv  = os.path.join(run_out_dir,    f"tau_o2b_reptran_summary_{chunk_tag}.csv")
    validation_prefix    = os.path.join(qc_dir,         f"tau_o2b_validation_{chunk_tag}")

    # ── Write intermediate artifacts ──────────────────────────────────────────
    pix.to_csv(pixel_csv, index=False)
    np.savez_compressed(profile_npz, **prof)
    np.savez_compressed(wavelength_npz, **wl_diag)
    np.savez_compressed(slit_npz, **slit_diag)
    slit_df.to_csv(slit_csv, index=False)
    np.savez_compressed(tau_npz, **tau_result)
    write_o2b_tau_netcdf(tau_nc, tau_result, pix)

    o2b_debug_png = os.path.join(qc_dir, f"o2b_vertical_debug_{chunk_tag}.png")
    o2b_debug_png = write_o2b_vertical_debug_plot(
        out_path=o2b_debug_png,
        pixel_df=pix,
        profiles=prof,
        wavelength_diag=wl_diag,
        slit_df=slit_df,
        tau_result=tau_result,
        scan_start=args.scan_start,
        xtrack_start=args.xtrack_start,
    )

    reptran_pixel_df = pd.DataFrame(
        pix.iloc[np.asarray(tau_result["pixel_index"], dtype=int)].copy().reset_index(drop=True)
        if tau_result["pixel_index"].size
        else pix.iloc[0:0].copy().reset_index(drop=True)
    )
    reptran_nc,     reptran_csv     = write_reptran_outputs(reptran_prefix,     reptran_result,     reptran_pixel_df)
    reptran_gas_nc, reptran_gas_csv = write_reptran_gas_outputs(reptran_gas_prefix, reptran_gas_result, pix)

    if tau_result["pixel_index"].size:
        tau_summary = pd.DataFrame({
            "pixel_index":  tau_result["pixel_index"],
            "mirror_step":  tau_result["mirror_step"],
            "xtrack":       tau_result["xtrack"],
            "sec_airmass":  tau_result["sec_airmass"],
            "tau_band_mean": tau_result["tau_band_mean"],
        })
    else:
        tau_summary = pd.DataFrame(columns=["pixel_index", "mirror_step", "xtrack", "sec_airmass", "tau_band_mean"])
    tau_summary.to_csv(tau_summary_csv, index=False)

    reptran_summary = pd.DataFrame({
        "band_index":       reptran_result["band_index"],
        "band_name":        reptran_result["band_name"],
        "rep_wavelength_nm": reptran_result["rep_wavelength_nm"],
    })
    reptran_summary.to_csv(reptran_summary_csv, index=False)

    validation_csv, validation_md = write_validation_outputs(validation_prefix, validation_table)

    # ── WP5 Rayleigh projection onto REPTRAN grid ─────────────────────────────
    tau_rayleigh_proj = project_tau_to_reptran(
        tau_result={
            "wavelength_nm": tau_result["wavelength_nm"],
            "tau_eff":       tau_result["tau_rayleigh_eff"],
        },
        reptran_cfg=reptran_cfg,
    )

    # ── WP7: Spectral fitting ─────────────────────────────────────────────────
    spectral_fitting_order = int(args.spectral_fit_order)
    obs_ln_T_native = extract_observed_lnT(
        tau_result=tau_result,
        pixel_table=pix,
        rad_file=args.rad_file,
        irr_file=args.irr_file,
        wavelength_diag=wl_diag,
        scan_start=args.scan_start,
        xtrack_start=args.xtrack_start,
        band="band_540_740_nm",
    )

    # REPTRAN representative wavelengths for gas-tau interpolation
    rep_wl      = np.asarray(reptran_gas_result["rep_wavelength_nm"], dtype=float)
    rep_sort    = np.argsort(rep_wl)
    rep_wl_sorted = rep_wl[rep_sort]

    ring_template_rep = build_ring_template_from_irr(
        irr_file=args.irr_file,
        rep_wavelength_nm=rep_wl,
        band="band_540_740_nm",
    )
    ring_native = compute_ring_optical_depth(tau_result, args.irr_file,
                                             band="band_540_740_nm")

    o2b_idx  = np.asarray(tau_result["pixel_index"],       dtype=int)
    gas_idx  = np.asarray(reptran_gas_result["pixel_index"], dtype=int)
    idx_to_o2b_row = {int(p): i for i, p in enumerate(o2b_idx)}
    idx_to_gas_row = {int(p): i for i, p in enumerate(gas_idx)}
    common_idx = sorted(set(int(p) for p in o2b_idx).intersection(int(p) for p in gas_idx))

    quality_ok_mask = (
        pix["quality_ok"].to_numpy(dtype=bool)
        if "quality_ok" in pix.columns
        else np.ones(len(pix), dtype=bool)
    )
    pix_cf = pix["cldo4_cloud_fraction"].to_numpy(dtype=float)
    cf_ok_mask = np.isfinite(pix_cf) & (pix_cf <= args.cf_threshold)
    skipped_bad_quality = 0
    skipped_cloudy = 0

    fit_pixel_idx       = []
    fit_tau_gas_rows    = []    # Mode 3: O2-B + NO2 + O3 + H2O (gas only, no Rayleigh/Ring)
    fit_tau_rayleigh_rows = []  # Mode 4: gas + Rayleigh (no Ring)
    fit_tau_total_rows  = []    # Mode 2: gas + Rayleigh + Ring
    fit_wl_rows         = []
    fit_lnT_rows        = []
    fit_ring_rows       = []

    for pidx in common_idx:
        if not bool(quality_ok_mask[pidx]):
            skipped_bad_quality += 1
            continue
        if not bool(cf_ok_mask[pidx]):
            skipped_cloudy += 1
            continue

        o2b_row = idx_to_o2b_row[pidx]
        gas_row = idx_to_gas_row[pidx]

        wl_native = np.asarray(tau_result["wavelength_nm"][o2b_row], dtype=float)
        lnT_row   = np.asarray(obs_ln_T_native[o2b_row], dtype=float)
        tau_o2b   = np.asarray(tau_result["tau_eff"][o2b_row], dtype=float)
        tau_ray   = np.asarray(tau_result["tau_rayleigh_eff"][o2b_row], dtype=float)
        tau_ring  = np.asarray(ring_native[o2b_row], dtype=float)

        in_range = np.isfinite(wl_native) & (wl_native >= args.wl_min) & (wl_native <= args.wl_max)
        if np.count_nonzero(in_range & np.isfinite(lnT_row)) < 2:
            continue

        wl_out   = np.where(in_range, wl_native, np.nan)
        lnT_out  = np.where(in_range, lnT_row,   np.nan)
        tau_o2b  = np.where(in_range, tau_o2b,   np.nan)
        tau_ray  = np.where(in_range, tau_ray,    np.nan)
        tau_ring = np.where(in_range, tau_ring,   np.nan)

        # H2O tau from HITRAN (native TEMPO grid, already in tau_result)
        tau_h2o_native = np.asarray(tau_result["tau_h2o_eff"][o2b_row], dtype=float)
        tau_h2o_native = np.where(in_range, tau_h2o_native, np.nan)

        # Secondary gas taus from REPTRAN — exclude O2 (double-counted with WP4)
        # and H2O (replaced by HITRAN above); use only NO2 + O3 here.
        tau_no2_rep  = np.asarray(reptran_gas_result["tau_no2"][gas_row],  dtype=float)[rep_sort]
        tau_o3_rep   = np.asarray(reptran_gas_result["tau_o3"][gas_row],   dtype=float)[rep_sort]
        tau_sec_rep  = tau_no2_rep + tau_o3_rep

        # Interpolate NO2 + O3 from REPTRAN to native TEMPO grid
        tau_sec_native = np.full_like(wl_native, np.nan)
        interp_mask = in_range & (wl_native >= rep_wl_sorted[0]) & (wl_native <= rep_wl_sorted[-1])
        if interp_mask.any():
            tau_sec_native[interp_mask] = np.interp(
                wl_native[interp_mask], rep_wl_sorted, tau_sec_rep,
            )

        # Build composite taus
        tau_gas_eff   = tau_o2b + tau_h2o_native + tau_sec_native          # Mode 3: O2-B + H2O + NO2 + O3
        tau_ray_eff   = tau_gas_eff + tau_ray                               # Mode 4: + Rayleigh
        tau_total_eff = tau_ray_eff + np.where(                             # Mode 2: + Ring
            np.isfinite(tau_ring), tau_ring, 0.0)

        fit_pixel_idx.append(pidx)
        fit_tau_gas_rows.append(tau_gas_eff)
        fit_tau_rayleigh_rows.append(tau_ray_eff)
        fit_tau_total_rows.append(tau_total_eff)
        fit_wl_rows.append(wl_out)
        fit_lnT_rows.append(lnT_out)
        fit_ring_rows.append(tau_ring)

    if fit_pixel_idx:
        tau_fit = {
            "pixel_index": np.asarray(fit_pixel_idx, dtype=int),
            "wavelength_nm": np.vstack(fit_wl_rows),
            "tau_eff":       np.vstack(fit_tau_gas_rows),
            "ring_basis":    np.vstack(fit_ring_rows),
        }
        tau_mode2 = np.vstack(fit_tau_total_rows)
        tau_mode3 = tau_fit["tau_eff"]
        tau_mode4 = np.vstack(fit_tau_rayleigh_rows)
        obs_ln_T_fit = np.vstack(fit_lnT_rows)
    else:
        tau_fit = {
            "pixel_index": np.array([], dtype=int),
            "wavelength_nm": np.empty((0, 0), dtype=float),
            "tau_eff":       np.empty((0, 0), dtype=float),
            "ring_basis":    np.empty((0, 0), dtype=float),
        }
        tau_mode2 = tau_mode3 = tau_mode4 = obs_ln_T_fit = np.empty((0, 0), dtype=float)

    if skipped_bad_quality:
        print(f"Skipped {skipped_bad_quality} bad-quality pixels before spectral fitting.")
    if skipped_cloudy:
        print(f"Skipped {skipped_cloudy} cloudy pixels (CF > {args.cf_threshold}) before spectral fitting.")

    spectral_output_dir = qc_dir
    spectral_tag = f"_{chunk_tag}_o2b_tau_total_eff"

    fitting_result = fit_pixel_ensemble(
        tau_fit, pix, fit_order=spectral_fitting_order, obs_ln_T=obs_ln_T_fit
    )
    spectral_paths = write_spectral_fitting_outputs(
        fitting_result, spectral_fitting_order, spectral_output_dir, tag=spectral_tag
    )
    spectral_plot = write_spectral_fitting_3panel_plot(
        fitting_result=fitting_result,
        pixel_table=pix,
        cldo4_file=args.cldo4_file,
        no2_file=args.no2_file,
        fit_order=spectral_fitting_order,
        wl_min_nm=args.wl_min,
        wl_max_nm=args.wl_max,
        output_dir=spectral_output_dir,
        tag=spectral_tag,
        cf_threshold=args.cf_threshold,
    )
    lnT_plot = write_lnT_tau_examples(
        tau_result=tau_fit,
        fitting_result=fitting_result,
        fit_order=spectral_fitting_order,
        output_dir=spectral_output_dir,
        n_examples=4,
        tag=spectral_tag,
        obs_ln_T=obs_ln_T_fit,
    )
    tau_components_plot = write_tau_component_examples(
        fitting_result=fitting_result,
        reptran_o2o2_result=reptran_result,
        reptran_rayleigh_result=tau_rayleigh_proj,
        reptran_gas_result=reptran_gas_result,
        reptran_o2o2_pixel_index=tau_result["pixel_index"],
        reptran_rayleigh_pixel_index=tau_result["pixel_index"],
        output_dir=spectral_output_dir,
        n_examples=4,
        tag=spectral_tag,
        ring_tau=ring_template_rep,
        tau_result=tau_result,
    )
    tau_components_no_rayleigh_plot = write_tau_component_examples(
        fitting_result=fitting_result,
        reptran_o2o2_result=reptran_result,
        reptran_rayleigh_result=tau_rayleigh_proj,
        reptran_gas_result=reptran_gas_result,
        reptran_o2o2_pixel_index=tau_result["pixel_index"],
        reptran_rayleigh_pixel_index=tau_result["pixel_index"],
        output_dir=spectral_output_dir,
        n_examples=4,
        tag=spectral_tag,
        include_rayleigh=False,
        tau_result=tau_result,
    )
    transmittance_plot = write_transmittance_wavelength_examples(
        tau_result=tau_fit,
        fitting_result=fitting_result,
        output_dir=spectral_output_dir,
        n_examples=4,
        tag=spectral_tag,
        obs_ln_T=obs_ln_T_fit,
    )

    # ── Mode 2: total tau (O2-B + sec-gases + Rayleigh + Ring) ───────────────
    tag_m2 = f"_{chunk_tag}_o2b_simple_total"
    fitting_result_m2 = fit_pixel_ensemble(
        tau_fit, pix, fit_order=spectral_fitting_order,
        obs_ln_T=obs_ln_T_fit, fit_mode="simple", tau_simple=tau_mode2,
    )
    write_spectral_fitting_outputs(fitting_result_m2, spectral_fitting_order, spectral_output_dir, tag=tag_m2)
    write_spectral_fitting_3panel_plot(
        fitting_result=fitting_result_m2, pixel_table=pix,
        cldo4_file=args.cldo4_file, no2_file=args.no2_file,
        fit_order=spectral_fitting_order, wl_min_nm=args.wl_min, wl_max_nm=args.wl_max,
        output_dir=spectral_output_dir, tag=tag_m2,
        cf_threshold=args.cf_threshold,
    )
    write_lnT_tau_examples(
        tau_result=tau_fit, fitting_result=fitting_result_m2,
        fit_order=spectral_fitting_order, output_dir=spectral_output_dir,
        n_examples=4, tag=tag_m2, obs_ln_T=obs_ln_T_fit,
        tau_simple=tau_mode2,
        tau_x_label="τ_total (O₂-B + H₂O + NO₂ + O₃ + Rayleigh + Ring)",
    )

    # ── Mode 4: gas + Rayleigh (no Ring) ─────────────────────────────────────
    tag_m4 = f"_{chunk_tag}_o2b_simple_gas_rayleigh"
    fitting_result_m4 = fit_pixel_ensemble(
        tau_fit, pix, fit_order=spectral_fitting_order,
        obs_ln_T=obs_ln_T_fit, fit_mode="simple", tau_simple=tau_mode4,
    )
    write_spectral_fitting_outputs(fitting_result_m4, spectral_fitting_order, spectral_output_dir, tag=tag_m4)
    write_spectral_fitting_3panel_plot(
        fitting_result=fitting_result_m4, pixel_table=pix,
        cldo4_file=args.cldo4_file, no2_file=args.no2_file,
        fit_order=spectral_fitting_order, wl_min_nm=args.wl_min, wl_max_nm=args.wl_max,
        output_dir=spectral_output_dir, tag=tag_m4,
        cf_threshold=args.cf_threshold,
    )
    lnT_plot_m4 = write_lnT_tau_examples(
        tau_result=tau_fit, fitting_result=fitting_result_m4,
        fit_order=spectral_fitting_order, output_dir=spectral_output_dir,
        n_examples=4, tag=tag_m4, obs_ln_T=obs_ln_T_fit,
        tau_simple=tau_mode4,
        tau_x_label="τ_gas+Rayleigh (O₂-B + H₂O + NO₂ + O₃ + Rayleigh)",
    )

    # ── Mode 3: gas only (O2-B + secondary gases) ────────────────────────────
    tag_m3 = f"_{chunk_tag}_o2b_simple_gas"
    fitting_result_m3 = fit_pixel_ensemble(
        tau_fit, pix, fit_order=spectral_fitting_order,
        obs_ln_T=obs_ln_T_fit, fit_mode="simple", tau_simple=tau_mode3,
    )
    write_spectral_fitting_outputs(fitting_result_m3, spectral_fitting_order, spectral_output_dir, tag=tag_m3)
    write_spectral_fitting_3panel_plot(
        fitting_result=fitting_result_m3, pixel_table=pix,
        cldo4_file=args.cldo4_file, no2_file=args.no2_file,
        fit_order=spectral_fitting_order, wl_min_nm=args.wl_min, wl_max_nm=args.wl_max,
        output_dir=spectral_output_dir, tag=tag_m3,
        cf_threshold=args.cf_threshold,
    )
    write_lnT_tau_examples(
        tau_result=tau_fit, fitting_result=fitting_result_m3,
        fit_order=spectral_fitting_order, output_dir=spectral_output_dir,
        n_examples=4, tag=tag_m3, obs_ln_T=obs_ln_T_fit,
        tau_simple=tau_mode3,
        tau_x_label="τ_gas (O₂-B + H₂O + NO₂ + O₃)",
    )

    # ── Mode 5: gas tau + linear-detrended residual lnT ──────────────────────
    residual_ln_T_fit = compute_residual_lnT(
        obs_ln_T=obs_ln_T_fit,
        wavelength_nm=tau_fit["wavelength_nm"],
    )
    tag_m5 = f"_{chunk_tag}_o2b_simple_gas_residual"
    fitting_result_m5 = fit_pixel_ensemble(
        tau_fit, pix, fit_order=spectral_fitting_order,
        obs_ln_T=residual_ln_T_fit, fit_mode="simple", tau_simple=tau_mode3,
    )
    write_spectral_fitting_outputs(fitting_result_m5, spectral_fitting_order, spectral_output_dir, tag=tag_m5)
    spectral_plot_m5 = write_spectral_fitting_3panel_plot(
        fitting_result=fitting_result_m5, pixel_table=pix,
        cldo4_file=args.cldo4_file, no2_file=args.no2_file,
        fit_order=spectral_fitting_order, wl_min_nm=args.wl_min, wl_max_nm=args.wl_max,
        output_dir=spectral_output_dir, tag=tag_m5,
        cf_threshold=args.cf_threshold,
    )
    lnT_plot_m5 = write_lnT_tau_examples(
        tau_result=tau_fit, fitting_result=fitting_result_m5,
        fit_order=spectral_fitting_order, output_dir=spectral_output_dir,
        n_examples=4, tag=tag_m5, obs_ln_T=residual_ln_T_fit,
        tau_simple=tau_mode3,
        tau_x_label="τ_gas (O₂-B + H₂O + NO₂ + O₃)",
    )
    residual_transmittance_plot = write_transmittance_wavelength_examples(
        tau_result=tau_fit, fitting_result=fitting_result_m5,
        output_dir=spectral_output_dir, n_examples=4,
        tag=tag_m5, obs_ln_T=residual_ln_T_fit,
    )

    valid_n = int(pix["valid_pixel"].sum())
    total_n = int(len(pix))
    print(f"Wrote: {pixel_csv}")
    print(f"Wrote: {profile_npz}")
    print(f"Wrote: {wavelength_npz}")
    print(f"Wrote: {slit_csv}")
    print(f"Wrote: {slit_npz}")
    print(f"Wrote: {tau_nc}")
    print(f"Wrote: {tau_npz}")
    print(f"Wrote: {o2b_debug_png}")
    print(f"Wrote: {tau_summary_csv}")
    print(f"Wrote: {reptran_nc}")
    print(f"Wrote: {reptran_csv}")
    print(f"Wrote: {reptran_gas_nc}")
    print(f"Wrote: {reptran_gas_csv}")
    print(f"Wrote: {reptran_summary_csv}")
    print(f"Wrote: {validation_csv}")
    print(f"Wrote: {validation_md}")
    print(f"Wrote: {spectral_paths['csv']}")
    print(f"Wrote: {spectral_paths['h5']}")
    print(f"Wrote: {spectral_paths['md']}")
    print(f"Wrote: {spectral_plot}")
    print(f"Wrote: {lnT_plot}")
    print(f"Wrote: {tau_components_plot}")
    print(f"Wrote: {tau_components_no_rayleigh_plot}")
    print(f"Wrote: {lnT_plot_m4}")
    print(f"Wrote: {transmittance_plot}")
    print(f"Wrote: {spectral_plot_m5}")
    print(f"Wrote: {lnT_plot_m5}")
    print(f"Wrote: {residual_transmittance_plot}")
    print(f"Valid pixels: {valid_n}/{total_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
python scripts/run_o2b_pipeline_subset.py \
    --lat-min 31.6 --lat-max 33.6 \
    --lon-min -123.1 --lon-max -121.1 \
    --wl-min 683 --wl-max 697 --spectral-fit-order 7 \
    --tag S010G09_190926_o2b

python scripts/run_o2b_pipeline_subset.py \
    --lat-min 31.6 --lat-max 33.6 \
    --lon-min -123.1 --lon-max -121.1 \
    --wl-min 683 --wl-max 697 --spectral-fit-order 7 \
    --rad-file data/TEMPO/TEMPO_RAD_L1_V03_20240708T160926Z_S007G09.nc \
    --cldo4-file data/TEMPO/TEMPO_CLDO4_L2_V03_20240708T160926Z_S007G09.nc \
    --no2-file data/TEMPO/TEMPO_NO2_L2_V03_20240708T160926Z_S007G09.nc \
    --irr-file data/TEMPO/TEMPO_IRR_L1_V03_20240711T042711Z.nc \
    --geos-file data/TEMPO/GEOS-CF.v01.rpl.sat_inst_1hr_r721x361_v72.20240708_1600z.nc4 \
    --tag S007G09_160926_o2b
"""
