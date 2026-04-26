#!/usr/bin/env python
"""Run the full TEMPO O2-O2 retrieval pipeline on a scan/xtrack subset.

Executes the complete staged workflow (WP1–WP7) on a user-defined spatial
and spectral subset: pixel table → GEOS profiles → slit kernels → tau
computation → REPTRAN projection → validation → spectral fitting.

Example:
python scripts/run_o2o2_pipeline_subset.py \
    --scan-start 0 --scan-end 1 --xtrack-start 0 --xtrack-end 128
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from wp1_pixel_table import PixelTableConfig, build_pixel_table
from wp2_profiles import ProfileConfig, build_profiles_for_pixels
from wp3_slit_kernel import SlitKernelConfig, build_slit_kernel_table
from wp4_tau import compute_tau_subset, write_o2o2_vertical_debug_plot, write_tau_netcdf
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
    """Parse the subset inputs, optional bbox, and output controls."""
    p = argparse.ArgumentParser()
    p.add_argument("--rad-file", default="data/TEMPO/TEMPO_RAD_L1_V03_20240708T190926Z_S010G09.nc")
    p.add_argument("--cldo4-file", default="data/TEMPO/TEMPO_CLDO4_L2_V03_20240708T190926Z_S010G09.nc")
    p.add_argument("--no2-file", default="data/TEMPO/TEMPO_NO2_L2_V03_20240708T190926Z_S010G09.nc")
    p.add_argument("--irr-file", default="data/TEMPO/TEMPO_IRR_L1_V03_20240711T042711Z.nc")
    p.add_argument("--geos-file", default="data/TEMPO/GEOS-CF.v01.rpl.sat_inst_1hr_r721x361_v72.20240708_1900z.nc4")
    p.add_argument("--scan-start", type=int, default=0,
                   help="First mirror step to read (default: 0 = start of granule).")
    p.add_argument("--scan-end", type=int, default=131,
                   help="Last mirror step (exclusive) to read (default: 131 = full granule).")
    p.add_argument("--xtrack-start", type=int, default=0,
                   help="First xtrack pixel to read (default: 0).")
    p.add_argument("--xtrack-end", type=int, default=2048,
                   help="Last xtrack pixel (exclusive) to read (default: 2048 = full swath).")
    p.add_argument("--wl-min", type=float, default=460.0)
    p.add_argument("--wl-max", type=float, default=490.0)
    p.add_argument("--lat-min", type=float, default=None)
    p.add_argument("--lat-max", type=float, default=None)
    p.add_argument("--lon-min", type=float, default=None)
    p.add_argument("--lon-max", type=float, default=None)
    p.add_argument(
        "--use-interest-region",
        action="store_true",
        help="Apply the preferred test bbox (lat 29.6-33.6, lon -123.8 to -120.5) if no bbox is supplied.",
    )
    p.add_argument(
        "--allow-wavecal-nonzero",
        action="store_true",
        help="Do not require wavecal_opt_status==0 in valid-pixel mask.",
    )
    p.add_argument("--out-dir", default="outputs/wp1_wp2")
    p.add_argument(
        "--tag",
        default="",
        help=(
            "Optional run tag. When set, outputs are written to tag-specific "
            "subfolders (e.g., outputs/qc/<tag>/)."
        ),
    )
    p.add_argument("--spectral-fit-order", type=int, default=2)
    p.add_argument("--cf-threshold", type=float, default=0.2,
                   help="Maximum cloud fraction for pixels to be included in spectral fitting.")
    return p.parse_args()


def main() -> int:
    """Run the full subset workflow and write the intermediate/native outputs."""
    args = parse_args()

    run_tag = str(args.tag).strip()
    if run_tag:
        # Keep tag as a single folder name, not an arbitrary path.
        run_tag = run_tag.replace(os.sep, "_")
        if os.altsep:
            run_tag = run_tag.replace(os.altsep, "_")

    run_out_dir = os.path.join(args.out_dir, run_tag) if run_tag else args.out_dir
    os.makedirs(run_out_dir, exist_ok=True)

    tau_native_dir = os.path.join(ROOT, "outputs", "tau_native", run_tag) if run_tag else os.path.join(ROOT, "outputs", "tau_native")
    tau_reptran_dir = os.path.join(ROOT, "outputs", "tau_reptran", run_tag) if run_tag else os.path.join(ROOT, "outputs", "tau_reptran")
    qc_dir = os.path.join(ROOT, "outputs", "qc", run_tag) if run_tag else os.path.join(ROOT, "outputs", "qc")
    os.makedirs(tau_native_dir, exist_ok=True)
    os.makedirs(tau_reptran_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    if args.use_interest_region and all(
        v is None for v in (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    ):
        args.lat_min = 29.6
        args.lat_max = 33.6
        args.lon_min = -123.8
        args.lon_max = -120.5

    # WP1: build the pixel table and corrected wavelength metadata.
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
        ),
    )

    base_valid_n = int(pix["valid_pixel"].sum())
    spatial_mask = pix["valid_pixel"].to_numpy(dtype=bool).copy()
    if args.lat_min is not None:
        spatial_mask &= pix["latitude"] >= args.lat_min
    if args.lat_max is not None:
        spatial_mask &= pix["latitude"] <= args.lat_max
    if args.lon_min is not None:
        spatial_mask &= pix["longitude"] >= args.lon_min
    if args.lon_max is not None:
        spatial_mask &= pix["longitude"] <= args.lon_max

    if base_valid_n == 0:
        print("Warning: no valid pixels found in the requested scan/xtrack subset before bbox filtering.")

    if not np.all(spatial_mask):
        pix = pd.DataFrame(pix.iloc[np.flatnonzero(spatial_mask)].copy().reset_index(drop=True))

    if len(pix) == 0:
        if base_valid_n > 0 and any(v is not None for v in (args.lat_min, args.lat_max, args.lon_min, args.lon_max)):
            print("Warning: bbox constraints removed all valid pixels from the requested subset.")
        else:
            print("Warning: requested scan/xtrack constraints produced no pixels in the subset footprint.")
        print("No valid pixels to process — exiting.")
        return 1

    # WP2: collocate each pixel to GEOS-CF and derive layer properties.
    prof = build_profiles_for_pixels(
        pixel_df=pix,
        geos_file=args.geos_file,
        config=ProfileConfig(),
    )

    # WP3: reconstruct the slit kernel used by the tau convolution.
    slit_df, slit_diag = build_slit_kernel_table(
        irr_file=args.irr_file,
        lambda_corrected_nm=wl_diag["lambda_corrected_nm"],
        xtrack_start=args.xtrack_start,
        xtrack_end=args.xtrack_end,
        config=SlitKernelConfig(
            wl_min_nm=args.wl_min,
            wl_max_nm=args.wl_max,
            require_wavecal_opt_status_zero=not args.allow_wavecal_nonzero,
        ),
    )

    # WP4: compute wavelength-resolved O2-O2 tau on the corrected grid.
    tau_result = compute_tau_subset(
        pixel_df=pix,
        profiles=prof,
        wavelength_diag=wl_diag,
        slit_df=slit_df,
        scan_start=args.scan_start,
        xtrack_start=args.xtrack_start,
    )

    reptran_result = project_tau_to_reptran(
        tau_result=tau_result,
        reptran_cfg=ReptranConfig(wavelength_min_nm=args.wl_min, wavelength_max_nm=args.wl_max),
    )
    reptran_gas_result = compute_tau_reptran_from_profiles(
        pixel_df=pix,
        profiles=prof,
        wavelength_diag=wl_diag,
        slit_df=slit_df,
        reptran_cfg=ReptranConfig(wavelength_min_nm=args.wl_min, wavelength_max_nm=args.wl_max),
        scan_start=args.scan_start,
        xtrack_start=args.xtrack_start,
    )

    validation_table = build_validation_table(
        pixel_df=pix,
        tau_result=tau_result,
        reptran_result=reptran_result,
    )

    chunk_tag = f"scan_{args.scan_start}_{args.scan_end}_xt_{args.xtrack_start}_{args.xtrack_end}"
    if args.lat_min is not None or args.lat_max is not None or args.lon_min is not None or args.lon_max is not None:
        chunk_tag += "_bbox"
    pixel_csv = os.path.join(run_out_dir, f"pixel_table_{chunk_tag}.csv")
    profile_npz = os.path.join(run_out_dir, f"profiles_{chunk_tag}.npz")
    wavelength_npz = os.path.join(run_out_dir, f"wavelength_diag_{chunk_tag}.npz")
    slit_csv = os.path.join(run_out_dir, f"slit_kernel_{chunk_tag}.csv")
    slit_npz = os.path.join(run_out_dir, f"slit_kernel_{chunk_tag}.npz")
    tau_nc = os.path.join(tau_native_dir, f"tau_o2o2_native_{chunk_tag}.nc")
    tau_npz = os.path.join(run_out_dir, f"tau_native_{chunk_tag}.npz")
    tau_summary_csv = os.path.join(run_out_dir, f"tau_summary_{chunk_tag}.csv")
    reptran_prefix = os.path.join(tau_reptran_dir, f"tau_o2o2_reptran_{chunk_tag}")
    reptran_gas_prefix = os.path.join(tau_reptran_dir, f"tau_gas_reptran_{chunk_tag}")
    reptran_summary_csv = os.path.join(run_out_dir, f"tau_reptran_summary_{chunk_tag}.csv")
    validation_prefix = os.path.join(qc_dir, f"tau_validation_{chunk_tag}")

    # Write all intermediate and final artifacts so each stage can be checked.
    pix.to_csv(pixel_csv, index=False)
    np.savez_compressed(profile_npz, **prof)
    np.savez_compressed(wavelength_npz, **wl_diag)
    np.savez_compressed(slit_npz, **slit_diag)
    slit_df.to_csv(slit_csv, index=False)
    np.savez_compressed(tau_npz, **tau_result)
    write_tau_netcdf(tau_nc, tau_result, pix)
    o2o2_vertical_debug_png = os.path.join(qc_dir, f"o2o2_vertical_debug_{chunk_tag}.png")
    o2o2_vertical_debug_png = write_o2o2_vertical_debug_plot(
        out_path=o2o2_vertical_debug_png,
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
    reptran_nc, reptran_csv = write_reptran_outputs(reptran_prefix, reptran_result, reptran_pixel_df)
    reptran_gas_nc, reptran_gas_csv = write_reptran_gas_outputs(reptran_gas_prefix, reptran_gas_result, pix)

    if tau_result["pixel_index"].size:
        tau_summary = pd.DataFrame(
            {
                "pixel_index": tau_result["pixel_index"],
                "mirror_step": tau_result["mirror_step"],
                "xtrack": tau_result["xtrack"],
                "sec_airmass": tau_result["sec_airmass"],
                "tau_band_mean": tau_result["tau_band_mean"],
            }
        )
    else:
        tau_summary = pd.DataFrame(columns=["pixel_index", "mirror_step", "xtrack", "sec_airmass", "tau_band_mean"])
    tau_summary.to_csv(tau_summary_csv, index=False)

    reptran_summary = pd.DataFrame(
        {
            "band_index": reptran_result["band_index"],
            "band_name": reptran_result["band_name"],
            "rep_wavelength_nm": reptran_result["rep_wavelength_nm"],
        }
    )
    reptran_summary.to_csv(reptran_summary_csv, index=False)

    validation_csv, validation_md = write_validation_outputs(validation_prefix, validation_table)

    # ── WP7: Spectral fitting (cumulant expansion tau-transmittance model) ──
    spectral_fitting_order = int(args.spectral_fit_order)
    obs_ln_T_native = extract_observed_lnT(
        tau_result=tau_result,
        pixel_table=pix,
        rad_file=args.rad_file,
        irr_file=args.irr_file,
        wavelength_diag=wl_diag,
        scan_start=args.scan_start,
        xtrack_start=args.xtrack_start,
    )

    # Build total tau_eff on REPTRAN wavelengths for fitting:
    # tau_total = tau_o2o2 + tau_rayleigh + tau_all_gases
    tau_rayleigh_proj = project_tau_to_reptran(
        tau_result={
            "wavelength_nm": tau_result["wavelength_nm"],
            "tau_eff": tau_result["tau_rayleigh_eff"],
        },
        reptran_cfg=ReptranConfig(wavelength_min_nm=args.wl_min, wavelength_max_nm=args.wl_max),
    )

    # REPTRAN rep wavelengths — used only for gas-tau interpolation and diagnostics
    rep_wl = np.asarray(reptran_gas_result["rep_wavelength_nm"], dtype=float)
    # Sort once for np.interp (requires monotonically increasing x)
    rep_sort = np.argsort(rep_wl)
    rep_wl_sorted = rep_wl[rep_sort]

    # Ring template on REPTRAN grid — kept for diagnostic plots only
    ring_template_rep = build_ring_template_from_irr(
        irr_file=args.irr_file,
        rep_wavelength_nm=rep_wl,
    )
    # Ring on native TEMPO grid — used for fitting
    ring_native = compute_ring_optical_depth(tau_result, args.irr_file)

    o2_idx = np.asarray(tau_result["pixel_index"], dtype=int)
    gas_idx = np.asarray(reptran_gas_result["pixel_index"], dtype=int)
    idx_to_o2_row = {int(p): i for i, p in enumerate(o2_idx)}
    idx_to_gas_row = {int(p): i for i, p in enumerate(gas_idx)}
    common_idx = sorted(set(int(p) for p in o2_idx).intersection(int(p) for p in gas_idx))
    quality_ok_mask = (
        pix["quality_ok"].to_numpy(dtype=bool)
        if "quality_ok" in pix.columns
        else np.ones(len(pix), dtype=bool)
    )
    pix_cf = pix["cldo4_cloud_fraction"].to_numpy(dtype=float)
    cf_ok_mask = np.isfinite(pix_cf) & (pix_cf <= args.cf_threshold)
    skipped_bad_quality = 0
    skipped_cloudy = 0

    fit_pixel_idx = []
    fit_tau_gas_rows = []          # Mode 3: O2-O2 + NO2 + O3 + H2O (gas only, no Rayleigh/Ring)
    fit_tau_rayleigh_rows = []     # Mode 4: gas + Rayleigh (no Ring)
    fit_tau_total_rows = []        # Mode 2: gas + Rayleigh + Ring
    fit_wl_rows = []
    fit_lnT_rows = []
    fit_ring_rows = []
    for pidx in common_idx:
        if not bool(quality_ok_mask[pidx]):
            skipped_bad_quality += 1
            continue
        if not bool(cf_ok_mask[pidx]):
            skipped_cloudy += 1
            continue

        o2_row = idx_to_o2_row[pidx]
        gas_row = idx_to_gas_row[pidx]

        # All components on native TEMPO wavelength grid
        wl_native = np.asarray(tau_result["wavelength_nm"][o2_row], dtype=float)
        lnT_row    = np.asarray(obs_ln_T_native[o2_row], dtype=float)
        tau_o2o2   = np.asarray(tau_result["tau_eff"][o2_row], dtype=float)
        tau_ray    = np.asarray(tau_result["tau_rayleigh_eff"][o2_row], dtype=float)
        tau_ring   = np.asarray(ring_native[o2_row], dtype=float)

        # Mask channels outside [wl_min, wl_max] or with invalid wavelength
        in_range = np.isfinite(wl_native) & (wl_native >= args.wl_min) & (wl_native <= args.wl_max)
        if np.count_nonzero(in_range & np.isfinite(lnT_row)) < 2:
            continue

        # NaN-mask out-of-range channels so fit functions skip them via valid_mask
        wl_out     = np.where(in_range, wl_native, np.nan)
        lnT_out    = np.where(in_range, lnT_row,   np.nan)
        tau_o2o2   = np.where(in_range, tau_o2o2,  np.nan)
        tau_ray    = np.where(in_range, tau_ray,    np.nan)
        tau_ring   = np.where(in_range, tau_ring,   np.nan)

        # Interpolate other-gas taus (NO2 + O3 + H2O) from REPTRAN to native TEMPO grid
        tau_other_gas_rep = np.asarray(
            reptran_gas_result["tau_total_gas"][gas_row], dtype=float
        )[rep_sort]
        tau_other_gas = np.full_like(wl_native, np.nan)
        interp_mask = in_range & (wl_native >= rep_wl_sorted[0]) & (wl_native <= rep_wl_sorted[-1])
        if interp_mask.any():
            tau_other_gas[interp_mask] = np.interp(
                wl_native[interp_mask], rep_wl_sorted, tau_other_gas_rep
            )

        # Build combined taus on native TEMPO grid
        tau_gas_eff   = tau_o2o2 + tau_other_gas             # Mode 3: all gases
        tau_ray_eff   = tau_gas_eff + tau_ray                 # Mode 4: + Rayleigh
        tau_total_eff = tau_ray_eff + np.where(               # Mode 2: + Ring
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
            "wavelength_nm": np.vstack(fit_wl_rows),   # native TEMPO grid per pixel
            "tau_eff": np.vstack(fit_tau_gas_rows),     # gas-only sum (composite + Mode 3)
            "ring_basis": np.vstack(fit_ring_rows),     # Ring on native TEMPO grid
        }
        tau_mode2 = np.vstack(fit_tau_total_rows)    # gas + Rayleigh + Ring (Mode 2)
        tau_mode3 = tau_fit["tau_eff"]               # gas only (Mode 3)
        tau_mode4 = np.vstack(fit_tau_rayleigh_rows) # gas + Rayleigh, no Ring (Mode 4)
        obs_ln_T_fit = np.vstack(fit_lnT_rows)
    else:
        tau_fit = {
            "pixel_index": np.array([], dtype=int),
            "wavelength_nm": np.empty((0, 0), dtype=float),
            "tau_eff": np.empty((0, 0), dtype=float),
            "ring_basis": np.empty((0, 0), dtype=float),
        }
        tau_mode2 = np.empty((0, 0), dtype=float)
        tau_mode3 = np.empty((0, 0), dtype=float)
        tau_mode4 = np.empty((0, 0), dtype=float)
        obs_ln_T_fit = np.empty((0, 0), dtype=float)

    if skipped_bad_quality > 0:
        print(f"Skipped {skipped_bad_quality} bad-quality pixels before spectral fitting.")
    if skipped_cloudy:
        print(f"Skipped {skipped_cloudy} cloudy pixels (CF > {args.cf_threshold}) before spectral fitting.")

    fitting_result = fit_pixel_ensemble(
        tau_fit, pix, fit_order=spectral_fitting_order, obs_ln_T=obs_ln_T_fit
    )
    spectral_output_dir = qc_dir
    spectral_tag = f"_{chunk_tag}_tau_total_eff"
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

    # ── Mode 2: simple cumulant, total tau (O2-O2 + NO2 + O3 + H2O + Rayleigh + Ring) ──
    tag_m2 = f"_{chunk_tag}_simple_total"
    fitting_result_m2 = fit_pixel_ensemble(
        tau_fit, pix,
        fit_order=spectral_fitting_order,
        obs_ln_T=obs_ln_T_fit,
        fit_mode="simple",
        tau_simple=tau_mode2,
    )
    write_spectral_fitting_outputs(
        fitting_result_m2, spectral_fitting_order, spectral_output_dir, tag=tag_m2
    )
    write_spectral_fitting_3panel_plot(
        fitting_result=fitting_result_m2,
        pixel_table=pix,
        cldo4_file=args.cldo4_file,
        no2_file=args.no2_file,
        fit_order=spectral_fitting_order,
        wl_min_nm=args.wl_min,
        wl_max_nm=args.wl_max,
        output_dir=spectral_output_dir,
        tag=tag_m2,
        cf_threshold=args.cf_threshold,
    )
    write_lnT_tau_examples(
        tau_result=tau_fit,
        fitting_result=fitting_result_m2,
        fit_order=spectral_fitting_order,
        output_dir=spectral_output_dir,
        n_examples=4,
        tag=tag_m2,
        obs_ln_T=obs_ln_T_fit,
        tau_simple=tau_mode2,
        tau_x_label="τ_total (O₂-O₂ + NO₂ + O₃ + H₂O + Rayleigh + Ring)",
    )

    # ── Mode 4: simple cumulant, gas + Rayleigh (no Ring) ────────────────────
    tag_m4 = f"_{chunk_tag}_simple_gas_rayleigh"
    fitting_result_m4 = fit_pixel_ensemble(
        tau_fit, pix,
        fit_order=spectral_fitting_order,
        obs_ln_T=obs_ln_T_fit,
        fit_mode="simple",
        tau_simple=tau_mode4,
    )
    write_spectral_fitting_outputs(
        fitting_result_m4, spectral_fitting_order, spectral_output_dir, tag=tag_m4
    )
    write_spectral_fitting_3panel_plot(
        fitting_result=fitting_result_m4,
        pixel_table=pix,
        cldo4_file=args.cldo4_file,
        no2_file=args.no2_file,
        fit_order=spectral_fitting_order,
        wl_min_nm=args.wl_min,
        wl_max_nm=args.wl_max,
        output_dir=spectral_output_dir,
        tag=tag_m4,
        cf_threshold=args.cf_threshold,
    )
    lnT_plot_m4 = write_lnT_tau_examples(
        tau_result=tau_fit,
        fitting_result=fitting_result_m4,
        fit_order=spectral_fitting_order,
        output_dir=spectral_output_dir,
        n_examples=4,
        tag=tag_m4,
        obs_ln_T=obs_ln_T_fit,
        tau_simple=tau_mode4,
        tau_x_label="τ_gas+Rayleigh (O₂-O₂ + NO₂ + O₃ + H₂O + Rayleigh)",
    )

    # ── Mode 3: simple cumulant, total gas tau (O2-O2 + NO2 + O3 + H2O) ──────
    tag_m3 = f"_{chunk_tag}_simple_gas"
    fitting_result_m3 = fit_pixel_ensemble(
        tau_fit, pix,
        fit_order=spectral_fitting_order,
        obs_ln_T=obs_ln_T_fit,
        fit_mode="simple",
        tau_simple=tau_mode3,
    )
    write_spectral_fitting_outputs(
        fitting_result_m3, spectral_fitting_order, spectral_output_dir, tag=tag_m3
    )
    write_spectral_fitting_3panel_plot(
        fitting_result=fitting_result_m3,
        pixel_table=pix,
        cldo4_file=args.cldo4_file,
        no2_file=args.no2_file,
        fit_order=spectral_fitting_order,
        wl_min_nm=args.wl_min,
        wl_max_nm=args.wl_max,
        output_dir=spectral_output_dir,
        tag=tag_m3,
        cf_threshold=args.cf_threshold,
    )
    write_lnT_tau_examples(
        tau_result=tau_fit,
        fitting_result=fitting_result_m3,
        fit_order=spectral_fitting_order,
        output_dir=spectral_output_dir,
        n_examples=4,
        tag=tag_m3,
        obs_ln_T=obs_ln_T_fit,
        tau_simple=tau_mode3,
        tau_x_label="τ_gas (O₂-O₂ + NO₂ + O₃ + H₂O)",
    )

    # ── Mode 5: simple cumulant, gas tau + linear-detrended residual ln(T) ──
    residual_ln_T_fit = compute_residual_lnT(
        obs_ln_T=obs_ln_T_fit,
        wavelength_nm=tau_fit["wavelength_nm"],
    )
    tag_m5 = f"_{chunk_tag}_simple_gas_residual"
    fitting_result_m5 = fit_pixel_ensemble(
        tau_fit, pix,
        fit_order=spectral_fitting_order,
        obs_ln_T=residual_ln_T_fit,
        fit_mode="simple",
        tau_simple=tau_mode3,
    )
    write_spectral_fitting_outputs(
        fitting_result_m5, spectral_fitting_order, spectral_output_dir, tag=tag_m5
    )
    spectral_plot_m5 = write_spectral_fitting_3panel_plot(
        fitting_result=fitting_result_m5,
        pixel_table=pix,
        cldo4_file=args.cldo4_file,
        no2_file=args.no2_file,
        fit_order=spectral_fitting_order,
        wl_min_nm=args.wl_min,
        wl_max_nm=args.wl_max,
        output_dir=spectral_output_dir,
        tag=tag_m5,
        cf_threshold=args.cf_threshold,
    )
    lnT_plot_m5 = write_lnT_tau_examples(
        tau_result=tau_fit,
        fitting_result=fitting_result_m5,
        fit_order=spectral_fitting_order,
        output_dir=spectral_output_dir,
        n_examples=4,
        tag=tag_m5,
        obs_ln_T=residual_ln_T_fit,
        tau_simple=tau_mode3,
        tau_x_label="τ_gas (O₂-O₂ + NO₂ + O₃ + H₂O)",
    )
    residual_transmittance_plot = write_transmittance_wavelength_examples(
        tau_result=tau_fit,
        fitting_result=fitting_result_m5,
        output_dir=spectral_output_dir,
        n_examples=4,
        tag=tag_m5,
        obs_ln_T=residual_ln_T_fit,
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
    print(f"Wrote: {o2o2_vertical_debug_png}")
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
python scripts/run_o2o2_pipeline_subset.py \
    --lat-min 31.6 --lat-max 33.6 \
    --lon-min -123.1 --lon-max -121.1 \
    --wl-min 462 --wl-max 488 --spectral-fit-order 2 \
    --tag S010G09_190926


python scripts/run_o2o2_pipeline_subset.py \
    --lat-min 31.6 --lat-max 33.6 \
    --lon-min -123.1 --lon-max -121.1 \
    --wl-min 462 --wl-max 488 --spectral-fit-order 2\
    --rad-file data/TEMPO/TEMPO_RAD_L1_V03_20240708T160926Z_S007G09.nc \
    --cldo4-file data/TEMPO/TEMPO_CLDO4_L2_V03_20240708T160926Z_S007G09.nc \
    --no2-file data/TEMPO/TEMPO_NO2_L2_V03_20240708T160926Z_S007G09.nc \
    --irr-file data/TEMPO/TEMPO_IRR_L1_V03_20240711T042711Z.nc \
    --geos-file data/TEMPO/GEOS-CF.v01.rpl.sat_inst_1hr_r721x361_v72.20240708_1600z.nc4 \
    --tag S007G09_160926

"""