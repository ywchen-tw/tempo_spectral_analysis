#!/usr/bin/env python
"""Multi-order spectral fitting convergence and physics validation.

Loads WP4 tau output (.npz) and WP1 pixel table (.csv), fits cumulant
expansion models at orders 2, 3, 5, 7, and produces:
  - convergence_table.csv   — K1/K2 mean±std per order
  - convergence_k1_k2.png   — stability plot across orders
  - k1_k2_vs_physics.png    — scatter vs cloud/surface pressure, airmass
  - analysis_summary.md     — text summary

Usage
-----
python scripts/analyze_spectral_fitting.py \\
    --tau-npz  outputs/wp1_wp2/tau_native_scan_0_1_xt_0_128.npz \\
    --pixel-csv outputs/wp1_wp2/pixel_table_scan_0_1_xt_0_128.csv \\
    --out-dir  outputs/analysis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from wp7_spectral_fitting import (
    extract_observed_lnT,
    fit_pixel_ensemble,
    write_lnT_tau_examples,
    write_spectral_fitting_outputs,
)

CONVERGENCE_ORDERS = [2, 3, 5, 7]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tau-npz",
        default="outputs/wp1_wp2/tau_native_scan_0_1_xt_0_128.npz",
        help="Path to tau_result .npz from run_wp1_wp2_subset.py",
    )
    p.add_argument(
        "--pixel-csv",
        default="outputs/wp1_wp2/pixel_table_scan_0_1_xt_0_128.csv",
        help="Pixel table .csv from WP1",
    )
    p.add_argument("--out-dir", default="outputs/analysis")
    p.add_argument(
        "--orders",
        nargs="+",
        type=int,
        default=CONVERGENCE_ORDERS,
        help="Cumulant orders to test (default: 2 3 5 7)",
    )
    # Optional inputs for physically correct observed ln(T)
    p.add_argument("--rad-file", default=None,
                   help="TEMPO RAD L1 file; required for observed ln(T) fitting.")
    p.add_argument("--irr-file", default=None,
                   help="TEMPO IRR L1 file; required for observed ln(T) fitting.")
    p.add_argument(
        "--wavelength-npz", default=None,
        help="wavelength_diag .npz saved by run_wp1_wp2_subset.py (contains "
             "channel_mask and lambda_corrected_nm).",
    )
    p.add_argument("--scan-start", type=int, default=0,
                   help="Mirror step offset used when the wavelength_diag subset was built.")
    p.add_argument("--xtrack-start", type=int, default=0,
                   help="Xtrack offset used when the wavelength_diag subset was built.")
    return p.parse_args()


def load_tau_npz(path: str) -> dict:
    """Load a compressed tau_result .npz into a plain dict of numpy arrays."""
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def fit_all_orders(
    tau_result: dict,
    pixel_table: pd.DataFrame,
    orders: list[int],
    obs_ln_T=None,
) -> dict[int, dict]:
    """Run fit_pixel_ensemble for each order, return {order: fitting_result}."""
    results = {}
    for order in orders:
        print(f"  Fitting order {order} ...", flush=True)
        results[order] = fit_pixel_ensemble(
            tau_result, pixel_table, fit_order=order, obs_ln_T=obs_ln_T
        )
    return results


def _kappa_stats(fitting_result: dict, col: int) -> tuple[float, float, float]:
    """Return (mean, std, success_rate) for kappa column `col`."""
    mask = np.asarray(fitting_result["fit_success"], dtype=bool)
    if mask.sum() == 0:
        return np.nan, np.nan, 0.0
    vals = fitting_result["kappas"][mask, col]
    return float(np.nanmean(vals)), float(np.nanstd(vals)), float(mask.mean())


def write_convergence_table(
    fitting_by_order: dict[int, dict],
    out_dir: Path,
) -> Path:
    rows = []
    for order, fr in fitting_by_order.items():
        k1_mean, k1_std, rate = _kappa_stats(fr, 0)
        k2_mean, k2_std, _ = _kappa_stats(fr, 1) if order >= 2 else (np.nan, np.nan, None)
        rows.append(
            {
                "order": order,
                "n_pixels": int(len(fr["fit_success"])),
                "n_success": int(np.sum(fr["fit_success"])),
                "success_rate": rate,
                "k1_mean": k1_mean,
                "k1_std": k1_std,
                "k2_mean": k2_mean,
                "k2_std": k2_std,
            }
        )
    df = pd.DataFrame(rows)
    out_path = out_dir / "convergence_table.csv"
    df.to_csv(out_path, index=False)
    return out_path


def plot_convergence(fitting_by_order: dict[int, dict], out_dir: Path) -> Path:
    """Two-row plot: K1 and K2 mean±std vs cumulant order."""
    orders = sorted(fitting_by_order.keys())
    k1_means, k1_stds = [], []
    k2_means, k2_stds = [], []
    for order in orders:
        fr = fitting_by_order[order]
        m1, s1, _ = _kappa_stats(fr, 0)
        m2, s2, _ = _kappa_stats(fr, 1) if order >= 2 else (np.nan, np.nan, None)
        k1_means.append(m1); k1_stds.append(s1)
        k2_means.append(m2); k2_stds.append(s2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax1.errorbar(orders, k1_means, yerr=k1_stds, fmt="o-", capsize=4, color="steelblue")
    ax1.axhline(1.0, color="gray", lw=0.8, ls="--", label="expected K1=1")
    ax1.set_xlabel("Cumulant order")
    ax1.set_ylabel("K1 mean ± std")
    ax1.set_title("K1 convergence across orders")
    ax1.legend()
    ax1.set_xticks(orders)

    ax2.errorbar(orders, k2_means, yerr=k2_stds, fmt="s-", capsize=4, color="darkorange")
    ax2.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("Cumulant order")
    ax2.set_ylabel("K2 mean ± std")
    ax2.set_title("K2 convergence across orders")
    ax2.set_xticks(orders)

    out_path = out_dir / "convergence_k1_k2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_physics_validation(
    fitting_result: dict,
    pixel_table: pd.DataFrame,
    out_dir: Path,
    fit_order: int = 2,
) -> Path:
    """Scatter K1/K2 vs cloud pressure, surface pressure, and sec(airmass)."""
    pixel_idx = np.asarray(fitting_result["pixel_index"], dtype=int)
    success = np.asarray(fitting_result["fit_success"], dtype=bool)
    if success.sum() == 0:
        out_path = out_dir / f"k1_k2_vs_physics_order{fit_order}.png"
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No fitted pixels", ha="center", va="center")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    pix_sub = pixel_table.iloc[pixel_idx].reset_index(drop=True)
    k1 = fitting_result["kappas"][:, 0]
    k2 = fitting_result["kappas"][:, 1] if fitting_result["kappas"].shape[1] >= 2 else np.full_like(k1, np.nan)

    cloud_p = pix_sub["cldo4_cloud_pressure_hpa"].to_numpy(dtype=float)
    sfc_p = pix_sub["cldo4_surface_pressure_hpa"].to_numpy(dtype=float)
    sza = pix_sub["sza_deg"].to_numpy(dtype=float)
    vza = pix_sub["vza_deg"].to_numpy(dtype=float)
    sec_am = 1.0 / np.cos(np.deg2rad(sza)) + 1.0 / np.cos(np.deg2rad(vza))

    # Only plot successfully fitted pixels
    m = success
    physics_cols = [
        (cloud_p[m], "Cloud pressure (hPa)"),
        (sfc_p[m], "Surface pressure (hPa)"),
        (sec_am[m], "sec(SZA) + sec(VZA)"),
    ]
    kappa_rows = [
        (k1[m], f"K1 (order {fit_order})"),
        (k2[m], f"K2 (order {fit_order})"),
    ]

    n_rows, n_cols = len(kappa_rows), len(physics_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), constrained_layout=True)
    for r, (kvals, klabel) in enumerate(kappa_rows):
        for c, (pvals, plabel) in enumerate(physics_cols):
            ax = axes[r, c]
            finite = np.isfinite(kvals) & np.isfinite(pvals)
            ax.scatter(pvals[finite], kvals[finite], s=12, alpha=0.6, color="steelblue" if r == 0 else "darkorange")
            if finite.sum() >= 2:
                corr = float(np.corrcoef(pvals[finite], kvals[finite])[0, 1])
                ax.set_title(f"r={corr:.3f}", fontsize=9)
            ax.set_xlabel(plabel, fontsize=8)
            ax.set_ylabel(klabel, fontsize=8)

    out_path = out_dir / f"k1_k2_vs_physics_order{fit_order}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_summary_md(
    fitting_by_order: dict[int, dict],
    convergence_csv: Path,
    convergence_png: Path,
    physics_png: Path,
    out_dir: Path,
) -> Path:
    df = pd.read_csv(convergence_csv)
    lines = [
        "# Spectral Fitting Convergence Analysis",
        "",
        "## Convergence Table (K1/K2 mean±std across cumulant orders)",
        "",
        df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
    ]

    baseline_row = df[df["order"] == 2]
    if not baseline_row.empty:
        k1_mean = float(baseline_row["k1_mean"].iloc[0])
        k1_std = float(baseline_row["k1_std"].iloc[0])
        lines.append(
            f"- **K1 (order 2)**: {k1_mean:.4f} ± {k1_std:.4e}  "
            f"{'(stable, ~1.0 as expected)' if abs(k1_mean - 1.0) < 0.05 else '(NOTE: deviates from expected ~1.0)'}"
        )

    k1_spread = df["k1_mean"].max() - df["k1_mean"].min()
    lines.append(
        f"- **K1 spread across orders**: {k1_spread:.4e}  "
        f"{'— stable convergence' if k1_spread < 0.01 else '— notable sensitivity to order'}"
    )

    k2_spread = df["k2_mean"].max() - df["k2_mean"].min()
    lines.append(
        f"- **K2 spread across orders**: {k2_spread:.4e}  "
        f"{'— K2 is order-sensitive (expected for non-linear effects)' if k2_spread > 0.05 else '— K2 stable'}"
    )
    lines += [
        "",
        f"See [{convergence_png.name}]({convergence_png.name}) for convergence plot.",
        f"See [{physics_png.name}]({physics_png.name}) for K1/K2 vs cloud/pressure scatter.",
    ]

    out_path = out_dir / "analysis_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tau_npz_path = os.path.join(ROOT, args.tau_npz) if not os.path.isabs(args.tau_npz) else args.tau_npz
    pixel_csv_path = os.path.join(ROOT, args.pixel_csv) if not os.path.isabs(args.pixel_csv) else args.pixel_csv

    print(f"Loading tau result: {tau_npz_path}")
    tau_result = load_tau_npz(tau_npz_path)
    n_pix = int(tau_result["tau_eff"].shape[0]) if tau_result["tau_eff"].ndim == 2 else 0
    print(f"  {n_pix} pixels, {tau_result['tau_eff'].shape[-1] if n_pix else 0} channels")

    print(f"Loading pixel table: {pixel_csv_path}")
    pixel_table = pd.read_csv(pixel_csv_path)

    # Compute observed ln(T) when RAD/IRR files and wavelength_diag are supplied
    obs_ln_T = None
    if args.rad_file and args.irr_file and args.wavelength_npz:
        wl_diag_path = (
            os.path.join(ROOT, args.wavelength_npz)
            if not os.path.isabs(args.wavelength_npz)
            else args.wavelength_npz
        )
        print(f"Loading wavelength diag: {wl_diag_path}")
        wl_diag_data = np.load(wl_diag_path, allow_pickle=False)
        wavelength_diag = {k: wl_diag_data[k] for k in wl_diag_data.files}
        rad_path = (
            os.path.join(ROOT, args.rad_file) if not os.path.isabs(args.rad_file) else args.rad_file
        )
        irr_path = (
            os.path.join(ROOT, args.irr_file) if not os.path.isabs(args.irr_file) else args.irr_file
        )
        print(f"Extracting observed ln(T) from RAD and IRR files ...")
        obs_ln_T = extract_observed_lnT(
            tau_result=tau_result,
            pixel_table=pixel_table,
            rad_file=rad_path,
            irr_file=irr_path,
            wavelength_diag=wavelength_diag,
            scan_start=args.scan_start,
            xtrack_start=args.xtrack_start,
        )
    else:
        print(
            "Warning: --rad-file, --irr-file, and --wavelength-npz not all provided; "
            "fitting will use synthetic ln(T) = −τ (not physically meaningful)."
        )

    orders = sorted(set(args.orders))
    print(f"Fitting orders: {orders}")
    fitting_by_order = fit_all_orders(tau_result, pixel_table, orders, obs_ln_T=obs_ln_T)

    for order, fr in fitting_by_order.items():
        write_spectral_fitting_outputs(fr, order, out_dir)
        n_ok = int(np.sum(fr["fit_success"]))
        k1_m, k1_s, _ = _kappa_stats(fr, 0)
        k2_m, k2_s, _ = _kappa_stats(fr, 1) if order >= 2 else (np.nan, np.nan, None)
        print(
            f"  Order {order}: {n_ok}/{len(fr['fit_success'])} fit  "
            f"K1={k1_m:.4f}±{k1_s:.2e}  K2={k2_m:.4f}±{k2_s:.2e}"
        )

    convergence_csv = write_convergence_table(fitting_by_order, out_dir)
    convergence_png = plot_convergence(fitting_by_order, out_dir)

    baseline_order = orders[0]
    physics_png = plot_physics_validation(
        fitting_by_order[baseline_order], pixel_table, out_dir, fit_order=baseline_order
    )

    # ln(T) vs tau example plots for each fitted order
    lnT_plots = []
    for order, fr in fitting_by_order.items():
        p = write_lnT_tau_examples(
            tau_result=tau_result,
            fitting_result=fr,
            fit_order=order,
            output_dir=out_dir,
            n_examples=4,
            obs_ln_T=obs_ln_T,
        )
        lnT_plots.append(p)
        print(f"Wrote: {p}")

    summary_md = write_summary_md(
        fitting_by_order, convergence_csv, convergence_png, physics_png, out_dir
    )

    print(f"Wrote: {convergence_csv}")
    print(f"Wrote: {convergence_png}")
    print(f"Wrote: {physics_png}")
    print(f"Wrote: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
