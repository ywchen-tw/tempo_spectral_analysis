#!/usr/bin/env python
"""Sensitivity study: K1/K2 consistency across bbox and xtrack range variations.

Runs run_wp1_wp2_subset.py for each test configuration, loads the spectral
fitting results, and produces a comparison table and plot showing K1/K2
stability across spatial subsets.

Usage
-----
python scripts/run_sensitivity_study.py [--out-dir outputs/sensitivity]

Each configuration is re-run from scratch, so the study is reproducible. Runs
that already have output are skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

RUNNER = os.path.join(ROOT, "scripts", "run_wp1_wp2_subset.py")

# Each entry defines one subset run; keys match run_wp1_wp2_subset.py CLI args.
SENSITIVITY_CONFIGS = [
    {
        "label": "full_xt_no_bbox",
        "scan_start": 0, "scan_end": 1,
        "xtrack_start": 0, "xtrack_end": 128,
    },
    {
        "label": "xt_left_half",
        "scan_start": 0, "scan_end": 1,
        "xtrack_start": 0, "xtrack_end": 64,
    },
    {
        "label": "xt_right_half",
        "scan_start": 0, "scan_end": 1,
        "xtrack_start": 64, "xtrack_end": 128,
    },
    {
        "label": "alaska_bbox",
        "scan_start": 0, "scan_end": 1,
        "xtrack_start": 0, "xtrack_end": 128,
        "lat_min": 62.4, "lat_max": 62.9,
        "lon_min": -154.6, "lon_max": -153.0,
        "allow_wavecal_nonzero": True,
    },
    {
        "label": "california_bbox",
        "scan_start": 0, "scan_end": 1,
        "xtrack_start": 0, "xtrack_end": 128,
        "lat_min": 29.6, "lat_max": 33.6,
        "lon_min": -123.8, "lon_max": -120.5,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="outputs/sensitivity")
    p.add_argument(
        "--spectral-fit-order", type=int, default=2,
        help="Cumulant order used for the runner's built-in spectral fitting."
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-run even if output already exists."
    )
    return p.parse_args()


def config_to_cli(cfg: dict, run_out_dir: str, fit_order: int) -> list[str]:
    """Build the argv list for run_wp1_wp2_subset.py from a config dict."""
    cmd = [
        sys.executable, RUNNER,
        "--scan-start", str(cfg["scan_start"]),
        "--scan-end", str(cfg["scan_end"]),
        "--xtrack-start", str(cfg["xtrack_start"]),
        "--xtrack-end", str(cfg["xtrack_end"]),
        "--out-dir", run_out_dir,
        "--spectral-fit-order", str(fit_order),
    ]
    for key in ("lat_min", "lat_max", "lon_min", "lon_max"):
        if key in cfg:
            cmd += [f"--{key.replace('_', '-')}", str(cfg[key])]
    if cfg.get("allow_wavecal_nonzero"):
        cmd.append("--allow-wavecal-nonzero")
    return cmd


def chunk_tag(cfg: dict) -> str:
    tag = f"scan_{cfg['scan_start']}_{cfg['scan_end']}_xt_{cfg['xtrack_start']}_{cfg['xtrack_end']}"
    if any(k in cfg for k in ("lat_min", "lat_max", "lon_min", "lon_max")):
        tag += "_bbox"
    return tag


def find_fitting_csv(run_out_dir: str, fit_order: int) -> Path | None:
    p = Path(run_out_dir) / f"spectral_fitting_order{fit_order}.csv"
    if p.exists():
        return p
    # Runner writes fitting output to outputs/qc/, but we also write it here
    return None


def load_fitting_stats(csv_path: Path, fit_order: int) -> dict:
    """Return summary stats for a single config's fitting CSV."""
    df = pd.read_csv(csv_path)
    ok = df["fit_success"].astype(bool)
    stats = {
        "n_pixels": len(df),
        "n_success": int(ok.sum()),
        "success_rate": float(ok.mean()),
    }
    for i in range(1, fit_order + 1):
        col = f"k{i}"
        if col in df.columns:
            vals = df.loc[ok, col]
            stats[f"k{i}_mean"] = float(vals.mean())
            stats[f"k{i}_std"] = float(vals.std())
        else:
            stats[f"k{i}_mean"] = np.nan
            stats[f"k{i}_std"] = np.nan
    return stats


def plot_comparison(comparison: pd.DataFrame, out_dir: Path, fit_order: int) -> Path:
    """Bar chart of K1 and K2 per configuration with error bars."""
    labels = comparison["label"].tolist()
    x = np.arange(len(labels))
    k1_mean = comparison["k1_mean"].to_numpy(dtype=float)
    k1_std = comparison["k1_std"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].bar(x, k1_mean, yerr=k1_std, capsize=4, color="steelblue", alpha=0.8)
    axes[0].axhline(1.0, color="gray", lw=0.8, ls="--", label="expected K1=1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel(f"K1 (order {fit_order})")
    axes[0].set_title("K1 mean ± std per subset")
    axes[0].legend()

    k2_col = "k2_mean"
    k2_std_col = "k2_std"
    if k2_col in comparison.columns:
        k2_mean = comparison[k2_col].to_numpy(dtype=float)
        k2_std = comparison[k2_std_col].to_numpy(dtype=float)
        axes[1].bar(x, k2_mean, yerr=k2_std, capsize=4, color="darkorange", alpha=0.8)
        axes[1].axhline(0.0, color="gray", lw=0.8, ls="--")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        axes[1].set_ylabel(f"K2 (order {fit_order})")
        axes[1].set_title("K2 mean ± std per subset")
    else:
        axes[1].set_visible(False)

    out_path = out_dir / f"sensitivity_k1_k2_order{fit_order}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    args = parse_args()
    out_dir = Path(ROOT) / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_order = args.spectral_fit_order
    rows = []

    for cfg in SENSITIVITY_CONFIGS:
        label = cfg["label"]
        run_out_dir = str(out_dir / label)
        os.makedirs(run_out_dir, exist_ok=True)

        fitting_csv = Path(run_out_dir) / f"spectral_fitting_order{fit_order}.csv"

        if fitting_csv.exists() and not args.force:
            print(f"[{label}] skipping — output exists ({fitting_csv})")
        else:
            cmd = config_to_cli(cfg, run_out_dir, fit_order)
            print(f"[{label}] running: {' '.join(cmd)}", flush=True)
            result = subprocess.run(cmd, cwd=ROOT, capture_output=False)
            if result.returncode != 0:
                print(f"[{label}] WARNING: runner exited with code {result.returncode}")

            # Runner writes spectral fitting to outputs/qc/; copy/move to run_out_dir
            default_csv = Path(ROOT) / "outputs" / "qc" / f"spectral_fitting_order{fit_order}.csv"
            if default_csv.exists() and not fitting_csv.exists():
                import shutil
                shutil.copy(default_csv, fitting_csv)

        if fitting_csv.exists():
            stats = load_fitting_stats(fitting_csv, fit_order)
            rows.append({"label": label, **stats})
            print(
                f"[{label}] {stats['n_success']}/{stats['n_pixels']} fit  "
                f"K1={stats.get('k1_mean', np.nan):.4f}±{stats.get('k1_std', np.nan):.2e}  "
                f"K2={stats.get('k2_mean', np.nan):.4f}±{stats.get('k2_std', np.nan):.2e}"
            )
        else:
            print(f"[{label}] no fitting output found — skipping stats")

    if not rows:
        print("No results collected; check runner output above.")
        return 1

    comparison = pd.DataFrame(rows)
    summary_csv = out_dir / f"sensitivity_summary_order{fit_order}.csv"
    comparison.to_csv(summary_csv, index=False)
    print(f"Wrote: {summary_csv}")

    plot_path = plot_comparison(comparison, out_dir, fit_order)
    print(f"Wrote: {plot_path}")

    # Markdown report
    md_lines = [
        f"# Sensitivity Study — Spectral Fitting (Order {fit_order})",
        "",
        "## K1/K2 Statistics per Spatial Subset",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Interpretation",
        "- K1 should be ~1.0 across all subsets if the O2-O2 optical depth is correctly scaled.",
        "- Large K1 spread across subsets indicates geometry or profile sensitivity.",
        "- K2 captures non-linear (e.g., scattering, cloud) effects and is expected to vary spatially.",
        "",
        f"![K1/K2 per subset]({plot_path.name})",
    ]
    md_path = out_dir / f"sensitivity_summary_order{fit_order}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
