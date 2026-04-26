"""WP6: CLDO4 comparison diagnostics and validation summary.

This module compares the modeled O2-O2 tau metrics against the CLDO4 fitted
slant-column field so the subset workflow produces a quick, human-readable QA
check before any broader production run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ValidationConfig:
    """Settings for the CLDO4 comparison step."""

    min_points_for_corr: int = 2


def build_validation_table(
    pixel_df: pd.DataFrame,
    tau_result: dict,
    reptran_result: dict,
) -> pd.DataFrame:
    """Assemble a per-pixel table of observed and modeled O2-O2 metrics."""
    if tau_result["pixel_index"].size == 0:
        return pd.DataFrame(
            columns=[
                "pixel_index",
                "mirror_step",
                "xtrack",
                "latitude",
                "longitude",
                "cldo4_fitted_slant_column",
                "tau_band_mean",
                "tau_reptran_mean",
                "sec_airmass",
            ]
        )

    subset = pixel_df.iloc[np.asarray(tau_result["pixel_index"], dtype=int)].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "pixel_index": tau_result["pixel_index"],
            "mirror_step": subset["mirror_step"].to_numpy(dtype=int),
            "xtrack": subset["xtrack"].to_numpy(dtype=int),
            "latitude": subset["latitude"].to_numpy(dtype=float),
            "longitude": subset["longitude"].to_numpy(dtype=float),
            "cldo4_fitted_slant_column": subset["cldo4_fitted_slant_column"].to_numpy(dtype=float),
            "tau_band_mean": np.asarray(tau_result["tau_band_mean"], dtype=float),
            "tau_reptran_mean": np.asarray(reptran_result["tau_reptran_mean"], dtype=float),
            "sec_airmass": np.asarray(tau_result["sec_airmass"], dtype=float),
        }
    )
    out["cldo4_over_tau_band"] = out["cldo4_fitted_slant_column"] / np.where(out["tau_band_mean"] != 0, out["tau_band_mean"], np.nan)
    out["cldo4_over_tau_reptran"] = out["cldo4_fitted_slant_column"] / np.where(out["tau_reptran_mean"] != 0, out["tau_reptran_mean"], np.nan)
    return out


def _corr(a: pd.Series, b: pd.Series) -> float:
    """Return a Pearson-style correlation coefficient for two finite series."""
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def write_validation_outputs(out_prefix: str, validation_table: pd.DataFrame) -> tuple[str, str]:
    """Write a CSV diagnostics table and a markdown validation summary."""
    out_prefix_path = Path(out_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = str(out_prefix_path.with_suffix(".csv"))
    md_path = str(out_prefix_path.with_suffix(".md"))

    validation_table.to_csv(csv_path, index=False)

    corr_band = _corr(validation_table["cldo4_fitted_slant_column"], validation_table["tau_band_mean"])
    corr_reptran = _corr(validation_table["cldo4_fitted_slant_column"], validation_table["tau_reptran_mean"])
    npts = int(len(validation_table))
    lines = [
        "# TEMPO O2-O2 validation summary",
        "",
        f"- Number of matched pixels: {npts}",
        f"- Correlation(CLDO4 fitted SCD, tau_band_mean): {corr_band:.6f}",
        f"- Correlation(CLDO4 fitted SCD, tau_reptran_mean): {corr_reptran:.6f}",
        "",
        "## Notes",
        "- These are quick-look diagnostics for the subset run.",
        "- Positive correlation is the first sanity check before adding plots and broader validation.",
    ]
    Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path
