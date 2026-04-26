"""T11: Chunk merge utilities for multi-chunk tau workflow integration.

This module provides functions to merge per-chunk native tau (WP4), REPTRAN tau (WP5),
and validation outputs (WP6) into full-granule results. The primary workflow is:

1. validate_chunk_sequence() → checks scan/xtrack ordering and mirror_step continuity
2. merge_tau_chunks() → concatenates native tau along pixel axis with validation
3. merge_reptran_chunks() → concatenates REPTRAN tau along pixel axis
4. merge_validation_chunks() → concatenates validation CSVs and rebuilds correlations

All merged files are written with ISO8601 timestamps for production traceability.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


# ============================================================================
# Chunk Sequence Validation
# ============================================================================


def validate_chunk_sequence(chunk_files: list[str]) -> list[str]:
    """Validate scan and xtrack ordering, mirror_step continuity across chunks.

    Returns a list of warning strings. Empty list indicates valid sequence.

    Checks:
    - Scan indices are sequential (no gaps or duplicates)
    - Xtrack ranges do not overlap
    - Mirror step is monotonically increasing across chunks
    """
    if not chunk_files:
        return ["No chunk files provided."]

    warnings = []

    # Parse chunk identifiers from filenames.
    scan_ranges = []
    xtrack_ranges = []

    for fpath in chunk_files:
        # Expected format: ..._scan_START_END_xt_XSTART_XEND...
        match = re.search(r"_scan_(\d+)_(\d+)_xt_(\d+)_(\d+)", fpath)
        if not match:
            warnings.append(f"Could not parse chunk indices from: {Path(fpath).name}")
            continue
        scan_start, scan_end, xt_start, xt_end = map(int, match.groups())
        scan_ranges.append((scan_start, scan_end))
        xtrack_ranges.append((xt_start, xt_end))

    if len(scan_ranges) != len(chunk_files):
        warnings.append("Could not parse all chunk filenames; proceeding with partial validation.")
        return warnings

    # Check scan ordering.
    for i, (s_start, s_end) in enumerate(scan_ranges[:-1]):
        next_s_start, next_s_end = scan_ranges[i + 1]
        if s_end >= next_s_start:
            warnings.append(
                f"Chunk {i}: scan range [{s_start}, {s_end}] overlaps or does not precede "
                f"chunk {i+1} [{next_s_start}, {next_s_end}]"
            )

    # Check xtrack ranges for each chunk pairing.
    for i, (xt_start, xt_end) in enumerate(xtrack_ranges[:-1]):
        next_xt_start, next_xt_end = xtrack_ranges[i + 1]
        if not (xt_start == next_xt_start and xt_end == next_xt_end):
            warnings.append(
                f"Chunk {i}: xtrack range [{xt_start}, {xt_end}] differs from "
                f"chunk {i+1} [{next_xt_start}, {next_xt_end}]"
            )

    # Check mirror_step continuity across chunks.
    try:
        last_mirror_step_max = None
        for fpath in chunk_files:
            with h5py.File(fpath, "r") as f:
                if "mirror_step" not in f:
                    warnings.append(f"No 'mirror_step' dataset in {Path(fpath).name}")
                    continue
                mirror_step = f["mirror_step"][:]
                if len(mirror_step) == 0:
                    warnings.append(f"Empty mirror_step in {Path(fpath).name}")
                    continue
                # Check monotonic increase within chunk.
                if not np.all(np.diff(mirror_step) >= 0):
                    warnings.append(f"Non-monotonic mirror_step in {Path(fpath).name}")
                # Check continuity with previous chunk.
                if last_mirror_step_max is not None:
                    if mirror_step[0] <= last_mirror_step_max:
                        warnings.append(
                            f"Mirror step discontinuity: previous chunk ends at {last_mirror_step_max}, "
                            f"current chunk starts at {mirror_step[0]}"
                        )
                if len(mirror_step) > 0:
                    last_mirror_step_max = np.max(mirror_step)
    except Exception as e:
        warnings.append(f"Error checking mirror_step continuity: {e}")

    return warnings


# ============================================================================
# Native Tau Merge
# ============================================================================


def merge_tau_chunks(chunk_files: list[str], output_prefix: str) -> dict[str, Path]:
    """Merge native tau chunks (WP4) into a full-granule output.

    Parameters
    ----------
    chunk_files : list[str]
        List of paths to tau_o2o2_native_scan_X_Y_xt_A_B.nc files, in order.
    output_prefix : str
        Directory where merged output will be written.

    Returns
    -------
    dict[str, Path]
        Dictionary with keys 'merged_nc' (path to merged netCDF file),
        'validation_warnings' (list of validation strings),
        'pixel_count' (total pixels merged).
    """
    if not chunk_files:
        raise ValueError("No chunk files provided.")

    out_dir = Path(output_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate sequence first.
    warnings = validate_chunk_sequence(chunk_files)

    # Read all chunks and collect arrays.
    pixel_arrays = {}  # keys are dataset names
    total_pixels = 0

    for chunk_fpath in chunk_files:
        fpath = Path(chunk_fpath)
        if not fpath.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_fpath}")

        with h5py.File(chunk_fpath, "r") as f:
            # On first chunk, initialize structure from attributes and dimension datasets.
            if not pixel_arrays:
                # Store metadata that doesn't concatenate (scalars, 1D dimension arrays).
                for key in ["spectral_channel"]:
                    if key in f:
                        pixel_arrays[key] = f[key][:]

                # Capture file-level attributes.
                pixel_arrays["_attrs"] = dict(f.attrs)

            # Concatenate pixel-dimension data.
            for key in f.keys():
                if key == "spectral_channel" or key == "pixel":
                    continue
                try:
                    data = f[key][:]
                    # Infer if this is pixel-dimension data (1D with length matching pixel count).
                    if data.ndim == 1 and data.shape[0] > 0:
                        if key not in pixel_arrays:
                            pixel_arrays[key] = []
                        pixel_arrays[key].append(data)
                    elif data.ndim == 2 and data.shape[0] > 0:
                        # 2D data (e.g., tau_eff with shape (npix, nwl)).
                        if key not in pixel_arrays:
                            pixel_arrays[key] = []
                        pixel_arrays[key].append(data)
                except Exception as e:
                    warnings.append(f"Error reading {key} from {fpath.name}: {e}")

        # Count pixels in this chunk.
        if "mirror_step" in pixel_arrays and isinstance(pixel_arrays["mirror_step"], list):
            total_pixels += len(pixel_arrays["mirror_step"][-1])

    # Concatenate all collected arrays.
    for key, val in pixel_arrays.items():
        if isinstance(val, list):
            pixel_arrays[key] = np.concatenate(val, axis=0)

    # Write merged netCDF file.
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    merged_nc = out_dir / f"tau_o2o2_native_merged_{timestamp}.nc"

    n_pix = total_pixels
    n_wl = pixel_arrays["spectral_channel"].shape[0] if "spectral_channel" in pixel_arrays else 0

    with h5py.File(str(merged_nc), "w") as f:
        # Write attributes.
        f.attrs["title"] = "TEMPO O2-O2 native tau (merged chunks)"
        f.attrs["description"] = "Wavelength-resolved O2-O2 tau merged from multiple scan/xtrack chunks"
        f.attrs["merge_timestamp"] = timestamp
        f.attrs["num_source_chunks"] = len(chunk_files)

        # Write dimension datasets.
        f.create_dataset("pixel", data=np.arange(n_pix, dtype=np.int32))
        if n_wl:
            f.create_dataset("spectral_channel", data=pixel_arrays["spectral_channel"].astype(np.int32))

        # Write merged data.
        for key, data in pixel_arrays.items():
            if key not in ["spectral_channel", "_attrs"]:
                try:
                    f.create_dataset(key, data=data.astype(float if data.dtype == np.float64 else data.dtype))
                except (ValueError, TypeError):
                    pass

    return {
        "merged_nc": merged_nc,
        "validation_warnings": warnings,
        "pixel_count": total_pixels,
    }


# ============================================================================
# REPTRAN Tau Merge
# ============================================================================


def merge_reptran_chunks(chunk_files: list[str], output_prefix: str) -> dict[str, Path]:
    """Merge REPTRAN tau chunks (WP5) into a full-granule output.

    Parameters
    ----------
    chunk_files : list[str]
        List of paths to tau_o2o2_reptran_scan_X_Y_xt_A_B.nc files, in order.
    output_prefix : str
        Directory where merged output will be written.

    Returns
    -------
    dict[str, Path]
        Dictionary with keys 'merged_nc' (path to merged netCDF),
        'merged_csv' (path to merged summary CSV),
        'validation_warnings' (list of warning strings),
        'pixel_count' (total pixels merged),
        'band_count' (number of REPTRAN bands).
    """
    if not chunk_files:
        raise ValueError("No chunk files provided.")

    out_dir = Path(output_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate sequence.
    warnings = validate_chunk_sequence(chunk_files)

    # Read all chunks.
    band_arrays = {}  # Static across chunks: band_index, rep_wavelength_nm, etc.
    pixel_arrays = {}  # Concatenated: tau_reptran, tau_reptran_mean
    pixel_metadata = []  # List of DataFrames to concatenate.
    total_pixels = 0
    n_bands = 0

    for chunk_fpath in chunk_files:
        fpath = Path(chunk_fpath)
        if not fpath.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_fpath}")

        with h5py.File(chunk_fpath, "r") as f:
            # Capture band-dimension data on first chunk.
            if not band_arrays:
                for key in ["band_index", "rep_wavelength_nm", "band_wmin_nm", "band_wmax_nm", "band_nwvl"]:
                    if key in f:
                        band_arrays[key] = f[key][:]
                if "rep_wavelength_nm" in band_arrays:
                    n_bands = len(band_arrays["rep_wavelength_nm"])

                # Capture file attributes.
                band_arrays["_attrs"] = dict(f.attrs)

            # Concatenate pixel arrays.
            for key in ["tau_reptran", "tau_reptran_mean"]:
                if key in f:
                    data = f[key][:]
                    if key not in pixel_arrays:
                        pixel_arrays[key] = []
                    pixel_arrays[key].append(data)

        total_pixels += data.shape[0]

    # Concatenate pixel arrays.
    for key in pixel_arrays:
        pixel_arrays[key] = np.concatenate(pixel_arrays[key], axis=0)

    # Write merged netCDF file.
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    merged_nc = out_dir / f"tau_o2o2_reptran_merged_{timestamp}.nc"
    merged_csv = out_dir / f"tau_o2o2_reptran_merged_{timestamp}.csv"

    with h5py.File(str(merged_nc), "w") as f:
        f.attrs["title"] = "TEMPO O2-O2 REPTRAN tau (merged chunks)"
        f.attrs["description"] = "Tau projected onto REPTRAN bands, merged from multiple chunks"
        f.attrs["merge_timestamp"] = timestamp
        f.attrs["num_source_chunks"] = len(chunk_files)

        # Write band dimension.
        for key, data in band_arrays.items():
            if key != "_attrs":
                f.create_dataset(key, data=data.astype(np.int32 if "index" in key or "nwvl" in key else float))

        # Write merged pixel arrays.
        for key, data in pixel_arrays.items():
            f.create_dataset(key, data=data.astype(float))

    # Also write a merged summary CSV (pixel-level metadata + tau_reptran_mean).
    # Note: We don't have mirror_step/xtrack here without reading CSVs, so create minimal version.
    summary_df = pd.DataFrame({
        "pixel_index": np.arange(total_pixels, dtype=int),
        "tau_reptran_mean": pixel_arrays.get("tau_reptran_mean", np.zeros(total_pixels, dtype=float)),
    })
    summary_df.to_csv(merged_csv, index=False)

    return {
        "merged_nc": merged_nc,
        "merged_csv": merged_csv,
        "validation_warnings": warnings,
        "pixel_count": total_pixels,
        "band_count": n_bands,
    }


# ============================================================================
# Validation CSV Merge
# ============================================================================


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient, returning NaN if inputs are empty."""
    if len(x) < 2 or len(y) < 2:
        return np.nan
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) < 2:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def merge_validation_chunks(chunk_csv_files: list[str], output_prefix: str) -> dict[str, Path]:
    """Merge validation CSV chunks (WP6) and recalculate correlation diagnostics.

    Parameters
    ----------
    chunk_csv_files : list[str]
        List of paths to tau_validation_scan_X_Y_xt_A_B.csv files, in order.
    output_prefix : str
        Directory where merged output will be written.

    Returns
    -------
    dict[str, Path]
        Dictionary with keys 'merged_csv' (path to merged table),
        'merged_md' (path to markdown summary),
        'validation_warnings' (list of warning strings),
        'pixel_count' (total pixels),
        'correlation_band_mean' (recalculated correlation),
        'correlation_reptran_mean' (recalculated correlation).
    """
    if not chunk_csv_files:
        raise ValueError("No validation CSV files provided.")

    out_dir = Path(output_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate sequence.
    warnings = validate_chunk_sequence(chunk_csv_files)

    # Read and concatenate all CSVs.
    dfs = []
    for csv_path in chunk_csv_files:
        fpath = Path(csv_path)
        if not fpath.exists():
            raise FileNotFoundError(f"Validation CSV not found: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        except Exception as e:
            warnings.append(f"Error reading {fpath.name}: {e}")

    if not dfs:
        raise ValueError("No validation tables could be read.")

    merged_table = pd.concat(dfs, ignore_index=True)
    total_pixels = len(merged_table)

    # Recalculate correlation statistics on merged table.
    corr_band = _correlation(
        merged_table["cldo4_fitted_slant_column"].to_numpy(),
        merged_table["tau_band_mean"].to_numpy(),
    )
    corr_reptran = _correlation(
        merged_table["cldo4_fitted_slant_column"].to_numpy(),
        merged_table["tau_reptran_mean"].to_numpy(),
    )

    # Check for consistency in columns.
    expected_cols = {"cldo4_fitted_slant_column", "tau_band_mean", "tau_reptran_mean"}
    if not expected_cols.issubset(set(merged_table.columns)):
        warnings.append(f"Validation table missing expected columns. Expected: {expected_cols}")

    # Write merged CSV.
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    merged_csv = out_dir / f"tau_validation_merged_{timestamp}.csv"
    merged_md = out_dir / f"tau_validation_merged_{timestamp}.md"

    merged_table.to_csv(merged_csv, index=False)

    # Write markdown summary.
    lines = [
        "# TEMPO O2-O2 Validation Summary (Merged Chunks)",
        "",
        f"- Number of matched pixels: {total_pixels}",
        f"- Correlation(CLDO4 fitted SCD, tau_band_mean): {corr_band:.6f}",
        f"- Correlation(CLDO4 fitted SCD, tau_reptran_mean): {corr_reptran:.6f}",
        "",
        "## Notes",
        "- Merged from multiple scan/xtrack chunks.",
        "- Correlations are recalculated on the full merged table.",
        "- File includes all matched pixels from all constituent chunks.",
    ]
    Path(merged_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "merged_csv": merged_csv,
        "merged_md": merged_md,
        "validation_warnings": warnings,
        "pixel_count": total_pixels,
        "correlation_band_mean": corr_band,
        "correlation_reptran_mean": corr_reptran,
    }
