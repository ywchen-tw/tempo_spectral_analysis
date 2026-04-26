#!/usr/bin/env python
"""T11: Chunk merge driver for production tau workflow.

This script merges output from one or more per-chunk tau runs (WP4, WP5, WP6)
into full-granule results. Chunks are validated for ordering, and outputs are
written with merge timestamps for traceability.

Example usage:
    python scripts/chunk_merge_driver.py \\
        --native-chunks 'outputs/tau_native/tau_o2o2_native_scan_*_xt_*.nc' \\
        --reptran-chunks 'outputs/tau_reptran/tau_o2o2_reptran_scan_*_xt_*.nc' \\
        --validation-chunks 'outputs/qc/tau_validation_scan_*_xt_*.csv' \\
        --output-dir outputs/merged/

This will:
1. Validate all chunk sequences for ordering and continuity.
2. Merge native tau arrays (WP4).
3. Merge REPTRAN tau arrays (WP5).
4. Merge validation tables and recalculate correlations (WP6).
5. Write all outputs with ISO8601 timestamps.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from chunk_merge import (
    merge_reptran_chunks,
    merge_tau_chunks,
    merge_validation_chunks,
    validate_chunk_sequence,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for chunk merge configuration."""
    p = argparse.ArgumentParser(
        description="Merge per-chunk TEMPO O2-O2 tau outputs into full-granule results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all chunks in the default output directories
  python scripts/chunk_merge_driver.py \\
    --native-chunks "outputs/tau_native/tau_o2o2_native_scan_*_xt_*.nc" \\
    --output-dir outputs/merged/

  # Merge only REPTRAN chunks
  python scripts/chunk_merge_driver.py \\
    --reptran-chunks "outputs/tau_reptran/tau_o2o2_reptran_scan_*_xt_*.nc" \\
    --output-dir outputs/merged/

  # Merge all available chunks with full validation
  python scripts/chunk_merge_driver.py \\
    --native-chunks "outputs/tau_native/tau_o2o2_native_scan_*_xt_*.nc" \\
    --reptran-chunks "outputs/tau_reptran/tau_o2o2_reptran_scan_*_xt_*.nc" \\
    --validation-chunks "outputs/qc/tau_validation_scan_*_xt_*.csv" \\
    --output-dir outputs/merged/ \\
    --verbose
        """,
    )
    p.add_argument(
        "--native-chunks",
        type=str,
        default=None,
        help="Glob pattern for native tau chunks (WP4 outputs). "
        "Default: outputs/tau_native/tau_o2o2_native_scan_*_xt_*.nc",
    )
    p.add_argument(
        "--reptran-chunks",
        type=str,
        default=None,
        help="Glob pattern for REPTRAN tau chunks (WP5 outputs). "
        "Default: outputs/tau_reptran/tau_o2o2_reptran_scan_*_xt_*.nc",
    )
    p.add_argument(
        "--validation-chunks",
        type=str,
        default=None,
        help="Glob pattern for validation CSV chunks (WP6 outputs). "
        "Default: outputs/qc/tau_validation_scan_*_xt_*.csv",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where merged outputs will be written.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed validation warnings and progress.",
    )
    return p.parse_args()


def resolve_glob_pattern(pattern: str | None) -> list[str]:
    """Resolve a glob pattern to a sorted list of files.

    If pattern is None or matches no files, returns an empty list.
    """
    if pattern is None:
        return []
    files = sorted(glob.glob(pattern))
    return files


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"{title!s:^70}")
    print(f"{'=' * 70}\n")


def main() -> int:
    """Execute chunk merge workflow."""
    args = parse_args()

    # Set default patterns if not provided.
    native_pattern = args.native_chunks or "outputs/tau_native/tau_o2o2_native_scan_*_xt_*.nc"
    reptran_pattern = args.reptran_chunks or "outputs/tau_reptran/tau_o2o2_reptran_scan_*_xt_*.nc"
    validation_pattern = args.validation_chunks or "outputs/qc/tau_validation_scan_*_xt_*.csv"

    # Resolve glob patterns to file lists.
    native_files = resolve_glob_pattern(native_pattern)
    reptran_files = resolve_glob_pattern(reptran_pattern)
    validation_files = resolve_glob_pattern(validation_pattern)

    # Check if any chunks were found.
    total_chunks = len(native_files) + len(reptran_files) + len(validation_files)
    if total_chunks == 0:
        print("ERROR: No matching chunk files found.")
        print(f"  Native pattern: {native_pattern}")
        print(f"  REPTRAN pattern: {reptran_pattern}")
        print(f"  Validation pattern: {validation_pattern}")
        return 1

    print_section("TEMPO O2-O2 Chunk Merge Workflow")
    print(f"Output directory: {args.output_dir}\n")
    print(f"Chunks found:")
    print(f"  Native tau (WP4): {len(native_files)}")
    print(f"  REPTRAN tau (WP5): {len(reptran_files)}")
    print(f"  Validation (WP6): {len(validation_files)}")

    # Create output directory.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_warnings = {}

    # Merge native tau chunks if provided.
    if native_files:
        print_section("Merging Native Tau Chunks (WP4)")
        print(f"Processing {len(native_files)} native tau files...")

        if args.verbose:
            for i, fpath in enumerate(native_files, 1):
                print(f"  [{i}/{len(native_files)}] {Path(fpath).name}")

        try:
            result = merge_tau_chunks(native_files, args.output_dir)
            print(f"\n✓ Merged native tau written to:")
            print(f"  {result['merged_nc']}")
            print(f"  Total pixels: {result['pixel_count']}")

            if result["validation_warnings"]:
                all_warnings["native_tau"] = result["validation_warnings"]
                if args.verbose:
                    print(f"\nValidation warnings ({len(result['validation_warnings'])}):")
                    for warn in result["validation_warnings"]:
                        print(f"  ⚠ {warn}")
        except Exception as e:
            print(f"✗ Error merging native tau: {e}")
            return 1

    # Merge REPTRAN chunks if provided.
    if reptran_files:
        print_section("Merging REPTRAN Tau Chunks (WP5)")
        print(f"Processing {len(reptran_files)} REPTRAN tau files...")

        if args.verbose:
            for i, fpath in enumerate(reptran_files, 1):
                print(f"  [{i}/{len(reptran_files)}] {Path(fpath).name}")

        try:
            result = merge_reptran_chunks(reptran_files, args.output_dir)
            print(f"\n✓ Merged REPTRAN tau written to:")
            print(f"  {result['merged_nc']}")
            print(f"  {result['merged_csv']}")
            print(f"  Total pixels: {result['pixel_count']}")
            print(f"  Total bands: {result['band_count']}")

            if result["validation_warnings"]:
                all_warnings["reptran_tau"] = result["validation_warnings"]
                if args.verbose:
                    print(f"\nValidation warnings ({len(result['validation_warnings'])}):")
                    for warn in result["validation_warnings"]:
                        print(f"  ⚠ {warn}")
        except Exception as e:
            print(f"✗ Error merging REPTRAN tau: {e}")
            return 1

    # Merge validation chunks if provided.
    if validation_files:
        print_section("Merging Validation Outputs (WP6)")
        print(f"Processing {len(validation_files)} validation CSV files...")

        if args.verbose:
            for i, fpath in enumerate(validation_files, 1):
                print(f"  [{i}/{len(validation_files)}] {Path(fpath).name}")

        try:
            result = merge_validation_chunks(validation_files, args.output_dir)
            print(f"\n✓ Merged validation outputs written to:")
            print(f"  {result['merged_csv']}")
            print(f"  {result['merged_md']}")
            print(f"  Total pixels: {result['pixel_count']}")
            print(f"  Correlation CLDO4 vs tau_band_mean: {result['correlation_band_mean']:.6f}")
            print(f"  Correlation CLDO4 vs tau_reptran_mean: {result['correlation_reptran_mean']:.6f}")

            if result["validation_warnings"]:
                all_warnings["validation"] = result["validation_warnings"]
                if args.verbose:
                    print(f"\nValidation warnings ({len(result['validation_warnings'])}):")
                    for warn in result["validation_warnings"]:
                        print(f"  ⚠ {warn}")
        except Exception as e:
            print(f"✗ Error merging validation outputs: {e}")
            return 1

    # Print summary of all warnings.
    print_section("Merge Complete")
    if all_warnings:
        total_warn_count = sum(len(v) for v in all_warnings.values())
        print(f"⚠  {total_warn_count} total warnings across all merge operations:")
        for source, warn_list in all_warnings.items():
            print(f"\n  {source}:")
            for warn in warn_list:
                print(f"    - {warn}")
    else:
        print("✓ No validation issues detected.")

    print(f"\nAll outputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
