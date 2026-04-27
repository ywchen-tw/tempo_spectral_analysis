"""goes_abi_rgb.py — Download and render GOES ABI true-colour RGB images.

Downloads ABI reflectance bands (C01, C02, C03) from NOAA's open S3 archive,
finds the closest available scan to a target UTC time, reprojects from the
GOES fixed-grid to geographic lat/lon, and composites a true-colour PNG.

Satellite selection
-------------------
'auto' picks GOES-East (G16, 75°W) for lon > -105° and GOES-West (G18, 137°W)
for lon < -105°.  Pass which='east' or which='west' to override.

True-colour synthesis (CIRA recipe)
------------------------------------
    R = C02  (0.64 µm red)
    G = 0.45 × C03 + 0.10 × C02 + 0.45 × C01   (synthesised green)
    B = C01  (0.47 µm blue)
Gamma correction (default 2.2) is applied before compositing.

S3 data source
--------------
    s3://noaa-goes16   (GOES-East, anonymous public access)
    s3://noaa-goes18   (GOES-West, anonymous public access)
Products:
    ABI-L2-CMIPC  — CONUS, ~5 min cadence
    ABI-L2-CMIPF  — Full Disk, ~15 min cadence

Usage (standalone)
------------------
    python src/goes_abi_rgb.py \\
        --utc-time 2024-07-08T16:09:26 \\
        --lon-min -119.7 --lon-max -117.2 \\
        --lat-min 18.6   --lat-max 20.1 \\
        --fdir outputs/ \\
        --coastline

Dependencies
------------
    s3fs, pyproj, scipy, netCDF4, numpy, matplotlib
    cartopy  (optional, for --coastline)
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import netCDF4 as nc4
import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

_GOES_BOUNDARY_LON = -105.0   # centre-lon threshold for East vs West selection

_SAT_CFG = {
    'east': (16, 'noaa-goes16'),
    'west': (18, 'noaa-goes18'),
}

# ── Internal helpers ───────────────────────────────────────────────────────────

def _pick_satellite(extent: list[float]) -> tuple[str, int, str]:
    """Return (label, goes_number, s3_bucket) for the given lon/lat extent.

    GOES-East (G16) subpoint: 75°W  — best for centre_lon > -105°.
    GOES-West (G18) subpoint: 137°W — best for centre_lon < -105°.
    """
    center_lon = (extent[0] + extent[1]) / 2.0
    label = 'east' if center_lon >= _GOES_BOUNDARY_LON else 'west'
    sat_num, bucket = _SAT_CFG[label]
    return label, sat_num, bucket


def _parse_start_time(filename: str) -> pd.Timestamp:
    """Parse scan start time from an ABI filename.

    Pattern: …_s{YYYYDDDHHMMSS[tenth]}_… → UTC-aware Timestamp.
    """
    m = re.search(r'_s(\d{14})_', filename)
    if not m:
        raise ValueError(f'Cannot parse start time from {filename!r}')
    ts   = m.group(1)
    year = int(ts[0:4])
    doy  = int(ts[4:7])
    hh   = int(ts[7:9])
    mm   = int(ts[9:11])
    ss   = int(ts[11:13])
    return (pd.Timestamp(f'{year}-01-01', tz='UTC')
            + pd.Timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss))


def _find_closest_scan(
        fs,
        utc_time: pd.Timestamp,
        bucket: str,
        product: str,
        search_window_hours: int = 2,
) -> tuple[pd.Timestamp, str]:
    """Return (scan_time, start_token) for the scan nearest to utc_time.

    Enumerates C02 files in S3 within ±search_window_hours to identify
    candidate scan start times, then picks the closest.  All three bands of
    the winning scan share the same start_token in their filenames.
    """
    ref = utc_time

    candidates: list[tuple[float, pd.Timestamp, str]] = []
    for dh in range(-search_window_hours, search_window_hours + 1):
        t_search = ref + pd.Timedelta(hours=dh)
        prefix   = (f'{bucket}/{product}/'
                    f'{t_search.year:04d}/{t_search.day_of_year:03d}/'
                    f'{t_search.hour:02d}/')
        try:
            files = fs.ls(prefix)
        except FileNotFoundError:
            continue
        for f in files:
            if 'C02_' not in f or not f.endswith('.nc'):
                continue
            fname = f.split('/')[-1]
            try:
                scan_t = _parse_start_time(fname)
            except ValueError:
                continue
            candidates.append((abs((scan_t - ref).total_seconds()), scan_t, fname))

    if not candidates:
        raise RuntimeError(
            f'No {product} scans found within ±{search_window_hours}h of {ref} '
            f'in {bucket}'
        )

    candidates.sort()
    _, scan_time, best_fname = candidates[0]
    m = re.search(r'_s(\d{14})_', best_fname)
    return scan_time, m.group(1)


def _load_band(
        fs,
        bucket: str,
        product: str,
        start_token: str,
        channel: str,
        sat_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download one ABI reflectance band and return (cmi, lon_2d, lat_2d).

    Converts GOES fixed-grid scan angles to geographic lat/lon via pyproj.
    Out-of-disk pixels (projected to ±1e30) are set to NaN.
    """
    import pyproj

    year = int(start_token[0:4])
    doy  = int(start_token[4:7])
    hh   = int(start_token[7:9])
    prefix = (f'{bucket}/{product}/'
              f'{year:04d}/{doy:03d}/{hh:02d}/')

    all_files = fs.ls(prefix)
    matches   = [f for f in all_files
                 if f'{channel}_G{sat_num}_s{start_token}' in f
                 and f.endswith('.nc')]
    if not matches:
        raise FileNotFoundError(
            f'No {channel} file with start_token={start_token} in {prefix}'
        )

    with fs.open(matches[0], 'rb') as fh:
        raw = fh.read()

    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        with nc4.Dataset(tmp_path) as ds:
            gip        = ds.variables['goes_imager_projection']
            sat_lon    = float(gip.longitude_of_projection_origin)
            sat_height = float(gip.perspective_point_height)
            x_rad      = np.array(ds.variables['x'][:],  dtype=np.float64)
            y_rad      = np.array(ds.variables['y'][:],  dtype=np.float64)
            cmi_var    = ds.variables['CMI']
            cmi        = np.array(cmi_var[:],             dtype=np.float32)
            fill_val   = float(cmi_var._FillValue)
    finally:
        os.unlink(tmp_path)

    cmi[cmi == fill_val] = np.nan

    proj = pyproj.Proj(proj='geos', lon_0=sat_lon, h=sat_height,
                       x_0=0, y_0=0, sweep='x', ellps='GRS80')

    xx, yy         = np.meshgrid(x_rad * sat_height, y_rad * sat_height)
    lon_2d, lat_2d = proj(xx, yy, inverse=True)
    lon_2d = lon_2d.astype(np.float32)
    lat_2d = lat_2d.astype(np.float32)
    lon_2d[np.abs(lon_2d) > 1e10] = np.nan
    lat_2d[np.abs(lat_2d) > 1e10] = np.nan

    return cmi, lon_2d, lat_2d


def _resample(
        cmi: np.ndarray,
        lon_2d: np.ndarray,
        lat_2d: np.ndarray,
        extent: list[float],
        resolution: float,
) -> np.ndarray:
    """Nearest-neighbour resample from GOES swath to a regular lat/lon grid.

    Returns a 2-D array on a (lat_out, lon_out) grid oriented north→south so
    imshow renders without flipping.
    """
    from scipy.spatial import cKDTree

    lon_min, lon_max, lat_min, lat_max = extent
    valid = (np.isfinite(lon_2d) & np.isfinite(lat_2d) & np.isfinite(cmi))
    if not valid.any():
        raise RuntimeError(
            'No valid pixels after NaN masking — '
            'check that the extent is within satellite coverage'
        )

    lon_out = np.arange(lon_min, lon_max + resolution, resolution)
    lat_out = np.arange(lat_max, lat_min - resolution, -resolution)   # N → S
    lon_mg, lat_mg = np.meshgrid(lon_out, lat_out)

    pts    = np.column_stack([lon_2d[valid], lat_2d[valid]])
    tree   = cKDTree(pts)
    _, idx = tree.query(
        np.column_stack([lon_mg.ravel(), lat_mg.ravel()]),
        workers=-1,
    )
    return cmi[valid][idx].reshape(lon_mg.shape)


# ── Public API ─────────────────────────────────────────────────────────────────

def download_goes_abi_rgb(
        utc_time,
        extent: list[float],
        which: str = 'auto',
        product: str = 'ABI-L2-CMIPC',
        resolution: float = 0.005,
        gamma: float = 2.2,
        fdir: str = '.',
        coastline: bool = False,
        run: bool = True,
) -> tuple[str, str, pd.Timestamp | None]:
    """Download and render a GOES ABI true-colour RGB from NOAA's S3 archive.

    Parameters
    ----------
    utc_time   : datetime-like or pd.Timestamp
        Target observation time (UTC).  Timezone-naive inputs are assumed UTC.
    extent     : [lon_min, lon_max, lat_min, lat_max]
    which      : 'auto' | 'east' | 'west'
        'auto' picks GOES-East for centre_lon > -105°, else GOES-West.
    product    : 'ABI-L2-CMIPC' (CONUS, ~5 min) or 'ABI-L2-CMIPF' (Full Disk).
    resolution : output grid spacing in degrees (default 0.005° ≈ 500 m).
    gamma      : gamma-correction exponent (default 2.2).
    fdir       : directory for the output PNG (created if absent).
    coastline  : overlay 10-m coastlines (requires cartopy).
    run        : if False, skip download/render and return the expected filename.

    Returns
    -------
    (png_path, satellite_label, actual_scan_time)
        satellite_label : 'east' or 'west'
        actual_scan_time : UTC-aware pd.Timestamp of the matched scan
                           (None when run=False)
    """
    import matplotlib.pyplot as plt

    t = pd.Timestamp(utc_time)
    if t.tzinfo is None:
        t = t.tz_localize('UTC')
    else:
        t = t.tz_convert('UTC')

    if which == 'auto':
        sat_label, sat_num, bucket = _pick_satellite(extent)
    elif which in ('east', 'west'):
        sat_label = which
        sat_num, bucket = _SAT_CFG[which]
    else:
        raise ValueError(f"which must be 'auto', 'east', or 'west'; got {which!r}")

    extent_str = '-'.join(f'{e:.2f}' for e in extent)
    time_str   = t.strftime('%Y%m%dT%H%M%SZ')
    fname      = str(Path(fdir) / f'goes_{sat_label}_{time_str}_{extent_str}.png')

    if not run:
        return fname, sat_label, None

    try:
        import s3fs as _s3fs
    except ImportError:
        raise ImportError(
            "download_goes_abi_rgb requires 's3fs'. Install with: pip install s3fs"
        )

    print(f'  GOES-{sat_label.upper()} ({bucket}): '
          f'finding closest scan to {t} …', flush=True)

    fs = _s3fs.S3FileSystem(anon=True)
    scan_time, start_token = _find_closest_scan(fs, t, bucket, product)
    dt_s = abs((scan_time - t).total_seconds())
    print(f'  Closest scan: {scan_time}  (Δt = {dt_s:.0f} s)', flush=True)

    # Pre-clip download region to reduce memory; pad slightly beyond target extent
    pad  = 0.5
    clip = [extent[0] - pad, extent[1] + pad,
            extent[2] - pad, extent[3] + pad]

    bands: dict[str, tuple] = {}
    for ch in ('C01', 'C02', 'C03'):
        print(f'  Loading {ch} …', flush=True)
        cmi, lon_2d, lat_2d = _load_band(fs, bucket, product,
                                          start_token, ch, sat_num)
        # Spatial pre-clip before building cKDTree
        mask = (np.isfinite(lon_2d) & np.isfinite(lat_2d) &
                (lon_2d >= clip[0]) & (lon_2d <= clip[1]) &
                (lat_2d >= clip[2]) & (lat_2d <= clip[3]))
        if not mask.any():
            raise RuntimeError(
                f'No {ch} pixels within extent {extent}.  '
                f'Try which="west" or product="ABI-L2-CMIPF".'
            )
        rows  = np.where(mask.any(axis=1))[0]
        cols  = np.where(mask.any(axis=0))[0]
        sl_r  = slice(rows[0], rows[-1] + 1)
        sl_c  = slice(cols[0], cols[-1] + 1)
        bands[ch] = (cmi[sl_r, sl_c], lon_2d[sl_r, sl_c], lat_2d[sl_r, sl_c])

    print(f'  Resampling to {resolution}° grid …', flush=True)
    c01 = _resample(*bands['C01'], extent, resolution)
    c02 = _resample(*bands['C02'], extent, resolution)
    c03 = _resample(*bands['C03'], extent, resolution)

    # CIRA true-colour synthesis
    R = np.clip(c02, 0.0, 1.0)
    G = np.clip(0.45 * c03 + 0.10 * c02 + 0.45 * c01, 0.0, 1.0)
    B = np.clip(c01, 0.0, 1.0)
    R = R ** (1.0 / gamma)
    G = G ** (1.0 / gamma)
    B = B ** (1.0 / gamma)
    rgb = np.stack([R, G, B], axis=-1)

    os.makedirs(fdir, exist_ok=True)

    imshow_extent = [extent[0], extent[1], extent[2], extent[3]]

    if coastline:
        try:
            import cartopy.crs as ccrs
            fig = plt.figure(figsize=(10, 8))
            ax  = fig.add_subplot(111, projection=ccrs.PlateCarree())
            ax.imshow(rgb, extent=imshow_extent,
                      transform=ccrs.PlateCarree(),
                      aspect='auto', origin='upper')
            ax.coastlines(resolution='10m', color='black', linewidth=0.5, alpha=0.8)
            ax.set_extent(imshow_extent, crs=ccrs.PlateCarree())
            ax.axis('off')
        except ImportError:
            print('  Warning: cartopy not found — coastlines skipped.', flush=True)
            coastline = False

    if not coastline:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(rgb, extent=imshow_extent, aspect='auto', origin='upper')
        ax.axis('off')

    plt.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    print(f'  GOES ABI RGB saved → {fname}', flush=True)

    return fname, sat_label, scan_time


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description='Download a GOES ABI true-colour RGB for a given time and extent.'
    )
    parser.add_argument('--utc-time',  required=True,
                        help='Target UTC time, e.g. 2024-07-08T16:09:26')
    parser.add_argument('--lon-min',   required=True, type=float)
    parser.add_argument('--lon-max',   required=True, type=float)
    parser.add_argument('--lat-min',   required=True, type=float)
    parser.add_argument('--lat-max',   required=True, type=float)
    parser.add_argument('--which',     default='auto',
                        choices=['auto', 'east', 'west'])
    parser.add_argument('--product',   default='ABI-L2-CMIPC',
                        choices=['ABI-L2-CMIPC', 'ABI-L2-CMIPF'])
    parser.add_argument('--resolution', default=0.005, type=float,
                        help='Output grid spacing in degrees (default 0.005)')
    parser.add_argument('--gamma',     default=2.2, type=float)
    parser.add_argument('--fdir',      default='outputs/')
    parser.add_argument('--coastline', action='store_true')
    args = parser.parse_args()

    extent = [args.lon_min, args.lon_max, args.lat_min, args.lat_max]
    png, sat, scan_t = download_goes_abi_rgb(
        utc_time   = args.utc_time,
        extent     = extent,
        which      = args.which,
        product    = args.product,
        resolution = args.resolution,
        gamma      = args.gamma,
        fdir       = args.fdir,
        coastline  = args.coastline,
    )
    print(f'\nDone.  Satellite: GOES-{sat.upper()}  Scan time: {scan_t}')
    print(f'Output: {png}')


if __name__ == '__main__':
    _cli()
