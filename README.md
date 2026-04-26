# tempo

TEMPO O2-O2 and O2-B band optical depth retrieval pipelines (WP1–WP8).

See [`prompts/TEMPO_pipeline_plan.md`](prompts/TEMPO_pipeline_plan.md) for the full implementation plan, physics notes, and test tracking matrix.

## Pipelines

| Script | Band | λ range | Tau physics |
|---|---|---|---|
| `scripts/run_o2o2_pipeline_subset.py` | O2-O2 CIA | 460–490 nm | τ ∝ n_O2² (CIA); cross-sections from pre-measured .xs tables |
| `scripts/run_o2b_pipeline_subset.py` | O2 B-band | 683–697 nm | τ ∝ n_O2 (monomer); HITRAN line-by-line Voigt profiles |

Both scripts run WP1–WP7 end-to-end on any geographic bbox or scan/xtrack subset.

## Environment

```bash
conda activate er3t_env
```

## Run

### O2-O2 (460–490 nm)

```bash
python scripts/run_o2o2_pipeline_subset.py \
    --lat-min 31.6 --lat-max 33.6 \
    --lon-min -123.1 --lon-max -121.1 \
    --wl-min 460 --wl-max 490 --spectral-fit-order 7 \
    --rad-file   data/TEMPO/TEMPO_RAD_L1_V03_20240708T190926Z_S010G09.nc \
    --cldo4-file data/TEMPO/TEMPO_CLDO4_L2_V03_20240708T190926Z_S010G09.nc \
    --no2-file   data/TEMPO/TEMPO_NO2_L2_V03_20240708T190926Z_S010G09.nc \
    --irr-file   data/TEMPO/TEMPO_IRR_L1_V03_20240711T042711Z.nc \
    --geos-file  data/TEMPO/GEOS-CF.v01.rpl.sat_inst_1hr_r721x361_v72.20240708_1900z.nc4 \
    --tag S010G09_190926
```

### O2-B (683–697 nm)

```bash
python scripts/run_o2b_pipeline_subset.py \
    --lat-min 36.5 --lat-max 38.5 \
    --lon-min -98.6 --lon-max -96.6 \
    --wl-min 683 --wl-max 697 --spectral-fit-order 7 \
    --rad-file   data/TEMPO/TEMPO_RAD_L1_V03_20240708T184934Z_S010G06.nc \
    --cldo4-file data/TEMPO/TEMPO_CLDO4_L2_V03_20240708T184934Z_S010G06.nc \
    --no2-file   data/TEMPO/TEMPO_NO2_L2_V03_20240708T184934Z_S010G06.nc \
    --irr-file   data/TEMPO/TEMPO_IRR_L1_V03_20240711T042711Z.nc \
    --geos-file  data/TEMPO/GEOS-CF.v01.rpl.sat_inst_1hr_r721x361_v72.20240708_1900z.nc4 \
    --tag S010G06_160926_o2b_1
```

### Chunk merge (after full-granule runs)

```bash
python scripts/chunk_merge_driver.py \
    --native-chunks 'outputs/tau_native/tau_o2o2_native_scan_*_xt_*.nc'
```

## Outputs

| Path | Contents |
|---|---|
| `outputs/tau_native/` | Per-chunk O2-O2 / O2-B tau on native TEMPO grid (`.nc`) |
| `outputs/tau_reptran/` | REPTRAN-grid gas tau per species (`.nc`, `.csv`) |
| `outputs/wp1_wp2/` | Pixel table and GEOS-CF profiles (`.csv`, `.npz`) |
| `outputs/qc/<tag>/` | Validation summary, spectral fitting results, diagnostic plots |

## Source layout

```
src/
  wp1_pixel_table.py      pixel metadata table
  wp2_profiles.py         GEOS-CF collocation and profile conversion
  wp3_slit_kernel.py      IRR super-Gaussian slit kernels
  wp4_tau.py              O2-O2 CIA tau + Rayleigh
  wp4_o2b_tau.py          O2 B-band HITRAN line-by-line tau + H2O
  wp5_reptran.py          REPTRAN k-distribution gas tau (H2O, O2, N2, N2O, NO2, O3)
  wp6_validation.py       comparison with CLDO4 fitted slant column
  wp7_spectral_fitting.py cumulant expansion spectral fitting (five modes) + plots
  chunk_merge.py          gap detection, merge, validation across chunks
  constants.py            physical constants
  io_tempo.py             low-level TEMPO/GEOS-CF file readers
```

## Notes

- TEMPO products are read with `h5py` (avoids `netCDF4` IRR parsing issues).
- Pressure profile uses sigma-based reconstruction from `PS` (hybrid coefficients absent from this granule).
- Wavecal: `λ_corr(m,x,s) = nominal_wavelength(x,s) + wavecal_params(m,x,0)`; only `wavecal_opt_status==0` pixels used by default.
- O2 is excluded from the REPTRAN composite tau in the O2-B pipeline to avoid double-counting with the HITRAN WP4 tau.
