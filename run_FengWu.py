import os
import re
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import xarray as xr

from earth2studio.models.px import FengWu
from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.run import deterministic


YEAR = 2020
N_STEPS = 40  # Earth2Studio will output 0h + 6h ... 240h. We remove 0h below.

WB2_LEVELS = np.array(
    [1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
     100, 125, 150, 175, 200, 225, 250, 300,
     350, 400, 450, 500, 550, 600, 650, 700,
     750, 775, 800, 825, 850, 875, 900, 925,
     950, 975, 1000],
    dtype=np.int64,
)

VAR_PREFIX_MAP = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "r": "relative_humidity",
    "w": "vertical_velocity",
}

SURFACE_VAR_MAP = {
    "t2m": "2m_temperature",
    "u10m": "10m_u_component_of_wind",
    "v10m": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "sp": "surface_pressure",
    "tcwv": "total_column_water_vapour",
}


def init_times_for_year(year: int):
    times = []
    t = datetime(year, 1, 1, 0)
    while t < datetime(year + 1, 1, 1, 0):
        times.append(t)
        t += timedelta(days=1)
    return times


def _standardize_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds
    if np.issubdtype(ds["time"].dtype, np.datetime64):
        return ds
    return ds.assign_coords(time=pd.to_datetime(ds["time"].values).values)


def _standardize_prediction_timedelta(ds: xr.Dataset) -> xr.Dataset:
    if "prediction_timedelta" not in ds.coords:
        return ds

    ptd = ds["prediction_timedelta"]

    if np.issubdtype(ptd.dtype, np.timedelta64):
        hours = (ptd.values / np.timedelta64(1, "h")).astype(np.int64)
    else:
        values = ptd.values.astype(np.int64)
        # Earth2Studio may store timedelta64 as raw nanoseconds.
        if values.size > 0 and np.nanmax(values) > 10000:
            hours = (values / 3_600_000_000_000).astype(np.int64)
        else:
            hours = values.astype(np.int64)

    ds = ds.assign_coords(prediction_timedelta=hours)

    # Match WeatherBench2 GraphCast: 6, 12, ..., 240. Remove the initial 0h state.
    ds = ds.sel(prediction_timedelta=ds["prediction_timedelta"] > 0)
    return ds


def _stack_pressure_level_variables(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert variables such as t500, z850, u250, q700 into WB2-style variables:
    temperature(level, lat, lon), geopotential(level, lat, lon), etc.
    """
    grouped = {}

    for var in list(ds.data_vars):
        m = re.match(r"^([A-Za-z]+)(\d+)$", var)
        if not m:
            continue

        prefix = m.group(1).lower()
        level = int(m.group(2))

        if prefix not in VAR_PREFIX_MAP:
            continue
        if level not in set(WB2_LEVELS.tolist()):
            continue

        wb2_name = VAR_PREFIX_MAP[prefix]
        grouped.setdefault(wb2_name, []).append((level, var))

    vars_to_drop = []

    for wb2_name, pairs in grouped.items():
        pairs = sorted(pairs, key=lambda x: x[0])
        arrays = []

        for level, var in pairs:
            da = ds[var].expand_dims(level=[np.int64(level)])
            arrays.append(da)
            vars_to_drop.append(var)

        ds[wb2_name] = xr.concat(arrays, dim="level").assign_coords(
            level=np.array([level for level, _ in pairs], dtype=np.int64)
        )

    if vars_to_drop:
        ds = ds.drop_vars(vars_to_drop)

    if "level" in ds.coords:
        ds = ds.sortby("level")

    return ds


def standardize_to_wb2(ds: xr.Dataset) -> xr.Dataset:
    rename_dict = {}

    if "lead_time" in ds.dims or "lead_time" in ds.coords:
        rename_dict["lead_time"] = "prediction_timedelta"

    # Make coordinates match WB2 GraphCast names.
    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"

    ds = ds.rename(rename_dict)

    # Rename common single-level variables to WB2-style names when present.
    surface_rename = {
        old: new for old, new in SURFACE_VAR_MAP.items()
        if old in ds.data_vars and new not in ds.data_vars
    }
    if surface_rename:
        ds = ds.rename(surface_rename)

    ds = _standardize_time(ds)
    ds = _standardize_prediction_timedelta(ds)

    if "lon" in ds.coords:
        lon = ds["lon"]
        if float(lon.min()) < 0:
            ds = ds.assign_coords(lon=(lon % 360))
            ds = ds.sortby("lon")

    # Match WB2 GraphCast latitude order: -90, -89.75, ..., 90.
    if "lat" in ds.coords and ds.sizes.get("lat", 0) > 1:
        if float(ds["lat"].values[0]) > float(ds["lat"].values[-1]):
            ds = ds.sortby("lat")

    ds = _stack_pressure_level_variables(ds)

    ds.attrs["dataset_type"] = "forecast"
    ds.attrs["model"] = "FengWu"
    ds.attrs["year"] = YEAR
    ds.attrs["format"] = "WeatherBench2-style forecast archive"
    ds.attrs["forecast_reference_time_dimension"] = "time"
    ds.attrs["lead_time_dimension"] = "prediction_timedelta"

    return ds


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Stop before running FengWu.")

device = torch.device("cuda")
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

scratch = os.environ.get("SCRATCH", f"/scratch/{os.environ['USER']}")
base_dir = Path(scratch) / "fengwu_outputs"

daily_dir = base_dir / "fengwu_2020_daily_wb2"
archive_dir = base_dir / "fengwu_2020_wb2_archive"

daily_dir.mkdir(parents=True, exist_ok=True)
archive_dir.mkdir(parents=True, exist_ok=True)

tmp_root = Path(os.environ.get("SLURM_TMPDIR", daily_dir))
tmp_root.mkdir(parents=True, exist_ok=True)

print(f"Daily archive: {daily_dir}")
print(f"Archive helper path: {archive_dir}")
print(f"Temporary working dir: {tmp_root}")

package = FengWu.load_default_package()
model = FengWu.load_model(package)
model.eval()

data = ARCO()

init_times = init_times_for_year(YEAR)
print(f"Total initialization times: {len(init_times)}")

for current_time in init_times:
    date_str = current_time.strftime("%Y%m%d%H")

    raw_tmp = tmp_root / f"tmp_raw_{date_str}.zarr"
    wb2_tmp = tmp_root / f"tmp_fengwu_{date_str}.zarr"
    wb2_out = daily_dir / f"fengwu_{date_str}.zarr"

    if wb2_out.exists():
        print(f"[SKIP] {current_time} already exists: {wb2_out}")
        continue

    for p in [raw_tmp, wb2_tmp]:
        if p.exists():
            shutil.rmtree(p)

    print(f"[RUN] FengWu forecast initialized at {current_time}")

    io = ZarrBackend(str(raw_tmp))

    with torch.inference_mode():
        deterministic(
            time=[current_time],
            nsteps=N_STEPS,
            prognostic=model,
            data=data,
            io=io,
            device=device,
        )

    with xr.open_zarr(raw_tmp, consolidated=False) as ds:
        ds_wb2 = standardize_to_wb2(ds)
        ds_wb2.to_zarr(wb2_tmp, mode="w", consolidated=False)

    if wb2_out.exists():
        shutil.rmtree(wb2_out)

    shutil.move(str(wb2_tmp), str(wb2_out))

    if raw_tmp.exists():
        shutil.rmtree(raw_tmp)

    print(f"[DONE] WB2-style daily store saved: {wb2_out}")

read_script = archive_dir / "open_fengwu_2020_wb2.py"

read_script.write_text(
f'''
from pathlib import Path
import xarray as xr

daily_dir = Path("{daily_dir}")

stores = sorted(daily_dir.glob("fengwu_*.zarr"))

if len(stores) == 0:
    raise RuntimeError(f"No zarr stores found in {{daily_dir}}")

ds = xr.open_mfdataset(
    [str(s) for s in stores],
    engine="zarr",
    combine="nested",
    concat_dim="time",
    consolidated=False,
    chunks={{
        "time": 1,
        "prediction_timedelta": 40,
        "level": 37,
    }},
)

print(ds)
print("prediction_timedelta =", ds.prediction_timedelta.values[:], "count=", ds.sizes.get("prediction_timedelta"))
if "level" in ds.coords:
    print("level count=", ds.sizes.get("level"))
    print("level values=", ds.level.values)
'''.strip()
)

print("Finished generating WB2-style FengWu forecast archive.")
print(f"Daily archive path: {daily_dir}")
print(f"Open script: {read_script}")
