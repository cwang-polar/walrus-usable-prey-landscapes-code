"""Build ship-traffic disturbance index (STDI) layers from AIS records.

The manuscript uses public s-AIS records to calculate a grid-cell ship
exposure layer. The working definition implemented here is:

1. assign AIS vessel positions to the same 5 km grid used for prey biomass;
2. count unique MMSI values per grid cell for each month;
3. average monthly counts within a biological season;
4. log-scale normalize each seasonal surface to [0, 1];
5. average seasonal surfaces across five-year ship windows.

Input AIS files are not included in this peer-review code package.
"""

from __future__ import annotations

import argparse
import glob
import os
from functools import reduce
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def assign_vessels_to_grid(
    df: pd.DataFrame,
    origin_x: float,
    origin_y: float,
    cols: int,
    rows: int,
    cell_size: float = 5000.0,
    epsg_from: str = "EPSG:4326",
    epsg_to: str = "EPSG:3338",
    mmsi_col: str = "mmsi",
    lon_col: str = "x",
    lat_col: str = "y",
) -> pd.DataFrame:
    """Assign AIS points to 5 km grid cells and de-duplicate MMSI per cell."""

    from pyproj import Transformer

    required = {mmsi_col, lon_col, lat_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"AIS table missing required columns: {sorted(missing)}")

    ais = df.copy()
    ais[mmsi_col] = pd.to_numeric(ais[mmsi_col], errors="coerce")
    ais[lon_col] = pd.to_numeric(ais[lon_col], errors="coerce")
    ais[lat_col] = pd.to_numeric(ais[lat_col], errors="coerce")
    ais = ais.dropna(subset=[mmsi_col, lon_col, lat_col])
    ais = ais[
        (ais[lon_col] >= -180)
        & (ais[lon_col] <= 180)
        & (ais[lat_col] >= -90)
        & (ais[lat_col] <= 90)
    ].copy()
    if ais.empty:
        return pd.DataFrame(columns=["grid_x", "grid_y", "mmsi"])

    transformer = Transformer.from_crs(epsg_from, epsg_to, always_xy=True)
    x_proj, y_proj = transformer.transform(ais[lon_col].values, ais[lat_col].values)
    ais["X"] = x_proj
    ais["Y"] = y_proj
    ais = ais[np.isfinite(ais["X"]) & np.isfinite(ais["Y"])].copy()
    if ais.empty:
        return pd.DataFrame(columns=["grid_x", "grid_y", "mmsi"])

    ais["grid_x"] = ((ais["X"] - origin_x) // cell_size).astype(int)
    ais["grid_y"] = ((ais["Y"] - origin_y) // cell_size).astype(int)
    ais = ais[
        (ais["grid_x"] >= 0)
        & (ais["grid_x"] < cols)
        & (ais["grid_y"] >= 0)
        & (ais["grid_y"] < rows)
    ].copy()
    if ais.empty:
        return pd.DataFrame(columns=["grid_x", "grid_y", "mmsi"])

    return (
        ais[["grid_x", "grid_y", mmsi_col]]
        .rename(columns={mmsi_col: "mmsi"})
        .drop_duplicates()
        .reset_index(drop=True)
    )


def season_months(year: int, season: str) -> list[tuple[int, int]]:
    """Return calendar year-month pairs for manuscript summer or winter season."""

    season_lower = season.lower()
    if season_lower == "summer":
        return [(year, month) for month in [6, 7, 8, 9, 10]]
    if season_lower == "winter":
        return [(year - 1, 12)] + [(year, month) for month in [1, 2, 3, 4]]
    raise ValueError("season must be 'summer' or 'winter'")


def find_ais_month_files(ais_dir: str | Path, year: int, month: int) -> list[str]:
    """Find AIS CSV batches for one month using the public file-name pattern."""

    month_str = str(month)
    patterns = [
        os.path.join(str(ais_dir), f"ArcticSea_{year}_{month_str}_*_ais*.csv"),
        os.path.join(str(ais_dir), f"ArcticSea_{year}_{month_str}_ais*.csv"),
    ]
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(set(files))


def full_grid_index(cols: int, rows: int) -> pd.MultiIndex:
    return pd.MultiIndex.from_product([range(cols), range(rows)], names=["grid_x", "grid_y"])


def compute_monthly_unique_vessels(
    file_list: Sequence[str | Path],
    origin_x: float,
    origin_y: float,
    cols: int,
    rows: int,
    cell_size: float = 5000.0,
) -> pd.DataFrame:
    """Count unique vessels per grid cell for all AIS files in one month."""

    month_records = []
    for file_path in file_list:
        df_raw = pd.read_csv(file_path)
        assigned = assign_vessels_to_grid(
            df_raw,
            origin_x=origin_x,
            origin_y=origin_y,
            cols=cols,
            rows=rows,
            cell_size=cell_size,
        )
        if not assigned.empty:
            month_records.append(assigned)

    all_cells = full_grid_index(cols, rows)
    if not month_records:
        return all_cells.to_frame(index=False).assign(mmsi_count=0)

    month_all = pd.concat(month_records, ignore_index=True).drop_duplicates()
    nonzero = (
        month_all.groupby(["grid_x", "grid_y"])["mmsi"]
        .nunique()
        .reset_index()
        .rename(columns={"mmsi": "mmsi_count"})
    )
    return nonzero.set_index(["grid_x", "grid_y"]).reindex(all_cells, fill_value=0).reset_index()


def compute_seasonal_stdi(
    year: int,
    season: str,
    ais_dir: str | Path,
    origin_x: float,
    origin_y: float,
    cols: int,
    rows: int,
    cell_size: float = 5000.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute seasonal grid-level STDI for one site and year."""

    monthly_results = []
    for yy, mm in season_months(year, season):
        files = find_ais_month_files(ais_dir, yy, mm)
        if not files:
            if verbose:
                print(f"No AIS files for {yy}-{mm}; month skipped.")
            continue
        month_counts = compute_monthly_unique_vessels(
            files,
            origin_x=origin_x,
            origin_y=origin_y,
            cols=cols,
            rows=rows,
            cell_size=cell_size,
        )
        month_counts["month_year"] = f"{yy}-{mm}"
        monthly_results.append(month_counts)

    if not monthly_results:
        return pd.DataFrame(columns=["grid_x", "grid_y", "mean_mmsi_count", "STDI_norm"])

    all_months = pd.concat(monthly_results, ignore_index=True)
    seasonal_mean = (
        all_months.groupby(["grid_x", "grid_y"])["mmsi_count"]
        .mean()
        .reset_index()
        .rename(columns={"mmsi_count": "mean_mmsi_count"})
    )
    max_count = seasonal_mean["mean_mmsi_count"].max()
    if max_count > 0:
        seasonal_mean["STDI_norm"] = np.log1p(seasonal_mean["mean_mmsi_count"]) / np.log1p(max_count)
    else:
        seasonal_mean["STDI_norm"] = 0.0
    return seasonal_mean


def compute_window_stdi(
    stdi_dict: Mapping[int, pd.DataFrame],
    window_years: Sequence[int],
    value_col: str = "STDI_norm",
) -> pd.DataFrame:
    """Average annual seasonal STDI surfaces across a multi-year window."""

    dfs = []
    for year in window_years:
        if year not in stdi_dict:
            raise KeyError(f"Year {year} not found in stdi_dict")
        df_year = stdi_dict[year][["grid_x", "grid_y", value_col]].copy()
        dfs.append(df_year.rename(columns={value_col: f"STDI_{year}"}))

    merged = reduce(lambda left, right: left.merge(right, on=["grid_x", "grid_y"], how="outer"), dfs)
    stdi_cols = [f"STDI_{year}" for year in window_years]
    merged["STDI_window_mean"] = merged[stdi_cols].mean(axis=1)
    return merged


def build_seasonal_stdi_dict(
    years: Sequence[int],
    season: str,
    ais_dir: str | Path,
    origin_x: float,
    origin_y: float,
    cols: int,
    rows: int,
    cell_size: float = 5000.0,
) -> dict[int, pd.DataFrame]:
    """Compute STDI surfaces for all annual seasons in a list."""

    return {
        year: compute_seasonal_stdi(
            year,
            season=season,
            ais_dir=ais_dir,
            origin_x=origin_x,
            origin_y=origin_y,
            cols=cols,
            rows=rows,
            cell_size=cell_size,
        )
        for year in years
    }


def build_ship_window_layers(
    years: Sequence[int],
    windows: Mapping[str, Sequence[int]],
    season: str,
    ais_dir: str | Path,
    origin_x: float,
    origin_y: float,
    cols: int,
    rows: int,
    cell_size: float = 5000.0,
) -> dict[str, pd.DataFrame]:
    """Compute all five-year STDI layers for one site."""

    annual = build_seasonal_stdi_dict(
        years,
        season=season,
        ais_dir=ais_dir,
        origin_x=origin_x,
        origin_y=origin_y,
        cols=cols,
        rows=rows,
        cell_size=cell_size,
    )
    return {label: compute_window_stdi(annual, year_list) for label, year_list in windows.items()}


def annual_stdi_summary(stdi_dict: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """Summarize annual STDI intensity for Appendix ship-traffic statistics."""

    rows = []
    for year, df in sorted(stdi_dict.items()):
        active = df[df["STDI_norm"] > 0]
        rows.append(
            {
                "year": year,
                "total_STDI": float(df["STDI_norm"].sum()),
                "mean_STDI": float(df["STDI_norm"].mean()),
                "active_grid_cells": int(len(active)),
                "mean_active_STDI": float(active["STDI_norm"].mean()) if not active.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_table_s3_ship_traffic_statistics(stdi_dict: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """Appendix Table S3: annual ship-traffic/STDI statistics."""

    return annual_stdi_summary(stdi_dict)


def compute_policy_ship_impact(
    df_biomass: pd.DataFrame,
    df_stdi: pd.DataFrame,
    alpha: float,
    species_cols: Sequence[str] = (
        "polychaeta_avg_final",
        "bivalvia_avg_final",
        "sipuncula_avg_final",
    ),
) -> pd.DataFrame:
    """Apply a ship-disturbance penalty to biomass as a diagnostic layer."""

    merged = df_biomass.merge(
        df_stdi[["grid_x", "grid_y", "STDI_window_mean"]],
        on=["grid_x", "grid_y"],
        how="left",
    )
    merged["STDI_window_mean"] = merged["STDI_window_mean"].fillna(0.0)
    penalty = 1.0 - alpha * merged["STDI_window_mean"]
    for col in species_cols:
        merged[f"{col}_ship"] = merged[col] * penalty
    merged["total_biomass_ship"] = merged[[f"{col}_ship" for col in species_cols]].sum(axis=1)
    return merged


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="print module purpose and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.describe:
        print("Builds seasonal and five-year STDI ship-exposure layers from AIS CSV files.")
        print("Requires external AIS files with mmsi, x, and y columns.")
    else:
        print("Import this module and call compute_seasonal_stdi or build_ship_window_layers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
