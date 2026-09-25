"""Prepare station-level benthic prey data and gridded biomass surfaces.

This module contains the working Python logic used to build the 5 km benthic
prey biomass surfaces and biomass time-series summaries used in the manuscript.
Input station data are not distributed in this peer-review code package; see
``data_availability_note.md`` for the data-access plan.

The implementation keeps the analysis logic explicit and parameterized so
reviewers can inspect or rerun it when the required input files are available.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


TAXA_SUM_COLUMNS: Mapping[str, str] = {
    "polychaeta": "sum_gc_Polychaeta_1",
    "bivalvia": "sum_gc_Bivalvia_1",
    "sipuncula": "sum_gc_Sipuncula_1",
}

TAXA_AVG_COLUMNS: Mapping[str, str] = {
    "polychaeta": "polychaeta_avg",
    "bivalvia": "bivalvia_avg",
    "sipuncula": "sipuncula_avg",
}

TAXA_FINAL_COLUMNS = (
    "polychaeta_avg_final",
    "bivalvia_avg_final",
    "sipuncula_avg_final",
)

# Settings used to generate the bootstrap summaries shown in manuscript Figure 3.
FIGURE_3_BOOTSTRAP_REPLICATES = 200
FIGURE_3_BOOTSTRAP_RANDOM_STATE = 123


DBO_REGIONS: dict[str, list[tuple[float, float]]] = {
    "DBO1": [
        (63.7694620, -172.3124220),
        (61.9459880, -172.1869270),
        (61.8465160, -175.7909780),
        (63.6627180, -176.1471110),
    ],
    "DBO2": [
        (65.1077470, -170.8651760),
        (65.0970010, -167.8595490),
        (64.4818280, -167.9077580),
        (64.4936580, -170.8406690),
    ],
    "DBO3": [
        (66.7858800, -171.3300080),
        (68.6088690, -171.4193340),
        (68.5724120, -166.4810290),
        (66.7522810, -166.7558080),
    ],
    "DBO4": [
        (71.9940000, -164.2260000),
        (71.8010000, -159.7290000),
        (70.6410000, -160.3050000),
        (70.8210000, -164.5560000),
    ],
    "DBO5": [
        (71.8077300, -158.5376910),
        (71.6355830, -155.9312200),
        (71.1109410, -156.3072770),
        (71.2782770, -158.8479200),
    ],
    "DBO6": [
        (71.8500000, -150.1080000),
        (70.8900000, -150.9760000),
        (71.1590000, -153.8650000),
        (72.1190000, -152.9800000),
    ],
    "DBO7": [
        (70.8680000, -141.5280000),
        (69.9460000, -142.6790000),
        (70.3330000, -145.4390000),
        (71.2530000, -144.2560000),
    ],
    "DBO8": [
        (70.8340000, -125.1550000),
        (70.0490000, -126.9970000),
        (70.6640000, -129.3540000),
        (71.4420000, -127.4310000),
    ],
}


@dataclass(frozen=True)
class GridSpec:
    """Projected 5 km grid definition for one DBO region."""

    origin_x: float
    origin_y: float
    cols: int
    rows: int
    cell_size: float = 5000.0
    crs: str = "EPSG:3338"


def require_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def make_sliding_windows(
    start_year: int,
    end_year: int,
    window_size: int = 5,
    step: int = 3,
    extra_windows: Sequence[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Return sliding year windows used for DBO surface construction."""

    windows: list[tuple[int, int]] = []
    year = start_year
    while year + window_size - 1 <= end_year:
        windows.append((year, year + window_size - 1))
        year += step
    for window in extra_windows or ():
        if window not in windows:
            windows.append(window)
    return windows


def generate_grid_origin_and_shape(
    corner_coords: Sequence[tuple[float, float]],
    cell_size: float = 5000.0,
    epsg_from: str = "EPSG:4326",
    epsg_to: str = "EPSG:3338",
) -> GridSpec:
    """Project DBO corner coordinates and return the 5 km grid extent."""

    from pyproj import Transformer

    transformer = Transformer.from_crs(epsg_from, epsg_to, always_xy=True)
    projected = [transformer.transform(lon, lat) for lat, lon in corner_coords]
    xs, ys = zip(*projected)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cols = int(np.ceil((max_x - min_x) / cell_size))
    rows = int(np.ceil((max_y - min_y) / cell_size))
    return GridSpec(min_x, min_y, cols, rows, cell_size=cell_size, crs=epsg_to)


def extract_region_data(
    df: pd.DataFrame,
    corner_coords: Sequence[tuple[float, float]],
    lon_col: str = "Longitude_Center",
    lat_col: str = "Latitude_Center",
    crs: str = "EPSG:4326",
) -> pd.DataFrame:
    """Filter station rows to one polygonal DBO region."""

    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    require_columns(df, [lon_col, lat_col], "station table")
    polygon = Polygon([(lon, lat) for lat, lon in corner_coords])
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)
    filtered = gdf[gdf.geometry.within(polygon)].copy()
    return pd.DataFrame(filtered.drop(columns=["geometry"]))


def extract_all_dbo_and_summarize(
    data_path: str | Path,
    output_excel: str | Path,
    regions: Mapping[str, Sequence[tuple[float, float]]] = DBO_REGIONS,
) -> pd.DataFrame:
    """Write DBO-specific station sheets and return year-count summaries."""

    data = pd.read_csv(data_path)
    valid_rows = data[data["DataYear"].notna()].copy()
    summaries: list[pd.DataFrame] = []

    with pd.ExcelWriter(output_excel) as writer:
        for name, coords in regions.items():
            filtered = extract_region_data(valid_rows, coords)
            filtered.to_excel(writer, sheet_name=f"{name}_data", index=False)
            counts = (
                filtered["DataYear"]
                .value_counts()
                .sort_index()
                .rename_axis("Year")
                .reset_index(name="Count")
            )
            counts["Region"] = name
            summaries.append(counts[["Region", "Year", "Count"]])

        summary = pd.concat(summaries, ignore_index=True)
        summary.to_excel(writer, sheet_name="Summary_by_year", index=False)

    return summary


def krige_species(
    species_col: str,
    observed_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    grid: GridSpec,
    variogram_model: str = "spherical",
    nlags: int = 6,
) -> pd.DataFrame:
    """Interpolate one taxon-specific biomass column onto the full 5 km grid."""

    from pykrige.ok import OrdinaryKriging

    observed = observed_df.dropna(subset=[species_col]).copy()
    if observed.empty:
        return pd.DataFrame(
            {"grid_index": grid_df["grid_index"], f"{species_col}_kriged": np.nan}
        )

    observed["grid_x"] = observed["grid_index"].apply(lambda item: item[0]).astype(int)
    observed["grid_y"] = observed["grid_index"].apply(lambda item: item[1]).astype(int)
    observed["x"] = grid.origin_x + observed["grid_x"] * grid.cell_size
    observed["y"] = grid.origin_y + observed["grid_y"] * grid.cell_size

    ok_model = OrdinaryKriging(
        observed["x"].to_numpy(),
        observed["y"].to_numpy(),
        observed[species_col].to_numpy(),
        variogram_model=variogram_model,
        nlags=nlags,
        weight=True,
        verbose=False,
        enable_plotting=False,
    )

    gridx = np.arange(grid_df["x"].min(), grid_df["x"].max() + grid.cell_size, grid.cell_size)
    gridy = np.arange(grid_df["y"].min(), grid_df["y"].max() + grid.cell_size, grid.cell_size)
    z_interp, _ = ok_model.execute("grid", gridx, gridy)

    xx, yy = np.meshgrid(gridx, gridy)
    interp_df = pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            f"{species_col}_kriged": np.asarray(z_interp).ravel(),
        }
    )
    interp_df["grid_x"] = ((interp_df["x"] - grid.origin_x) / grid.cell_size).astype(int)
    interp_df["grid_y"] = ((interp_df["y"] - grid.origin_y) / grid.cell_size).astype(int)
    interp_df["grid_index"] = list(zip(interp_df["grid_x"], interp_df["grid_y"]))
    return interp_df[["grid_index", f"{species_col}_kriged"]]


def generate_full_grid_surface(
    df_period: pd.DataFrame,
    year_range: str,
    grid: GridSpec,
    lon_col: str = "Longitude_Center",
    lat_col: str = "Latitude_Center",
    point_count_col: str = "Point_Count_1",
) -> pd.DataFrame:
    """Generate the complete gridded prey biomass surface for one time window."""

    import geopandas as gpd
    from shapely.geometry import Point

    required = [lon_col, lat_col, point_count_col, *TAXA_SUM_COLUMNS.values()]
    require_columns(df_period, required, f"station data for {year_range}")

    gdf = gpd.GeoDataFrame(
        df_period.copy(),
        geometry=[Point(xy) for xy in zip(df_period[lon_col], df_period[lat_col])],
        crs="EPSG:4326",
    ).to_crs(grid.crs)

    gdf["grid_x"] = ((gdf.geometry.x - grid.origin_x) // grid.cell_size).astype(int)
    gdf["grid_y"] = ((gdf.geometry.y - grid.origin_y) // grid.cell_size).astype(int)
    gdf["grid_index"] = list(zip(gdf["grid_x"], gdf["grid_y"]))

    aggregated = (
        gdf.groupby("grid_index")
        .agg(
            total_polychaeta=(TAXA_SUM_COLUMNS["polychaeta"], "sum"),
            total_bivalvia=(TAXA_SUM_COLUMNS["bivalvia"], "sum"),
            total_sipuncula=(TAXA_SUM_COLUMNS["sipuncula"], "sum"),
            total_samples=(point_count_col, "sum"),
        )
        .reset_index()
    )

    aggregated["polychaeta_avg"] = aggregated["total_polychaeta"] / aggregated["total_samples"]
    aggregated["bivalvia_avg"] = aggregated["total_bivalvia"] / aggregated["total_samples"]
    aggregated["sipuncula_avg"] = aggregated["total_sipuncula"] / aggregated["total_samples"]

    full_grid = pd.DataFrame(
        [(x, y) for x in range(grid.cols) for y in range(grid.rows)],
        columns=["grid_x", "grid_y"],
    )
    full_grid["grid_index"] = list(zip(full_grid["grid_x"], full_grid["grid_y"]))
    full_grid["x"] = grid.origin_x + full_grid["grid_x"] * grid.cell_size
    full_grid["y"] = grid.origin_y + full_grid["grid_y"] * grid.cell_size

    for taxon, species_col in TAXA_AVG_COLUMNS.items():
        kriged = krige_species(species_col, aggregated.copy(), full_grid, grid)
        full_grid = full_grid.merge(kriged, on="grid_index", how="left")
        full_grid = full_grid.merge(
            aggregated[["grid_index", species_col]], on="grid_index", how="left"
        )
        full_grid[f"{species_col}_final"] = full_grid[species_col].combine_first(
            full_grid[f"{species_col}_kriged"]
        )

    return full_grid


def build_surfaces_for_windows(
    site: str,
    df_site: pd.DataFrame,
    windows: Iterable[tuple[int, int]],
    grid: GridSpec,
    out_dir: str | Path,
    min_points: int = 10,
    save_csv: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build gridded prey surfaces for all valid sliding windows."""

    require_columns(df_site, ["DataYear"], "station table")
    out_path = Path(out_dir)
    if save_csv:
        out_path.mkdir(parents=True, exist_ok=True)

    surfaces: dict[str, pd.DataFrame] = {}
    df_site = df_site[df_site["DataYear"].notna()].copy()
    df_site["DataYear"] = df_site["DataYear"].astype(int)

    for start_year, end_year in windows:
        label = f"{start_year}_{end_year}"
        df_period = df_site[
            (df_site["DataYear"] >= start_year) & (df_site["DataYear"] <= end_year)
        ].copy()
        if len(df_period) < min_points:
            continue
        surface = generate_full_grid_surface(df_period, label, grid)
        surfaces[label] = surface
        if save_csv:
            surface.to_csv(out_path / f"{site}_surface_{label}.csv", index=False)

    return surfaces


def compute_total_biomass_timeseries(
    data_dir: str | Path,
    site_id: str,
    taxa_cols: Sequence[str] = TAXA_FINAL_COLUMNS,
) -> pd.DataFrame:
    """Summarize total gridded prey biomass for each saved surface."""

    pattern = re.compile(rf"{re.escape(site_id)}_surface_(\d{{4}})_(\d{{4}})\.csv$")
    records: list[dict[str, float | str | int]] = []
    for path in sorted(Path(data_dir).glob(f"{site_id}_surface_*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        start_year, end_year = map(int, match.groups())
        df = pd.read_csv(path)
        require_columns(df, taxa_cols, path.name)
        records.append(
            {
                "window_label": f"{start_year}_{end_year}",
                "start_year": start_year,
                "end_year": end_year,
                "mid_year": 0.5 * (start_year + end_year),
                "total_biomass": float(df[list(taxa_cols)].sum(axis=1).sum()),
            }
        )
    return pd.DataFrame(records).sort_values("mid_year").reset_index(drop=True)


def bootstrap_total_biomass_for_window(
    df_period: pd.DataFrame,
    year_label: str,
    grid: GridSpec,
    n_boot: int = FIGURE_3_BOOTSTRAP_REPLICATES,
    min_points: int = 10,
    random_state: int | None = None,
) -> np.ndarray:
    """Bootstrap station rows, rebuild surfaces, and sum total prey biomass."""

    if len(df_period) < min_points:
        return np.array([])

    rng = np.random.default_rng(random_state)
    totals: list[float] = []
    for boot_id in range(n_boot):
        sample_idx = rng.integers(0, len(df_period), size=len(df_period))
        df_boot = df_period.iloc[sample_idx].reset_index(drop=True)
        surface = generate_full_grid_surface(df_boot, f"{year_label}_boot{boot_id}", grid)
        for col in TAXA_FINAL_COLUMNS:
            if col not in surface:
                surface[col] = 0.0
            surface[col] = surface[col].fillna(0.0)
        totals.append(float(surface[list(TAXA_FINAL_COLUMNS)].sum(axis=1).sum()))
    return np.array(totals)


def bootstrap_biomass_timeseries(
    df_site: pd.DataFrame,
    windows: Iterable[tuple[int, int]],
    grid: GridSpec,
    n_boot: int = FIGURE_3_BOOTSTRAP_REPLICATES,
    min_points: int = 10,
    random_state: int = FIGURE_3_BOOTSTRAP_RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Return mean and 95 percent bootstrap intervals for each window."""

    stats: list[dict[str, float | str | int]] = []
    boot_values: dict[str, np.ndarray] = {}
    for start_year, end_year in windows:
        label = f"{start_year}_{end_year}"
        df_period = df_site[
            (df_site["DataYear"] >= start_year) & (df_site["DataYear"] <= end_year)
        ].copy()
        totals = bootstrap_total_biomass_for_window(
            df_period,
            label,
            grid,
            n_boot=n_boot,
            min_points=min_points,
            random_state=random_state,
        )
        if totals.size == 0:
            continue
        boot_values[label] = totals
        stats.append(
            {
                "window_label": label,
                "start_year": start_year,
                "end_year": end_year,
                "mid_year": 0.5 * (start_year + end_year),
                "mean_total": float(np.mean(totals)),
                "lower_95": float(np.percentile(totals, 2.5)),
                "upper_95": float(np.percentile(totals, 97.5)),
            }
        )
    return pd.DataFrame(stats).sort_values("mid_year").reset_index(drop=True), boot_values


def gini(values: Sequence[float]) -> float:
    """Gini index for non-negative biomass values."""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    if arr.size == 0:
        return float("nan")
    if np.all(arr == 0):
        return 0.0
    arr_sorted = np.sort(arr)
    n = arr_sorted.size
    cumulative = np.cumsum(arr_sorted)
    return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)


def compute_gini_timeseries_for_site(
    folder: str | Path,
    site_prefix: str,
    taxa_cols: Sequence[str] = TAXA_FINAL_COLUMNS,
) -> pd.DataFrame:
    """Compute prey-biomass Gini values for each saved surface."""

    pattern = re.compile(rf"{re.escape(site_prefix)}_surface_(\d{{4}})_(\d{{4}})\.csv$")
    rows: list[dict[str, float | str | int]] = []
    for path in sorted(Path(folder).glob(f"{site_prefix}_surface_*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        start_year, end_year = map(int, match.groups())
        df = pd.read_csv(path)
        if not all(col in df.columns for col in taxa_cols):
            continue
        total = df[list(taxa_cols)].sum(axis=1)
        rows.append(
            {
                "window_label": f"{start_year}_{end_year}",
                "start_year": start_year,
                "end_year": end_year,
                "mid_year": 0.5 * (start_year + end_year),
                "gini": gini(total),
            }
        )
    return pd.DataFrame(rows).sort_values("mid_year").reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="print module purpose and exit")
    parser.add_argument("--input-excel", type=Path, help="DBO_data_and_summary.xlsx")
    parser.add_argument("--site", choices=["DBO1", "DBO4"], help="site to process")
    parser.add_argument("--sheet", help="Excel sheet name, for example DBO1_data")
    parser.add_argument("--output-dir", type=Path, help="directory for surface CSV files")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--extra-window", action="append", help="extra window as START:END")
    return parser


def parse_extra_windows(items: Sequence[str] | None) -> list[tuple[int, int]]:
    windows = []
    for item in items or []:
        start, end = item.split(":")
        windows.append((int(start), int(end)))
    return windows


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.describe or not args.input_excel:
        print("Builds DBO benthic prey surfaces, biomass summaries, bootstrap CIs, and Gini series.")
        print("Required data are not included in this review repository.")
        return 0

    if not (args.site and args.sheet and args.output_dir and args.start_year and args.end_year):
        parser.error("--site, --sheet, --output-dir, --start-year, and --end-year are required")

    df_site = pd.read_excel(args.input_excel, sheet_name=args.sheet)
    grid = generate_grid_origin_and_shape(DBO_REGIONS[args.site])
    windows = make_sliding_windows(
        args.start_year,
        args.end_year,
        extra_windows=parse_extra_windows(args.extra_window),
    )
    surfaces = build_surfaces_for_windows(args.site, df_site, windows, grid, args.output_dir)
    print(f"Wrote {len(surfaces)} surface files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
