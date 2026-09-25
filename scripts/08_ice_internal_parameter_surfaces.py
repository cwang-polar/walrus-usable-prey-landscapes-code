"""Build the six ice-accessibility scenarios reported in Tables S5-S6."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from revision_support import (
    DBO1_WINDOWS,
    DBO4_WINDOWS,
    DBO_REGIONS,
    INPUT_DIR,
    MAIN_BETA,
    NETCDF_FILES,
    OUTPUT_DIR,
    SPECIES_COLS,
    ensure_safe_paths,
    generate_grid_origin_and_shape,
    load_review_script,
    tuples_from_labels,
    write_json,
)


OUT_DIR = OUTPUT_DIR / "sensitivity"


SCENARIOS = [
    {
        "scenario": "baseline",
        "sic_low": 0.15,
        "sic_high": 0.80,
        "sic_high_weight": 0.70,
        "sit_low": 0.20,
        "sit_high": 0.30,
        "sit_low_weight": 0.10,
        "sit_mid_weight": 0.50,
        "sit_high_weight": 1.00,
        "good_cutoff": 0.60,
        "d_core_km": 15.0,
        "d_max_km": 50.0,
        "nonlocal_multiplier": 0.60,
    },
    {
        "scenario": "distance_narrow_10_40",
        "sic_low": 0.15,
        "sic_high": 0.80,
        "sic_high_weight": 0.70,
        "sit_low": 0.20,
        "sit_high": 0.30,
        "sit_low_weight": 0.10,
        "sit_mid_weight": 0.50,
        "sit_high_weight": 1.00,
        "good_cutoff": 0.60,
        "d_core_km": 10.0,
        "d_max_km": 40.0,
        "nonlocal_multiplier": 0.60,
    },
    {
        "scenario": "distance_broad_20_60",
        "sic_low": 0.15,
        "sic_high": 0.80,
        "sic_high_weight": 0.70,
        "sit_low": 0.20,
        "sit_high": 0.30,
        "sit_low_weight": 0.10,
        "sit_mid_weight": 0.50,
        "sit_high_weight": 1.00,
        "good_cutoff": 0.60,
        "d_core_km": 20.0,
        "d_max_km": 60.0,
        "nonlocal_multiplier": 0.60,
    },
    {
        "scenario": "sic_sit_lenient",
        "sic_low": 0.10,
        "sic_high": 0.85,
        "sic_high_weight": 0.80,
        "sit_low": 0.15,
        "sit_high": 0.25,
        "sit_low_weight": 0.10,
        "sit_mid_weight": 0.50,
        "sit_high_weight": 1.00,
        "good_cutoff": 0.50,
        "d_core_km": 15.0,
        "d_max_km": 50.0,
        "nonlocal_multiplier": 0.60,
    },
    {
        "scenario": "sic_sit_strict",
        "sic_low": 0.20,
        "sic_high": 0.75,
        "sic_high_weight": 0.60,
        "sit_low": 0.25,
        "sit_high": 0.40,
        "sit_low_weight": 0.10,
        "sit_mid_weight": 0.50,
        "sit_high_weight": 1.00,
        "good_cutoff": 0.70,
        "d_core_km": 15.0,
        "d_max_km": 50.0,
        "nonlocal_multiplier": 0.60,
    },
    {
        "scenario": "nonlocal_multiplier_0p37",
        "sic_low": 0.15,
        "sic_high": 0.80,
        "sic_high_weight": 0.70,
        "sit_low": 0.20,
        "sit_high": 0.30,
        "sit_low_weight": 0.10,
        "sit_mid_weight": 0.50,
        "sit_high_weight": 1.00,
        "good_cutoff": 0.60,
        "d_core_km": 15.0,
        "d_max_km": 50.0,
        "nonlocal_multiplier": 0.37,
    },
]


def prepare_land_distances(ice_module, site: str, grid) -> np.ndarray:
    land = gpd.read_file(INPUT_DIR / "ne_10m_land" / "ne_10m_land.shp")
    bbox = gpd.GeoDataFrame(
        geometry=[ice_module.make_bbox_from_corners(DBO_REGIONS[site], margin_deg=1.0)],
        crs="EPSG:4326",
    )
    land_site = gpd.overlay(land, bbox, how="intersection").to_crs(grid.crs)
    grid_points = ice_module.build_grid_points(
        origin_x=grid.origin_x,
        origin_y=grid.origin_y,
        cols=grid.cols,
        rows=grid.rows,
        cell_size_m=grid.cell_size,
        epsg=grid.crs,
    )
    dist_land_m, _points = ice_module.compute_grid_to_land_distance(grid_points, land_site)
    return dist_land_m / 1000.0


def build_ice5km_for_site(ice_module, ds, windows: tuple[str, ...], months: tuple[int, ...], grid) -> dict[tuple[int, int], object]:
    ice5km = {}
    for window in tuples_from_labels(windows):
        ds_win = ice_module.subset_ds_by_window_and_months(ds, window, months)
        if ds_win.sizes.get("time", 0) == 0:
            continue
        ice5km[window] = ice_module.regrid_ice_to_5km(
            ds_win,
            origin_x=grid.origin_x,
            origin_y=grid.origin_y,
            cols=grid.cols,
            rows=grid.rows,
            cell_size=grid.cell_size,
        )
    return ice5km


def build_parameterized_index(ice5km: dict[tuple[int, int], object], dist_land_km: np.ndarray, params: dict[str, float]) -> dict[tuple[int, int], object]:
    from scipy.ndimage import distance_transform_edt

    results = {}
    for window, ds in ice5km.items():
        sic = ds["sic_5km"].values
        sit = ds["sit_5km"].values
        _time_n, rows, cols = sic.shape

        weight_c = np.zeros_like(sic, dtype=float)
        mid_c = (sic >= params["sic_low"]) & (sic <= params["sic_high"])
        weight_c[mid_c] = sic[mid_c]
        weight_c[sic > params["sic_high"]] = params["sic_high_weight"]

        weight_t = np.zeros_like(sit, dtype=float)
        weight_t[sit < params["sit_low"]] = params["sit_low_weight"]
        mid_t = (sit >= params["sit_low"]) & (sit < params["sit_high"])
        weight_t[mid_t] = params["sit_mid_weight"]
        weight_t[sit >= params["sit_high"]] = params["sit_high_weight"]

        h_local = weight_c * weight_t
        a_ice = np.zeros_like(h_local, dtype=float)
        for t_idx in range(h_local.shape[0]):
            h_t = h_local[t_idx]
            good_ice = h_t >= params["good_cutoff"]
            if np.any(good_ice):
                dist_to_ice = distance_transform_edt(~good_ice) * 5.0
                dist_union = np.minimum(dist_to_ice, dist_land_km)
            else:
                dist_union = dist_land_km.copy()

            distance_factor = np.zeros((rows, cols), dtype=float)
            distance_factor[dist_union <= params["d_core_km"]] = 1.0
            mid_d = (dist_union > params["d_core_km"]) & (dist_union <= params["d_max_km"])
            distance_factor[mid_d] = (params["d_max_km"] - dist_union[mid_d]) / (params["d_max_km"] - params["d_core_km"])

            a_t = np.maximum(h_t, params["nonlocal_multiplier"] * distance_factor)
            if np.any(good_ice):
                a_t[good_ice] = 1.0
            a_ice[t_idx] = a_t

        ds_out = ds.copy()
        ds_out["H_local"] = (("time", "y", "x"), h_local)
        ds_out["A_ice"] = (("time", "y", "x"), a_ice)
        results[window] = ds_out
    return results


def biomass_maf_for_surfaces(site: str, windows: tuple[str, ...], ice_surfaces: dict[tuple[int, int], pd.DataFrame], scenario: str) -> pd.DataFrame:
    ice = {f"{key[0]}_{key[1]}": value for key, value in ice_surfaces.items()}
    rows = []
    for window in windows:
        if window not in ice:
            continue
        df_pure = pd.read_csv(INPUT_DIR / f"{site}_surfaces" / f"{site}_surface_{window}.csv")
        pure_total = float(df_pure[list(SPECIES_COLS)].sum(axis=1).sum())
        df = df_pure.merge(
            ice[window][["y_idx", "x_idx", "A_mean"]],
            left_on=["grid_y", "grid_x"],
            right_on=["y_idx", "x_idx"],
            how="left",
        )
        df["A_mean"] = df["A_mean"].fillna(0.0).clip(0.0, 1.0)
        a_beta = 1.0 - MAIN_BETA * (1.0 - df["A_mean"])
        adjusted = 0.0
        for species_col in SPECIES_COLS:
            adjusted += float((df[species_col] * a_beta).sum())
        rows.append(
            {
                "scenario": scenario,
                "site": site,
                "window": window,
                "beta": MAIN_BETA,
                "A_mean_mean": float(df["A_mean"].mean()),
                "total_biomass": pure_total,
                "ice_adjusted_biomass": adjusted,
                "max_attainable_fraction_pct": (adjusted / pure_total) * 100.0 if pure_total > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def gini(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if np.any(arr < 0):
        arr = arr - np.min(arr)
    total = arr.sum()
    if total == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    return float((2.0 * np.sum((np.arange(1, n + 1)) * arr) / (n * total)) - (n + 1.0) / n)


def run_site(ice_module, site: str, windows: tuple[str, ...], ds_path, months: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Preparing 5 km ice fields for {site}")
    grid = generate_grid_origin_and_shape(DBO_REGIONS[site], cell_size=5000.0)
    dist_land_km = prepare_land_distances(ice_module, site, grid)
    ds = xr.open_dataset(ds_path)
    try:
        ice5km = build_ice5km_for_site(ice_module, ds, windows, months, grid)
    finally:
        ds.close()

    all_rows = []
    surface_summaries = []
    for params in SCENARIOS:
        scenario = str(params["scenario"])
        print(f"Running {site} scenario {scenario}")
        index = build_parameterized_index(ice5km, dist_land_km, params)
        surfaces = ice_module.build_ice_surfaces_from_index_dict(index)
        scenario_surface_dir = OUT_DIR / "ice_internal_parameter_surfaces" / scenario / site
        scenario_surface_dir.mkdir(parents=True, exist_ok=True)
        all_rows.append(biomass_maf_for_surfaces(site, windows, surfaces, scenario))
        for window, df in sorted(surfaces.items()):
            window_label = f"{window[0]}_{window[1]}"
            df.to_csv(scenario_surface_dir / f"{site}_ice_surface_{window_label}.csv", index=False)
            surface_summaries.append(
                {
                    "scenario": scenario,
                    "site": site,
                    "window": window_label,
                    "A_mean_min": float(df["A_mean"].min()),
                    "A_mean_mean": float(df["A_mean"].mean()),
                    "A_mean_max": float(df["A_mean"].max()),
                    "A_mean_gini": gini(df["A_mean"]),
                }
            )
    return pd.concat(all_rows, ignore_index=True), pd.DataFrame(surface_summaries)


def describe() -> None:
    print(
        "Rebuilds ice-accessibility surfaces for the baseline and five targeted "
        "internal-parameter alternatives defined in Appendix S1: Table S5."
    )
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUT_DIR}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Describe the analysis without reading data.")
    args = parser.parse_args(argv)
    if args.describe:
        describe()
        return 0

    ensure_safe_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ice_module = load_review_script("ice_accessibility_review", "02_build_ice_accessibility.py")

    scenario_df = pd.DataFrame(SCENARIOS)
    scenario_df.to_csv(OUT_DIR / "ice_internal_parameter_scenario_definitions.csv", index=False)

    dbo1_table, dbo1_surface_summary = run_site(
        ice_module,
        "DBO1",
        DBO1_WINDOWS,
        INPUT_DIR / NETCDF_FILES["DBO1_GLORYS_ICE_TEMP"],
        (12, 1, 2, 3, 4),
    )
    dbo4_table, dbo4_surface_summary = run_site(
        ice_module,
        "DBO4",
        DBO4_WINDOWS,
        INPUT_DIR / NETCDF_FILES["DBO4_NEXTSIM_ICE"],
        (6, 7, 8, 9, 10),
    )

    full = pd.concat([dbo1_table, dbo4_table], ignore_index=True)
    surface_summary = pd.concat([dbo1_surface_summary, dbo4_surface_summary], ignore_index=True)
    full.to_csv(OUT_DIR / "ice_internal_parameter_sensitivity.csv", index=False)
    surface_summary.to_csv(OUT_DIR / "ice_internal_parameter_Amean_summary.csv", index=False)

    baseline = (
        full[full["scenario"].eq("baseline")][["site", "window", "max_attainable_fraction_pct"]]
        .rename(columns={"max_attainable_fraction_pct": "baseline_pct"})
        .copy()
    )
    range_summary = (
        full.groupby(["site", "window"], as_index=False)
        .agg(
            min_pct=("max_attainable_fraction_pct", "min"),
            max_pct=("max_attainable_fraction_pct", "max"),
            scenario_count=("scenario", "nunique"),
        )
    )
    range_summary = range_summary.merge(baseline, on=["site", "window"], how="left")
    range_summary["range_width_pct_points"] = range_summary["max_pct"] - range_summary["min_pct"]
    range_summary.to_csv(OUT_DIR / "ice_internal_parameter_sensitivity_range_by_window.csv", index=False)

    write_json(
        OUT_DIR / "ice_internal_parameter_sensitivity_summary.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "beta": MAIN_BETA,
            "scenario_count": len(SCENARIOS),
            "scenario_definitions": "ice_internal_parameter_scenario_definitions.csv",
            "full_results": "ice_internal_parameter_sensitivity.csv",
            "range_by_window": "ice_internal_parameter_sensitivity_range_by_window.csv",
        },
    )
    print(f"Ice internal-parameter sensitivity outputs written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
