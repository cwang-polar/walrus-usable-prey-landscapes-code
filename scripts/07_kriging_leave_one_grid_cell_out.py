"""Leave-one-grid-cell-out validation reported in Appendix S1: Table S7."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from shapely.geometry import Point

from revision_support import (
    DBO1_WINDOWS,
    DBO4_WINDOWS,
    DBO_REGIONS,
    INPUT_DIR,
    OUTPUT_DIR,
    ensure_safe_paths,
    load_review_script,
    window_to_tuple,
    write_json,
)


OUT_DIR = OUTPUT_DIR / "validation"
TAXA = {
    "Polychaeta": ("sum_gc_Polychaeta_1", "polychaeta_avg"),
    "Bivalvia": ("sum_gc_Bivalvia_1", "bivalvia_avg"),
    "Sipuncula": ("sum_gc_Sipuncula_1", "sipuncula_avg"),
}


def aggregate_period_to_grid(df_period: pd.DataFrame, grid, point_count_col: str = "Point_Count_1") -> pd.DataFrame:
    required = ["Longitude_Center", "Latitude_Center", point_count_col, *(col for col, _avg in TAXA.values())]
    missing = [col for col in required if col not in df_period.columns]
    if missing:
        raise KeyError(f"Missing required columns for kriging validation: {missing}")

    gdf = gpd.GeoDataFrame(
        df_period.copy(),
        geometry=[Point(xy) for xy in zip(df_period["Longitude_Center"], df_period["Latitude_Center"])],
        crs="EPSG:4326",
    ).to_crs(grid.crs)
    gdf["grid_x"] = ((gdf.geometry.x - grid.origin_x) // grid.cell_size).astype(int)
    gdf["grid_y"] = ((gdf.geometry.y - grid.origin_y) // grid.cell_size).astype(int)
    gdf = gdf[
        (gdf["grid_x"] >= 0)
        & (gdf["grid_x"] < grid.cols)
        & (gdf["grid_y"] >= 0)
        & (gdf["grid_y"] < grid.rows)
    ].copy()
    gdf["grid_index"] = list(zip(gdf["grid_x"], gdf["grid_y"]))

    aggregated = (
        gdf.groupby("grid_index")
        .agg(
            grid_x=("grid_x", "first"),
            grid_y=("grid_y", "first"),
            total_polychaeta=("sum_gc_Polychaeta_1", "sum"),
            total_bivalvia=("sum_gc_Bivalvia_1", "sum"),
            total_sipuncula=("sum_gc_Sipuncula_1", "sum"),
            total_samples=(point_count_col, "sum"),
            station_rows=("grid_index", "size"),
        )
        .reset_index()
    )
    aggregated["polychaeta_avg"] = aggregated["total_polychaeta"] / aggregated["total_samples"]
    aggregated["bivalvia_avg"] = aggregated["total_bivalvia"] / aggregated["total_samples"]
    aggregated["sipuncula_avg"] = aggregated["total_sipuncula"] / aggregated["total_samples"]
    aggregated.replace([np.inf, -np.inf], np.nan, inplace=True)
    return aggregated


def loocv_one_taxon(observed: pd.DataFrame, taxon: str, value_col: str, grid) -> pd.DataFrame:
    data = observed.dropna(subset=[value_col]).copy().reset_index(drop=True)
    data["x"] = grid.origin_x + data["grid_x"].astype(float) * grid.cell_size
    data["y"] = grid.origin_y + data["grid_y"].astype(float) * grid.cell_size

    rows = []
    if len(data) < 4:
        return pd.DataFrame(
            [
                {
                    "taxon": taxon,
                    "grid_x": int(row["grid_x"]),
                    "grid_y": int(row["grid_y"]),
                    "observed": float(row[value_col]),
                    "predicted": np.nan,
                    "error": np.nan,
                    "status": "skipped_too_few_points",
                }
                for _idx, row in data.iterrows()
            ]
        )

    for idx, row in data.iterrows():
        train = data.drop(index=idx)
        try:
            model = OrdinaryKriging(
                train["x"].to_numpy(),
                train["y"].to_numpy(),
                train[value_col].to_numpy(),
                variogram_model="spherical",
                nlags=6,
                weight=True,
                verbose=False,
                enable_plotting=False,
            )
            z_pred, _z_var = model.execute("points", np.array([row["x"]]), np.array([row["y"]]))
            predicted = float(np.asarray(z_pred)[0])
            status = "ok"
        except Exception as exc:
            predicted = np.nan
            status = f"failed:{type(exc).__name__}"
        observed_value = float(row[value_col])
        rows.append(
            {
                "taxon": taxon,
                "grid_x": int(row["grid_x"]),
                "grid_y": int(row["grid_y"]),
                "observed": observed_value,
                "predicted": predicted,
                "error": predicted - observed_value if np.isfinite(predicted) else np.nan,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def summarize_predictions(pred: pd.DataFrame, site: str, window: str, source_grid_cells: int, source_station_rows: int) -> dict[str, object]:
    ok = pred[pred["status"].eq("ok")].copy()
    errors = ok["error"].to_numpy(dtype=float)
    observed = ok["observed"].to_numpy(dtype=float)
    predicted = ok["predicted"].to_numpy(dtype=float)
    ss_res = float(np.sum((observed - predicted) ** 2)) if len(ok) else np.nan
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2)) if len(ok) else np.nan
    return {
        "site": site,
        "window": window,
        "taxon": str(pred["taxon"].iloc[0]) if not pred.empty else "",
        "source_station_rows": source_station_rows,
        "source_grid_cells": source_grid_cells,
        "n_predictions": int(len(ok)),
        "n_failures": int((~pred["status"].eq("ok")).sum()),
        "rmse": float(np.sqrt(np.mean(errors**2))) if len(ok) else np.nan,
        "mae": float(np.mean(np.abs(errors))) if len(ok) else np.nan,
        "bias": float(np.mean(errors)) if len(ok) else np.nan,
        "observed_mean": float(np.mean(observed)) if len(ok) else np.nan,
        "predicted_mean": float(np.mean(predicted)) if len(ok) else np.nan,
        "r2_cv": 1.0 - ss_res / ss_tot if len(ok) and ss_tot > 0 else np.nan,
    }


def run_site(benthic_module, site: str, windows: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = benthic_module.generate_grid_origin_and_shape(benthic_module.DBO_REGIONS[site], cell_size=5000.0)
    df_site = pd.read_excel(INPUT_DIR / "DBO_data_and_summary.xlsx", sheet_name=f"{site}_data")
    df_site = df_site[df_site["DataYear"].notna()].copy()
    df_site["DataYear"] = df_site["DataYear"].astype(int)

    prediction_rows = []
    summary_rows = []
    for window in windows:
        start, end = window_to_tuple(window)
        df_period = df_site[(df_site["DataYear"] >= start) & (df_site["DataYear"] <= end)].copy()
        if len(df_period) < 10:
            continue
        aggregated = aggregate_period_to_grid(df_period, grid)
        for taxon, (_sum_col, avg_col) in TAXA.items():
            print(f"LOOCV {site} {window} {taxon}")
            pred = loocv_one_taxon(aggregated, taxon, avg_col, grid)
            pred.insert(0, "site", site)
            pred.insert(1, "window", window)
            prediction_rows.append(pred)
            summary_rows.append(
                summarize_predictions(
                    pred=pred,
                    site=site,
                    window=window,
                    source_grid_cells=len(aggregated),
                    source_station_rows=len(df_period),
                )
            )
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(summary_rows)


def build_table_s7_summary(summary: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Combine taxon-specific and summed-prey LOOCV statistics."""

    taxon_rows = summary[
        [
            "site",
            "window",
            "taxon",
            "source_grid_cells",
            "observed_mean",
            "predicted_mean",
            "rmse",
            "mae",
            "bias",
            "r2_cv",
        ]
    ].rename(columns={"taxon": "surface", "source_grid_cells": "observed_grid_cells_n"})

    ok = predictions[predictions["status"].eq("ok")].copy()
    counts = (
        ok.groupby(["site", "window", "grid_x", "grid_y"])["taxon"]
        .nunique()
        .reset_index(name="taxon_count")
    )
    complete = ok.merge(counts, on=["site", "window", "grid_x", "grid_y"])
    complete = complete[complete["taxon_count"].eq(len(TAXA))]
    summed_points = (
        complete.groupby(["site", "window", "grid_x", "grid_y"])
        .agg(observed=("observed", "sum"), predicted=("predicted", "sum"))
        .reset_index()
    )
    summed_points["error"] = summed_points["predicted"] - summed_points["observed"]

    total_rows = []
    for (site, window), group in summed_points.groupby(["site", "window"]):
        observed = group["observed"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        errors = group["error"].to_numpy(dtype=float)
        ss_res = float(np.sum((observed - predicted) ** 2))
        ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
        total_rows.append(
            {
                "site": site,
                "window": window,
                "surface": "Total prey (summed)",
                "observed_grid_cells_n": int(len(group)),
                "observed_mean": float(np.mean(observed)),
                "predicted_mean": float(np.mean(predicted)),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mae": float(np.mean(np.abs(errors))),
                "bias": float(np.mean(errors)),
                "r2_cv": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            }
        )

    table = pd.concat([taxon_rows, pd.DataFrame(total_rows)], ignore_index=True)
    order = {name: idx for idx, name in enumerate([*TAXA, "Total prey (summed)"])}
    table["_surface_order"] = table["surface"].map(order)
    return table.sort_values(["site", "window", "_surface_order"]).drop(columns="_surface_order")


def describe() -> None:
    print(
        "Runs leave-one-grid-cell-out ordinary-kriging validation for each site, "
        "interval, and prey taxon, then calculates the summed-prey statistics "
        "reported in Appendix S1: Table S7."
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
    benthic_module = load_review_script("benthic_surfaces_review", "01_prepare_benthic_surfaces.py")

    dbo1_pred, dbo1_summary = run_site(benthic_module, "DBO1", DBO1_WINDOWS)
    dbo4_pred, dbo4_summary = run_site(benthic_module, "DBO4", DBO4_WINDOWS)
    predictions = pd.concat([dbo1_pred, dbo4_pred], ignore_index=True)
    summary = pd.concat([dbo1_summary, dbo4_summary], ignore_index=True)
    table_s7 = build_table_s7_summary(summary, predictions)

    predictions.to_csv(OUT_DIR / "kriging_loocv_predictions.csv", index=False)
    summary.to_csv(OUT_DIR / "kriging_loocv_summary.csv", index=False)
    table_s7.to_csv(OUT_DIR / "kriging_loocv_table_s7.csv", index=False)

    summed = table_s7[table_s7["surface"].eq("Total prey (summed)")].copy()
    summed["relative_bias_pct"] = summed["bias"] / summed["observed_mean"] * 100.0
    summed["relative_rmse_pct"] = summed["rmse"] / summed["observed_mean"] * 100.0
    write_json(
        OUT_DIR / "kriging_validation_summary.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "leave-one-grid-cell-out OrdinaryKriging validation after station rows were aggregated to the manuscript 5 km grid",
            "variogram_model": "spherical",
            "nlags": 6,
            "weight": True,
            "taxon_specific_runs": int(len(summary)),
            "prediction_failures": int(summary["n_failures"].sum()),
            "summed_prey_median_relative_bias_pct": float(summed["relative_bias_pct"].median()),
            "summed_prey_median_relative_rmse_pct": float(summed["relative_rmse_pct"].median()),
            "summary_csv": "kriging_loocv_summary.csv",
            "table_s7_csv": "kriging_loocv_table_s7.csv",
            "predictions_csv": "kriging_loocv_predictions.csv",
        },
    )
    print(f"Kriging validation outputs written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
