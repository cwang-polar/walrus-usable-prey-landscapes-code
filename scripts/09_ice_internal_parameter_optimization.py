"""Run CPLEX sensitivity curves for the scenarios reported in Table S6."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from revision_support import INPUT_DIR, MAIN_BETA, OUTPUT_DIR, SPECIES_COLS, ensure_safe_paths, load_review_script, write_json


OUT_DIR = OUTPUT_DIR / "sensitivity"
SURFACE_DIR = OUT_DIR / "ice_internal_parameter_surfaces"
THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))

WINDOWS_TO_ANALYZE = {
    "DBO1": ("1993_1997", "2009_2013", "2012_2016", "2017_2020"),
    "DBO4": ("2007_2011", "2009_2013", "2012_2016", "2016_2020"),
}


def pure_surface_path(site: str, window: str) -> Path:
    return INPUT_DIR / f"{site}_surfaces" / f"{site}_surface_{window}.csv"


def scenario_ice_surface_path(scenario: str, site: str, window: str) -> Path:
    return SURFACE_DIR / scenario / site / f"{site}_ice_surface_{window}.csv"


def pure_total(df_pure: pd.DataFrame) -> float:
    return float(df_pure[list(SPECIES_COLS)].sum(axis=1).sum())


def build_ice_beta_surface(df_pure: pd.DataFrame, df_ice: pd.DataFrame, beta: float = MAIN_BETA) -> pd.DataFrame:
    df = df_pure.merge(
        df_ice[["y_idx", "x_idx", "A_mean"]],
        left_on=["grid_y", "grid_x"],
        right_on=["y_idx", "x_idx"],
        how="left",
    )
    df["A_mean"] = df["A_mean"].fillna(0.0).clip(0.0, 1.0)
    df["A_beta"] = 1.0 - beta * (1.0 - df["A_mean"])
    for col in SPECIES_COLS:
        df[f"{col}_beta"] = df[col] * df["A_beta"]
    return df


def exact_ceiling(df: pd.DataFrame, biomass_cols: tuple[str, ...], rhs_total: float) -> float:
    available = float(df[list(biomass_cols)].sum(axis=1).sum())
    return available / rhs_total if rhs_total > 0 else np.nan


def curve_to_records(
    scenario: str,
    site: str,
    window: str,
    layer: str,
    curve: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "scenario": scenario,
            "site": site,
            "window": window,
            "layer": layer,
            "threshold": point["threshold"],
            "feasible": point["feasible"],
            "area_fraction": point["area_fraction"],
            "selected_patches": point["selected_patches"],
            "obj_value": point["obj_value"],
        }
        for point in curve
    ]


def average_area_difference(reference: pd.DataFrame, adjusted: pd.DataFrame) -> dict[str, float | int]:
    merged = reference[["threshold", "area_fraction", "feasible"]].rename(
        columns={"area_fraction": "ref_area", "feasible": "ref_feasible"}
    ).merge(
        adjusted[["threshold", "area_fraction", "feasible"]].rename(
            columns={"area_fraction": "adj_area", "feasible": "adj_feasible"}
        ),
        on="threshold",
        how="inner",
    )
    both = merged[merged["ref_feasible"].eq(True) & merged["adj_feasible"].eq(True)].copy()
    diff_pct_points = (both["adj_area"] - both["ref_area"]) * 100.0
    return {
        "threshold_count_both_feasible": int(len(both)),
        "mean_area_inflation_pct_points": float(diff_pct_points.mean()) if not both.empty else np.nan,
        "median_area_inflation_pct_points": float(diff_pct_points.median()) if not both.empty else np.nan,
        "max_area_inflation_pct_points": float(diff_pct_points.max()) if not both.empty else np.nan,
    }


def read_scenarios() -> list[str]:
    path = OUT_DIR / "ice_internal_parameter_scenario_definitions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run 05_ice_internal_parameter_sensitivity.py first: missing {path}")
    return pd.read_csv(path)["scenario"].astype(str).tolist()


def describe() -> None:
    print(
        "Solves pure-biomass and ice-adjusted minimum-area curves for the six "
        "internal ice-accessibility scenarios and focal intervals in Appendix "
        "S1: Table S6. Run 08_ice_internal_parameter_surfaces.py first."
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
    opt = load_review_script("capacity_optimization_review", "04_run_capacity_optimization.py")
    scenarios = read_scenarios()

    curve_records = []
    result_rows = []
    pure_curves: dict[tuple[str, str], pd.DataFrame] = {}
    pure_totals: dict[tuple[str, str], float] = {}

    for site, windows in WINDOWS_TO_ANALYZE.items():
        for window in windows:
            print(f"Solving pure minimum-area curve for {site} {window}")
            df_pure = pd.read_csv(pure_surface_path(site, window))
            rhs_total = pure_total(df_pure)
            curve = opt.compute_min_area_curve_given_df(
                df_pure,
                biomass_cols=SPECIES_COLS,
                thresholds=THRESHOLDS,
                rhs_total_biomass=rhs_total,
                time_limit=100,
                mipgap=1e-4,
            )
            curve_df = pd.DataFrame(curve)
            pure_curves[(site, window)] = curve_df
            pure_totals[(site, window)] = rhs_total
            curve_records.extend(curve_to_records("pure_reference", site, window, "pure", curve))

    for scenario in scenarios:
        for site, windows in WINDOWS_TO_ANALYZE.items():
            for window in windows:
                print(f"Solving ice curve for {scenario} {site} {window}")
                df_pure = pd.read_csv(pure_surface_path(site, window))
                ice_path = scenario_ice_surface_path(scenario, site, window)
                if not ice_path.exists():
                    raise FileNotFoundError(f"Missing scenario ice surface: {ice_path}")
                df_ice = pd.read_csv(ice_path)
                rhs_total = pure_totals[(site, window)]
                df_ice_beta = build_ice_beta_surface(df_pure, df_ice, MAIN_BETA)
                beta_cols = tuple(f"{col}_beta" for col in SPECIES_COLS)
                curve = opt.compute_min_area_curve_given_df(
                    df_ice_beta,
                    biomass_cols=beta_cols,
                    thresholds=THRESHOLDS,
                    rhs_total_biomass=rhs_total,
                    time_limit=100,
                    mipgap=1e-4,
                )
                curve_df = pd.DataFrame(curve)
                curve_records.extend(curve_to_records(scenario, site, window, "ice", curve))
                area = average_area_difference(pure_curves[(site, window)], curve_df)
                ceiling = exact_ceiling(df_ice_beta, beta_cols, rhs_total) * 100.0
                result_rows.append(
                    {
                        "scenario": scenario,
                        "site": site,
                        "window": window,
                        "beta": MAIN_BETA,
                        "functional_ceiling_pct": ceiling,
                        **area,
                    }
                )

    curves = pd.DataFrame(curve_records)
    results = pd.DataFrame(result_rows)
    aice = pd.read_csv(OUT_DIR / "ice_internal_parameter_Amean_summary.csv")
    key_results = results.merge(
        aice[["scenario", "site", "window", "A_mean_mean", "A_mean_gini"]],
        on=["scenario", "site", "window"],
        how="left",
    )

    baseline = key_results[key_results["scenario"].eq("baseline")][
        ["site", "window", "functional_ceiling_pct", "mean_area_inflation_pct_points", "A_mean_mean", "A_mean_gini"]
    ].rename(
        columns={
            "functional_ceiling_pct": "baseline_functional_ceiling_pct",
            "mean_area_inflation_pct_points": "baseline_mean_area_inflation_pct_points",
            "A_mean_mean": "baseline_A_mean_mean",
            "A_mean_gini": "baseline_A_mean_gini",
        }
    )
    range_summary = (
        key_results.groupby(["site", "window"], as_index=False)
        .agg(
            scenario_count=("scenario", "nunique"),
            ceiling_min_pct=("functional_ceiling_pct", "min"),
            ceiling_max_pct=("functional_ceiling_pct", "max"),
            area_inflation_min_pct_points=("mean_area_inflation_pct_points", "min"),
            area_inflation_max_pct_points=("mean_area_inflation_pct_points", "max"),
            A_mean_min=("A_mean_mean", "min"),
            A_mean_max=("A_mean_mean", "max"),
            A_gini_min=("A_mean_gini", "min"),
            A_gini_max=("A_mean_gini", "max"),
        )
    )
    range_summary = range_summary.merge(baseline, on=["site", "window"], how="left")
    range_summary["ceiling_range_width_pct_points"] = range_summary["ceiling_max_pct"] - range_summary["ceiling_min_pct"]
    range_summary["area_inflation_range_width_pct_points"] = (
        range_summary["area_inflation_max_pct_points"] - range_summary["area_inflation_min_pct_points"]
    )

    curves.to_csv(OUT_DIR / "ice_internal_parameter_cplex_curve_points.csv", index=False)
    results.to_csv(OUT_DIR / "ice_internal_parameter_area_inflation.csv", index=False)
    key_results.to_csv(OUT_DIR / "ice_internal_parameter_key_results.csv", index=False)
    range_summary.to_csv(OUT_DIR / "ice_internal_parameter_key_window_ranges.csv", index=False)
    write_json(
        OUT_DIR / "ice_internal_parameter_cplex_summary.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "beta": MAIN_BETA,
            "thresholds": list(THRESHOLDS),
            "scenarios": scenarios,
            "windows_to_analyze": {key: list(value) for key, value in WINDOWS_TO_ANALYZE.items()},
            "curve_points_csv": "ice_internal_parameter_cplex_curve_points.csv",
            "key_results_csv": "ice_internal_parameter_key_results.csv",
            "range_summary_csv": "ice_internal_parameter_key_window_ranges.csv",
        },
    )
    print(f"Internal-parameter CPLEX sensitivity outputs written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
