"""Check the committed revision-result summaries for internal consistency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
EXPECTED_SCENARIOS = {
    "baseline",
    "distance_narrow_10_40",
    "distance_broad_20_60",
    "sic_sit_lenient",
    "sic_sit_strict",
    "nonlocal_multiplier_0p37",
}
EXPECTED_FOCAL_WINDOWS = 8
EXPECTED_THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_sensitivity_results() -> list[str]:
    scenarios = pd.read_csv(RESULTS_DIR / "ice_internal_parameter_scenario_definitions.csv")
    key = pd.read_csv(RESULTS_DIR / "ice_internal_parameter_key_results.csv")
    ranges = pd.read_csv(RESULTS_DIR / "ice_internal_parameter_key_window_ranges.csv")
    curves = pd.read_csv(RESULTS_DIR / "ice_internal_parameter_cplex_curve_points.csv")

    scenario_names = set(scenarios["scenario"])
    require(scenario_names == EXPECTED_SCENARIOS, "Ice-scenario names or count changed.")
    require(set(key["scenario"]) == EXPECTED_SCENARIOS, "Detailed results do not contain all six scenarios.")
    scenario_index = scenarios.set_index("scenario")
    expected_parameters = {
        "baseline": {"sic_low": 0.15, "sic_high": 0.80, "sit_low": 0.20, "sit_high": 0.30, "good_cutoff": 0.60, "d_core_km": 15.0, "d_max_km": 50.0, "nonlocal_multiplier": 0.60},
        "distance_narrow_10_40": {"d_core_km": 10.0, "d_max_km": 40.0},
        "distance_broad_20_60": {"d_core_km": 20.0, "d_max_km": 60.0},
        "sic_sit_lenient": {"sic_low": 0.10, "sic_high": 0.85, "sit_low": 0.15, "sit_high": 0.25, "good_cutoff": 0.50},
        "sic_sit_strict": {"sic_low": 0.20, "sic_high": 0.75, "sit_low": 0.25, "sit_high": 0.40, "good_cutoff": 0.70},
        "nonlocal_multiplier_0p37": {"nonlocal_multiplier": 0.37},
    }
    for scenario, expected in expected_parameters.items():
        for parameter, value in expected.items():
            require(
                np.isclose(scenario_index.loc[scenario, parameter], value),
                f"Parameter {parameter} changed for scenario {scenario}.",
            )
    require(len(key) == len(EXPECTED_SCENARIOS) * EXPECTED_FOCAL_WINDOWS, "Expected 48 scenario-window results.")
    require(len(ranges) == EXPECTED_FOCAL_WINDOWS, "Expected eight focal-window range summaries.")
    require(key["beta"].eq(0.5).all(), "Internal-parameter analysis must use beta = 0.5.")
    require(key[["functional_ceiling_pct", "mean_area_inflation_pct_points"]].notna().all().all(), "Missing key result.")

    threshold_values = tuple(sorted(curves["threshold"].dropna().unique().round(2)))
    require(threshold_values == EXPECTED_THRESHOLDS, "Curve thresholds differ from 0.05-0.95 by 0.05.")
    require(
        len(curves)
        == (EXPECTED_FOCAL_WINDOWS + len(EXPECTED_SCENARIOS) * EXPECTED_FOCAL_WINDOWS)
        * len(EXPECTED_THRESHOLDS),
        "Unexpected number of CPLEX curve points.",
    )

    pure = curves[curves["scenario"].eq("pure_reference")][
        ["site", "window", "threshold", "feasible", "area_fraction"]
    ].rename(columns={"feasible": "pure_feasible", "area_fraction": "pure_area"})
    adjusted = curves[~curves["scenario"].eq("pure_reference")].merge(
        pure,
        on=["site", "window", "threshold"],
        how="left",
    )
    paired = adjusted[adjusted["feasible"].eq(True) & adjusted["pure_feasible"].eq(True)].copy()
    paired["inflation_pct_points"] = (paired["area_fraction"] - paired["pure_area"]) * 100.0
    recalculated = (
        paired.groupby(["scenario", "site", "window"], as_index=False)
        .agg(
            threshold_count_check=("threshold", "size"),
            mean_inflation_check=("inflation_pct_points", "mean"),
        )
    )
    compared = key.merge(recalculated, on=["scenario", "site", "window"], how="left")
    require(
        compared["threshold_count_both_feasible"].eq(compared["threshold_count_check"]).all(),
        "Feasible-threshold counts do not match the curve points.",
    )
    require(
        np.allclose(
            compared["mean_area_inflation_pct_points"],
            compared["mean_inflation_check"],
            rtol=0.0,
            atol=1e-12,
        ),
        "Mean minimum-area inflation does not match the curve points.",
    )

    recalculated_ranges = (
        key.groupby(["site", "window"], as_index=False)
        .agg(
            ceiling_min_check=("functional_ceiling_pct", "min"),
            ceiling_max_check=("functional_ceiling_pct", "max"),
            area_min_check=("mean_area_inflation_pct_points", "min"),
            area_max_check=("mean_area_inflation_pct_points", "max"),
        )
    )
    range_check = ranges.merge(recalculated_ranges, on=["site", "window"], how="left")
    for reported, calculated in (
        ("ceiling_min_pct", "ceiling_min_check"),
        ("ceiling_max_pct", "ceiling_max_check"),
        ("area_inflation_min_pct_points", "area_min_check"),
        ("area_inflation_max_pct_points", "area_max_check"),
    ):
        require(np.allclose(range_check[reported], range_check[calculated]), f"Range check failed for {reported}.")

    return [
        f"PASS: {len(scenarios)} ice-accessibility scenarios",
        f"PASS: {len(key)} scenario-window sensitivity results",
        f"PASS: {len(curves)} CPLEX curve points at 19 thresholds",
        "PASS: mean minimum-area inflation recalculates from the committed curves",
        "PASS: focal-window ranges recalculate from the detailed results",
    ]


def verify_kriging_results() -> list[str]:
    taxon = pd.read_csv(RESULTS_DIR / "kriging_loocv_taxon_summary.csv")
    table = pd.read_csv(RESULTS_DIR / "kriging_loocv_table_s7.csv")
    summed = table[table["surface"].eq("Total prey (summed)")].copy()

    require(len(taxon) == 51, "Expected 51 taxon-specific LOOCV summaries.")
    require(taxon["n_failures"].sum() == 0, "At least one LOOCV prediction failed.")
    require(len(table) == 68, "Expected 68 Table S7 rows.")
    require(len(summed) == 17, "Expected 17 summed-prey interval summaries.")
    require(table["observed_grid_cells_n"].gt(0).all(), "Invalid observed-grid-cell count.")

    relative_bias = summed["bias"] / summed["observed_mean"] * 100.0
    relative_rmse = summed["rmse"] / summed["observed_mean"] * 100.0
    require(np.isclose(relative_bias.median(), 1.1699992191977437), "Summed-prey median relative bias changed.")
    require(np.isclose(relative_rmse.median(), 40.67806284957447), "Summed-prey median relative RMSE changed.")

    return [
        "PASS: 51 taxon-specific LOOCV summaries with zero failures",
        "PASS: 68 Table S7 rows, including 17 summed-prey rows",
        f"PASS: summed-prey median relative bias = {relative_bias.median():.1f}%",
        f"PASS: summed-prey median relative RMSE = {relative_rmse.median():.1f}%",
    ]


def main() -> int:
    messages = [*verify_sensitivity_results(), *verify_kriging_results()]
    print("Revision-result verification")
    for message in messages:
        print(message)
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
