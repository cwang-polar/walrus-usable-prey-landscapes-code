"""Minimum-area foraging-capacity optimization.

Main-text analyses use a no-connectivity minimum-area model:

    minimize number of selected grid cells
    subject to selected biomass >= threshold * reference biomass

The reference biomass is the pure prey-biomass total. Ice and ship analyses use
functional biomass weights on the left-hand side while retaining the pure
biomass target on the right-hand side.

Appendix sensitivity analyses also include an adjacency-penalty variant. It is
kept separate below as ``solve_min_area_with_adjacency_penalty``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_SPECIES_COLS = (
    "polychaeta_avg_final",
    "bivalvia_avg_final",
    "sipuncula_avg_final",
)

DEFAULT_THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))


class OptimizationState:
    """Mutable state matching the original CPLEX workflow."""

    def __init__(self) -> None:
        self.patches: list[tuple[int, int]] = []
        self.cost: dict[tuple[int, int], float] = {}
        self.weight: dict[tuple[int, int], float] = {}
        self.total_biomass: float = 0.0


STATE = OptimizationState()


def require_cplex():
    """Import CPLEX only when an optimization solve is actually requested."""

    try:
        import cplex
        from cplex.exceptions import CplexError
    except Exception as exc:  # pragma: no cover - depends on licensed CPLEX
        raise ImportError(
            "IBM ILOG CPLEX Python API is required to solve optimization models. "
            "Install CPLEX according to IBM license terms before running this step."
        ) from exc
    return cplex, CplexError


def patch_name(patch: tuple[int, int] | object) -> str:
    return f"{patch[0]}_{patch[1]}" if isinstance(patch, tuple) else str(patch)


def prepare_min_area_state(
    full_grid: pd.DataFrame,
    biomass_cols: Sequence[str],
    state: OptimizationState = STATE,
) -> OptimizationState:
    """Load one gridded biomass surface into the optimization state."""

    missing = [col for col in ["grid_x", "grid_y", *biomass_cols] if col not in full_grid.columns]
    if missing:
        raise KeyError(f"Optimization grid missing required columns: {missing}")

    grid = full_grid.copy()
    grid["grid_x"] = grid["grid_x"].astype(int)
    grid["grid_y"] = grid["grid_y"].astype(int)
    grid["grid_index"] = list(zip(grid["grid_x"], grid["grid_y"]))
    grid["total_biomass_for_or"] = grid[list(biomass_cols)].sum(axis=1)

    state.patches = list(set(grid["grid_index"]))
    state.cost = {patch: 1.0 for patch in state.patches}
    state.weight = dict(zip(grid["grid_index"], grid["total_biomass_for_or"]))
    state.total_biomass = float(grid["total_biomass_for_or"].sum())
    return state


def min_cost_noconn(
    percentage: float,
    state: OptimizationState = STATE,
    time_limit: int = 200,
    mipgap: float = 1e-4,
    quiet: bool = True,
) -> tuple[float | None, list[tuple[int, int]] | None, object]:
    """Solve the main no-connectivity minimum-area model."""

    cplex, CplexError = require_cplex()
    patches = state.patches
    if not patches:
        raise ValueError("No patches loaded into optimization state.")

    x_vars = [f"x_{patch_name(patch)}" for patch in patches]
    model = cplex.Cplex()
    model.set_problem_type(cplex.Cplex.problem_type.MILP)
    model.objective.set_sense(model.objective.sense.minimize)
    model.variables.add(
        obj=[state.cost[p] for p in patches],
        lb=[0.0] * len(patches),
        ub=[1.0] * len(patches),
        types="B" * len(patches),
        names=x_vars,
    )
    model.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=x_vars, val=[state.weight[p] for p in patches])],
        senses=["G"],
        rhs=[state.total_biomass * percentage],
    )
    model.parameters.timelimit.set(time_limit)
    model.parameters.mip.tolerances.mipgap.set(mipgap)
    if quiet:
        model.set_results_stream(None)
        model.set_warning_stream(None)

    try:
        model.solve()
    except CplexError:
        return None, None, model

    status = model.solution.get_status()
    if not (100 <= status < 200):
        return None, None, model

    objective = float(model.solution.get_objective_value())
    selected = [
        patch for patch in patches if model.solution.get_values(f"x_{patch_name(patch)}") > 0.5
    ]
    return objective, selected, model


def compute_min_area_curve_for_window(
    full_grid: pd.DataFrame,
    biomass_cols: Sequence[str],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    rhs_total_biomass: float | None = None,
    time_limit: int = 200,
    mipgap: float = 1e-4,
) -> list[dict[str, float | int | bool | None]]:
    """Compute a threshold-to-area curve for one gridded surface."""

    prepare_min_area_state(full_grid, biomass_cols)
    if rhs_total_biomass is not None:
        STATE.total_biomass = float(rhs_total_biomass)

    total_patches = len(STATE.patches)
    max_available = float(sum(STATE.weight.values()))
    results = []
    for threshold in thresholds:
        target = threshold * STATE.total_biomass
        if target > max_available + 1e-9:
            results.append(
                {
                    "threshold": float(threshold),
                    "selected_patches": None,
                    "area_fraction": np.nan,
                    "obj_value": None,
                    "feasible": False,
                }
            )
            continue

        objective, selected, _model = min_cost_noconn(
            threshold, time_limit=time_limit, mipgap=mipgap
        )
        if objective is None or selected is None:
            results.append(
                {
                    "threshold": float(threshold),
                    "selected_patches": None,
                    "area_fraction": np.nan,
                    "obj_value": None,
                    "feasible": False,
                }
            )
            continue

        selected_count = len(selected)
        results.append(
            {
                "threshold": float(threshold),
                "selected_patches": selected_count,
                "area_fraction": selected_count / total_patches if total_patches else np.nan,
                "obj_value": objective,
                "feasible": True,
            }
        )
    return results


def compute_min_area_curves_for_site(
    site_name: str,
    windows: Sequence[str],
    base_dir: str | Path = ".",
    biomass_cols: Sequence[str] = DEFAULT_SPECIES_COLS,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    time_limit: int = 200,
    mipgap: float = 1e-4,
) -> dict[str, list[dict[str, float | int | bool | None]]]:
    """Compute pure-biomass minimum-area curves from saved surface CSVs."""

    site_dir = Path(base_dir) / f"{site_name}_surfaces"
    curves = {}
    for window in windows:
        df = pd.read_csv(site_dir / f"{site_name}_surface_{window}.csv")
        curves[window] = compute_min_area_curve_for_window(
            df,
            biomass_cols=biomass_cols,
            thresholds=thresholds,
            time_limit=time_limit,
            mipgap=mipgap,
        )
    return curves


def compute_pure_total_biomass_dict(
    site_name: str,
    windows: Sequence[str],
    base_dir: str | Path = ".",
    biomass_cols: Sequence[str] = DEFAULT_SPECIES_COLS,
) -> dict[str, float]:
    """Read pure prey surfaces and sum total biomass for each window."""

    site_dir = Path(base_dir) / f"{site_name}_surfaces"
    totals = {}
    for window in windows:
        path = site_dir / f"{site_name}_surface_{window}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        totals[window] = float(df[list(biomass_cols)].sum(axis=1).sum())
    return totals


def compute_min_area_curve_given_df(
    full_grid: pd.DataFrame,
    biomass_cols: Sequence[str],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    rhs_total_biomass: float | None = None,
    time_limit: int = 200,
    mipgap: float = 1e-4,
) -> list[dict[str, float | int | bool | None]]:
    """Wrapper used by ice and ice+ship functional biomass surfaces."""

    return compute_min_area_curve_for_window(
        full_grid=full_grid,
        biomass_cols=biomass_cols,
        thresholds=thresholds,
        rhs_total_biomass=rhs_total_biomass,
        time_limit=time_limit,
        mipgap=mipgap,
    )


def compute_ice_curves_with_pure_target(
    surfaces_dict_ice: Mapping[str, pd.DataFrame],
    pure_totals_dict: Mapping[str, float],
    biomass_cols_beta: Sequence[str] = (
        "polychaeta_avg_final_beta",
        "bivalvia_avg_final_beta",
        "sipuncula_avg_final_beta",
    ),
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    time_limit: int = 200,
    mipgap: float = 1e-4,
) -> dict[str, list[dict[str, float | int | bool | None]]]:
    """Compute ice-adjusted minimum-area curves with pure-biomass RHS targets."""

    curves = {}
    for window, df_ice in surfaces_dict_ice.items():
        if window not in pure_totals_dict:
            continue
        curves[window] = compute_min_area_curve_given_df(
            df_ice,
            biomass_cols=biomass_cols_beta,
            thresholds=thresholds,
            rhs_total_biomass=float(pure_totals_dict[window]),
            time_limit=time_limit,
            mipgap=mipgap,
        )
    return curves


def build_ice_plus_ship_surface(
    df_ice: pd.DataFrame,
    df_stdi: pd.DataFrame,
    alpha: float,
    biomass_cols_ice: Sequence[str] = (
        "polychaeta_avg_final_beta",
        "bivalvia_avg_final_beta",
        "sipuncula_avg_final_beta",
    ),
) -> pd.DataFrame:
    """Apply ship penalty to an already ice-adjusted biomass surface."""

    df = df_ice.copy()
    df = df.merge(
        df_stdi[["grid_x", "grid_y", "STDI_window_mean"]],
        on=["grid_x", "grid_y"],
        how="left",
    )
    df["STDI_window_mean"] = df["STDI_window_mean"].fillna(0.0)
    ship_penalty = 1.0 - alpha * df["STDI_window_mean"]
    for col in biomass_cols_ice:
        df[col] = df[col] * ship_penalty
    return df


def compute_min_area_curves_for_ship_windows(
    site: str,
    ship_windows: Sequence[str],
    base_dir: str | Path,
    beta_surfaces_dict: Mapping[str, pd.DataFrame],
    stdi_dict: Mapping[str, pd.DataFrame],
    alpha: float = 0.5,
    biomass_cols_pure: Sequence[str] = DEFAULT_SPECIES_COLS,
    biomass_cols_ice: Sequence[str] = (
        "polychaeta_avg_final_beta",
        "bivalvia_avg_final_beta",
        "sipuncula_avg_final_beta",
    ),
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> tuple[dict, dict, dict]:
    """Compute pure, ice-only, and ice+ship minimum-area curves for ship windows."""

    curves_pure, curves_ice, curves_ice_ship = {}, {}, {}
    site_dir = Path(base_dir) / f"{site}_surfaces"
    for window in ship_windows:
        df_pure = pd.read_csv(site_dir / f"{site}_surface_{window}.csv")
        pure_total = float(df_pure[list(biomass_cols_pure)].sum(axis=1).sum())

        curves_pure[window] = compute_min_area_curve_given_df(
            df_pure,
            biomass_cols=biomass_cols_pure,
            thresholds=thresholds,
            rhs_total_biomass=pure_total,
        )

        if window not in beta_surfaces_dict or window not in stdi_dict:
            continue
        df_ice = beta_surfaces_dict[window]
        curves_ice[window] = compute_min_area_curve_given_df(
            df_ice,
            biomass_cols=biomass_cols_ice,
            thresholds=thresholds,
            rhs_total_biomass=pure_total,
        )
        df_ice_ship = build_ice_plus_ship_surface(df_ice, stdi_dict[window], alpha)
        curves_ice_ship[window] = compute_min_area_curve_given_df(
            df_ice_ship,
            biomass_cols=biomass_cols_ice,
            thresholds=thresholds,
            rhs_total_biomass=pure_total,
        )
    return curves_pure, curves_ice, curves_ice_ship


def _curve_map(curve: Sequence[Mapping[str, object]], require_feasible: bool = True) -> dict[float, float]:
    out = {}
    for row in curve:
        if require_feasible and ("feasible" in row) and not bool(row["feasible"]):
            continue
        area = row.get("area_fraction")
        if area is None:
            continue
        try:
            area_float = float(area)
        except Exception:
            continue
        if np.isfinite(area_float):
            out[float(row["threshold"])] = area_float
    return out


def compute_area_inflation_decomposition(
    pure_curves: Mapping[str, Sequence[Mapping[str, object]]],
    ice_curves: Mapping[str, Sequence[Mapping[str, object]]],
    ice_ship_curves: Mapping[str, Sequence[Mapping[str, object]]],
    require_feasible: bool = True,
) -> pd.DataFrame:
    """Average differences among pure, ice, and ice+ship curves where data overlap."""

    rows = []
    for window in sorted(set(pure_curves) & set(ice_curves) & set(ice_ship_curves)):
        pure = _curve_map(pure_curves[window], require_feasible=require_feasible)
        ice = _curve_map(ice_curves[window], require_feasible=require_feasible)
        ship = _curve_map(ice_ship_curves[window], require_feasible=require_feasible)

        thr_ice = sorted(set(pure) & set(ice))
        thr_ship = sorted(set(ice) & set(ship))
        thr_total = sorted(set(pure) & set(ship))

        rows.append(
            {
                "window": window,
                "avg_inflation_ice": float(np.mean([ice[t] - pure[t] for t in thr_ice])) if thr_ice else np.nan,
                "n_used_ice": len(thr_ice),
                "avg_inflation_ship_given_ice": float(np.mean([ship[t] - ice[t] for t in thr_ship])) if thr_ship else np.nan,
                "n_used_ship_given_ice": len(thr_ship),
                "avg_inflation_total": float(np.mean([ship[t] - pure[t] for t in thr_total])) if thr_total else np.nan,
                "n_used_total": len(thr_total),
            }
        )
    return pd.DataFrame(rows)


def build_table_s2_shipping_accessibility_loss(
    pure_curves: Mapping[str, Sequence[Mapping[str, object]]],
    ice_curves: Mapping[str, Sequence[Mapping[str, object]]],
    ice_ship_curves: Mapping[str, Sequence[Mapping[str, object]]],
    require_feasible: bool = True,
) -> pd.DataFrame:
    """Appendix Table S2: shipping-induced accessibility loss.

    This reviewer-facing wrapper names the exact table output. It computes the
    average differences among prey-only, prey+ice, and prey+ice+ship
    minimum-area curves at thresholds where paired curves have valid data.
    """

    return compute_area_inflation_decomposition(
        pure_curves=pure_curves,
        ice_curves=ice_curves,
        ice_ship_curves=ice_ship_curves,
        require_feasible=require_feasible,
    )


def generate_full_area_ice_curves(
    full_data_dict: Mapping[str, Mapping[float, Mapping[str, pd.DataFrame]]],
    pure_totals_map: Mapping[str, Mapping[str, float]],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Generate long-form ice-only minimum-area curves for all sites and betas."""

    records = []
    for site, beta_dict in full_data_dict.items():
        pure_totals = pure_totals_map.get(site, {})
        for beta, windows_dict in beta_dict.items():
            for window, df_ice in windows_dict.items():
                if window not in pure_totals:
                    continue
                curve = compute_min_area_curve_given_df(
                    df_ice,
                    biomass_cols=["total_biomass_beta"],
                    thresholds=thresholds,
                    rhs_total_biomass=float(pure_totals[window]),
                    time_limit=100,
                    mipgap=1e-4,
                )
                for point in curve:
                    records.append(
                        {
                            "Site": site,
                            "Beta": beta,
                            "Window": window,
                            "Threshold_Pct": point["threshold"],
                            "Feasible": point["feasible"],
                            "Min_Area_Ratio": point["area_fraction"],
                            "Selected_Patches": point["selected_patches"],
                        }
                    )
    return pd.DataFrame(records)


def normalize_ice_dict(ice_dict: Mapping[str | tuple[int, int], pd.DataFrame]) -> dict[str, pd.DataFrame]:
    normalized = {}
    for key, value in ice_dict.items():
        normalized[f"{key[0]}_{key[1]}" if isinstance(key, tuple) else str(key)] = value
    return normalized


def generate_ice_ship_sensitivity_data(
    site: str,
    ship_windows: Sequence[str],
    base_dir: str | Path,
    ice_dict_full: Mapping[str | tuple[int, int], pd.DataFrame],
    stdi_dict: Mapping[str, pd.DataFrame],
    pure_totals_dict: Mapping[str, float],
    alphas: Sequence[float] = (0.3, 0.5, 0.8),
    betas: Sequence[float] = (0.3, 0.5, 0.8),
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    species_cols: Sequence[str] = DEFAULT_SPECIES_COLS,
) -> pd.DataFrame:
    """Generate ice+ship sensitivity curves for all alpha and beta combinations."""

    ice_dict = normalize_ice_dict(ice_dict_full)
    records = []
    for window in ship_windows:
        df_pure = pd.read_csv(Path(base_dir) / f"{site}_surfaces" / f"{site}_surface_{window}.csv")
        pure_total = float(pure_totals_dict[window])
        df_ice_base = ice_dict[window]
        df_stdi = stdi_dict[window]

        for beta in betas:
            for alpha in alphas:
                df = df_pure.copy()
                df = df.merge(
                    df_ice_base[["y_idx", "x_idx", "A_mean"]],
                    left_on=["grid_y", "grid_x"],
                    right_on=["y_idx", "x_idx"],
                    how="left",
                )
                df["A_mean"] = df["A_mean"].fillna(0.0).clip(0.0, 1.0)
                df["A_beta"] = 1.0 - beta * (1.0 - df["A_mean"])
                df = df.merge(
                    df_stdi[["grid_x", "grid_y", "STDI_window_mean"]],
                    on=["grid_x", "grid_y"],
                    how="left",
                )
                df["STDI_window_mean"] = df["STDI_window_mean"].fillna(0.0)
                ship_penalty = 1.0 - alpha * df["STDI_window_mean"]

                biomass_cols = []
                for col in species_cols:
                    func_col = f"{col}_func"
                    df[func_col] = df[col] * df["A_beta"] * ship_penalty
                    biomass_cols.append(func_col)

                curve = compute_min_area_curve_given_df(
                    df,
                    biomass_cols=biomass_cols,
                    thresholds=thresholds,
                    rhs_total_biomass=pure_total,
                    time_limit=100,
                    mipgap=1e-4,
                )
                for point in curve:
                    records.append(
                        {
                            "Site": site,
                            "Window": window,
                            "Beta": beta,
                            "Alpha": alpha,
                            "Threshold_Pct": point["threshold"],
                            "Min_Area_Ratio": point["area_fraction"],
                            "Feasible": point["feasible"],
                        }
                    )
    return pd.DataFrame(records)


def compute_exact_ship_ceilings(
    site: str,
    ship_windows: Sequence[str],
    base_dir: str | Path,
    ice_dict: Mapping[str | tuple[int, int], pd.DataFrame],
    stdi_dict: Mapping[str, pd.DataFrame],
    pure_totals_dict: Mapping[str, float],
    beta: float = 0.5,
    alpha: float = 0.5,
    species_cols: Sequence[str] = DEFAULT_SPECIES_COLS,
) -> pd.DataFrame:
    """Compute exact functional-biomass ceiling for ice+ship conditions."""

    normalized_ice = normalize_ice_dict(ice_dict)
    rows = []
    for window in ship_windows:
        df_pure = pd.read_csv(Path(base_dir) / f"{site}_surfaces" / f"{site}_surface_{window}.csv")
        df = df_pure.merge(
            normalized_ice[window][["y_idx", "x_idx", "A_mean"]],
            left_on=["grid_y", "grid_x"],
            right_on=["y_idx", "x_idx"],
            how="left",
        )
        df["A_mean"] = df["A_mean"].fillna(0.0).clip(0.0, 1.0)
        df["A_beta"] = 1.0 - beta * (1.0 - df["A_mean"])
        df = df.merge(
            stdi_dict[window][["grid_x", "grid_y", "STDI_window_mean"]],
            on=["grid_x", "grid_y"],
            how="left",
        )
        df["STDI_window_mean"] = df["STDI_window_mean"].fillna(0.0)
        ship_penalty = 1.0 - alpha * df["STDI_window_mean"]
        adjusted = sum((df[col] * df["A_beta"] * ship_penalty).sum() for col in species_cols)
        pure_total = float(pure_totals_dict[window])
        rows.append(
            {
                "Site": site,
                "Window": window,
                "Beta": beta,
                "Alpha": alpha,
                "Exact_Ceiling": adjusted / pure_total if pure_total > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def get_neighbors(patch: tuple[int, int], patches: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    x, y = patch
    patch_set = set(patches)
    candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return [candidate for candidate in candidates if candidate in patch_set]


def solve_min_area_with_adjacency_penalty(
    percentage: float,
    adjacency_beta: float,
    state: OptimizationState = STATE,
    time_limit: int = 300,
    mipgap: float = 1e-4,
    quiet: bool = True,
) -> tuple[int | None, int | None, list[tuple[int, int]] | None, list[str] | None]:
    """Appendix sensitivity variant with adjacency-related y variables."""

    cplex, CplexError = require_cplex()
    patches = state.patches
    x_vars = [f"x_{patch_name(p)}" for p in patches]
    y_vars = []
    neighbor_map = {}
    for patch in patches:
        neighbors = get_neighbors(patch, patches)
        neighbor_map[patch] = neighbors
        y_vars.extend([f"y_{patch_name(patch)}_{patch_name(neighbor)}" for neighbor in neighbors])

    model = cplex.Cplex()
    model.set_problem_type(cplex.Cplex.problem_type.MILP)
    model.objective.set_sense(model.objective.sense.minimize)
    model.variables.add(
        obj=[state.cost[p] for p in patches] + [adjacency_beta] * len(y_vars),
        lb=[0.0] * (len(x_vars) + len(y_vars)),
        ub=[1.0] * (len(x_vars) + len(y_vars)),
        types="B" * (len(x_vars) + len(y_vars)),
        names=x_vars + y_vars,
    )
    model.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=x_vars, val=[state.weight[p] for p in patches])],
        senses=["G"],
        rhs=[state.total_biomass * percentage],
    )

    for patch in patches:
        for neighbor in neighbor_map[patch]:
            y_var = f"y_{patch_name(patch)}_{patch_name(neighbor)}"
            x_i = f"x_{patch_name(patch)}"
            x_j = f"x_{patch_name(neighbor)}"
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[y_var, x_i], val=[1.0, -1.0])],
                senses=["L"],
                rhs=[0.0],
            )
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[y_var, x_j], val=[1.0, -1.0])],
                senses=["L"],
                rhs=[0.0],
            )
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[y_var, x_i, x_j], val=[1.0, -1.0, -1.0])],
                senses=["G"],
                rhs=[-1.0],
            )

    model.parameters.timelimit.set(time_limit)
    model.parameters.mip.tolerances.mipgap.set(mipgap)
    if quiet:
        model.set_results_stream(None)
        model.set_warning_stream(None)

    try:
        model.solve()
    except CplexError:
        return None, None, None, None
    status = model.solution.get_status()
    if not (100 <= status < 200):
        return None, None, None, None

    x_values = model.solution.get_values(x_vars)
    y_values = model.solution.get_values(y_vars)
    selected_x = [patch for idx, patch in enumerate(patches) if x_values[idx] > 0.5]
    selected_y = [var for idx, var in enumerate(y_vars) if y_values[idx] > 0.5]
    return len(selected_x), len(selected_x) + len(selected_y), selected_x, selected_y


def gini(values: Sequence[float]) -> float:
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


def compute_gini_from_patches(
    patches: Sequence[tuple[int, int]],
    full_grid: pd.DataFrame,
    biomass_cols: Sequence[str],
) -> float:
    grid = full_grid.copy()
    grid["grid_x"] = grid["grid_x"].astype(int)
    grid["grid_y"] = grid["grid_y"].astype(int)
    grid["grid_index"] = list(zip(grid["grid_x"], grid["grid_y"]))
    selected = grid[grid["grid_index"].isin(patches)]
    if selected.empty:
        return 0.0
    return gini(selected[list(biomass_cols)].sum(axis=1))


def run_adjacency_sensitivity_for_single_window(
    full_grid: pd.DataFrame,
    thresholds: Sequence[float],
    adjacency_betas: Sequence[float],
    biomass_cols: Sequence[str],
    site_name: str,
    window_label: str,
) -> tuple[pd.DataFrame, int]:
    """Run the SI adjacency-penalty sensitivity for one site-window surface."""

    prepare_min_area_state(full_grid, biomass_cols)
    total_patches = len(STATE.patches)
    rows = []
    for threshold in thresholds:
        for beta in adjacency_betas:
            min_area, conn_area, selected, _ = solve_min_area_with_adjacency_penalty(
                threshold, adjacency_beta=beta
            )
            if min_area is None or selected is None:
                rows.append(
                    {
                        "Site": site_name,
                        "Window": window_label,
                        "Threshold": threshold,
                        "Beta": beta,
                        "MinAreaFraction": np.nan,
                        "ConnAreaFraction": np.nan,
                        "SelectedPatchesCount": np.nan,
                        "Gini": np.nan,
                    }
                )
                continue
            rows.append(
                {
                    "Site": site_name,
                    "Window": window_label,
                    "Threshold": threshold,
                    "Beta": beta,
                    "MinAreaFraction": min_area / total_patches,
                    "ConnAreaFraction": conn_area / total_patches if conn_area is not None else np.nan,
                    "SelectedPatchesCount": min_area,
                    "Gini": compute_gini_from_patches(selected, full_grid, biomass_cols),
                }
            )
    return pd.DataFrame(rows).sort_values(["Threshold", "Beta"]).reset_index(drop=True), total_patches


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="print module purpose and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.describe:
        print("Runs minimum-area CPLEX optimization for pure, ice, and ice+ship biomass surfaces.")
        print("Requires IBM ILOG CPLEX Python API and prepared surface CSV files.")
    else:
        print("Import this module and call compute_min_area_curves_for_site or generate_ice_ship_sensitivity_data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
