"""Build sea-ice accessibility and ice-adjusted prey biomass layers.

The manuscript uses sea-ice concentration and thickness fields to calculate a
cell-level ice accessibility index (A_ice). A_ice is then converted to a soft
biomass penalty:

    A_beta = 1 - beta * (1 - A_mean)

where beta = 0.5 for the main-text analyses and beta = 0.3, 0.5, 0.8 for
sensitivity analyses.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_SPECIES_COLS = (
    "polychaeta_avg_final",
    "bivalvia_avg_final",
    "sipuncula_avg_final",
)


def regrid_ice_to_5km(
    ds,
    origin_x: float,
    origin_y: float,
    cols: int,
    rows: int,
    cell_size: float = 5000.0,
    epsg_from: str = "EPSG:4326",
    epsg_to: str = "EPSG:3338",
    sic_name: str = "siconc",
    sit_name: str = "sithick",
):
    """Aggregate source SIC/SIT fields onto the manuscript 5 km prey grid."""

    import xarray as xr
    from pyproj import Transformer

    transformer = Transformer.from_crs(epsg_from, epsg_to, always_xy=True)
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    x_proj, y_proj = transformer.transform(lon_grid, lat_grid)

    col_idx = ((x_proj - origin_x) / cell_size).astype(int)
    row_idx = ((y_proj - origin_y) / cell_size).astype(int)
    in_grid = (
        (col_idx >= 0)
        & (col_idx < cols)
        & (row_idx >= 0)
        & (row_idx < rows)
    )

    time_n = ds.sizes["time"]
    sic_src = ds[sic_name].values.reshape(time_n, -1)
    sit_src = ds[sit_name].values.reshape(time_n, -1)
    flat_mask = in_grid.ravel()
    flat_rows = row_idx.ravel()
    flat_cols = col_idx.ravel()

    sic_5km = np.zeros((time_n, rows, cols), dtype=float)
    sit_5km = np.zeros((time_n, rows, cols), dtype=float)

    for t_idx in range(time_n):
        rr = flat_rows[flat_mask]
        cc = flat_cols[flat_mask]
        sic_vals = sic_src[t_idx, flat_mask]
        sit_vals = sit_src[t_idx, flat_mask]

        counts = np.zeros((rows, cols), dtype=int)
        sic_sum = np.zeros((rows, cols), dtype=float)
        sit_sum = np.zeros((rows, cols), dtype=float)

        for value, row, col in zip(sic_vals, rr, cc):
            if np.isfinite(value):
                sic_sum[row, col] += value
                counts[row, col] += 1
        for value, row, col in zip(sit_vals, rr, cc):
            if np.isfinite(value):
                sit_sum[row, col] += value

        sic_5km[t_idx] = np.divide(
            sic_sum, counts, out=np.zeros_like(sic_sum), where=counts > 0
        )
        sit_5km[t_idx] = np.divide(
            sit_sum, counts, out=np.zeros_like(sit_sum), where=counts > 0
        )

    return xr.Dataset(
        {
            "sic_5km": (("time", "y", "x"), sic_5km),
            "sit_5km": (("time", "y", "x"), sit_5km),
        },
        coords={"time": ds["time"].values, "y": np.arange(rows), "x": np.arange(cols)},
    )


def subset_ds_by_window_and_months(ds, window: tuple[int, int], months: Sequence[int]):
    """Return dataset rows within a year window and selected calendar months."""

    start_year, end_year = window
    years = ds["time"].dt.year
    month_values = ds["time"].dt.month
    mask = (years >= start_year) & (years <= end_year) & month_values.isin(list(months))
    return ds.sel(time=mask)


def build_ice_index_for_all_windows(
    ice5km_dict: Mapping[tuple[int, int], object],
    cell_size_km: float = 5.0,
    sic_name: str = "sic_5km",
    sit_name: str = "sit_5km",
    h_name: str = "H_local",
    a_name: str = "A_ice",
    d_core_km: float = 15.0,
    d_max_km: float = 50.0,
    dist_land_km: np.ndarray | None = None,
    verbose: bool = False,
) -> dict[tuple[int, int], object]:
    """Calculate local ice suitability and A_ice for all windows."""

    from scipy.ndimage import distance_transform_edt

    results = {}
    for window, ds in ice5km_dict.items():
        if sic_name not in ds.data_vars or sit_name not in ds.data_vars:
            raise KeyError(f"Dataset for {window} lacks {sic_name} or {sit_name}")

        sic = ds[sic_name].values
        sit = ds[sit_name].values
        time_n, rows, cols = sic.shape

        weight_c = np.zeros_like(sic, dtype=float)
        weight_c[(sic >= 0.15) & (sic <= 0.8)] = sic[(sic >= 0.15) & (sic <= 0.8)]
        weight_c[sic > 0.8] = 0.7

        weight_t = np.zeros_like(sit, dtype=float)
        weight_t[sit < 0.2] = 0.1
        weight_t[(sit >= 0.2) & (sit < 0.3)] = 0.5
        weight_t[sit >= 0.3] = 1.0

        h_local = weight_c * weight_t
        a_ice = np.zeros_like(h_local, dtype=float)

        for t_idx in range(time_n):
            h_t = h_local[t_idx]
            good_ice = h_t >= 0.6
            if np.any(good_ice):
                dist_to_ice = distance_transform_edt(~good_ice) * cell_size_km
                dist_union = np.minimum(dist_to_ice, dist_land_km) if dist_land_km is not None else dist_to_ice
            elif dist_land_km is not None:
                dist_union = dist_land_km.copy()
            else:
                a_ice[t_idx] = h_t
                continue

            distance_factor = np.zeros((rows, cols), dtype=float)
            distance_factor[dist_union <= d_core_km] = 1.0
            mid = (dist_union > d_core_km) & (dist_union <= d_max_km)
            distance_factor[mid] = (d_max_km - dist_union[mid]) / (d_max_km - d_core_km)

            candidate = 0.6 * distance_factor
            a_t = np.maximum(h_t, candidate)
            if np.any(good_ice):
                a_t[good_ice] = 1.0
            a_ice[t_idx] = a_t

        ds_out = ds.copy()
        ds_out[h_name] = (("time", "y", "x"), h_local)
        ds_out[a_name] = (("time", "y", "x"), a_ice)
        results[window] = ds_out

        if verbose:
            print(
                f"{window}: H range {np.nanmin(h_local):.3f}-{np.nanmax(h_local):.3f}; "
                f"A range {np.nanmin(a_ice):.3f}-{np.nanmax(a_ice):.3f}"
            )

    return results


def build_ice_surfaces_from_index_dict(
    ice_index_dict: Mapping[tuple[int, int], object],
    sic_name: str = "sic_5km",
    sit_name: str = "sit_5km",
    h_name: str = "H_local",
    a_name: str = "A_ice",
) -> dict[tuple[int, int], pd.DataFrame]:
    """Collapse time-varying ice fields to window-mean grid-cell tables."""

    surfaces: dict[tuple[int, int], pd.DataFrame] = {}
    for window, ds in ice_index_dict.items():
        for name in (sic_name, sit_name, h_name, a_name):
            if name not in ds.data_vars:
                raise KeyError(f"Dataset for {window} missing {name}")

        y_vals = ds["y"].values
        x_vals = ds["x"].values
        yy, xx = np.meshgrid(y_vals, x_vals, indexing="ij")
        surface = pd.DataFrame(
            {
                "y_idx": yy.ravel(),
                "x_idx": xx.ravel(),
                "grid_idx": list(zip(yy.ravel(), xx.ravel())),
                "sic_mean": ds[sic_name].mean(dim="time").values.ravel(),
                "sit_mean": ds[sit_name].mean(dim="time").values.ravel(),
                "H_mean": ds[h_name].mean(dim="time").values.ravel(),
                "A_mean": ds[a_name].mean(dim="time").values.ravel(),
            }
        )
        surfaces[window] = surface.sort_values(["y_idx", "x_idx"]).reset_index(drop=True)
    return surfaces


def make_bbox_from_corners(corners: Sequence[tuple[float, float]], margin_deg: float = 1.0):
    """Return a WGS84 bounding box around DBO corners."""

    from shapely.geometry import box

    lats = np.array([lat for lat, _lon in corners])
    lons = np.array([lon for _lat, lon in corners])
    return box(
        lons.min() - margin_deg,
        lats.min() - margin_deg,
        lons.max() + margin_deg,
        lats.max() + margin_deg,
    )


def build_grid_points(
    origin_x: float,
    origin_y: float,
    cols: int,
    rows: int,
    cell_size_m: float = 5000.0,
    epsg: str = "EPSG:3338",
):
    """Build GeoDataFrame of grid-cell center points."""

    import geopandas as gpd
    from shapely.geometry import Point

    xs = origin_x + (np.arange(cols) + 0.5) * cell_size_m
    ys = origin_y + (np.arange(rows) + 0.5) * cell_size_m
    xx, yy = np.meshgrid(xs, ys)
    return gpd.GeoDataFrame(
        {"col": np.tile(np.arange(cols), rows), "row": np.repeat(np.arange(rows), cols)},
        geometry=[Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())],
        crs=epsg,
    )


def compute_grid_to_land_distance(grid_points_gdf, land_gdf) -> tuple[np.ndarray, object]:
    """Calculate distance from each grid-cell center to nearest land polygon."""

    from shapely.ops import unary_union

    land_union = unary_union(land_gdf.geometry.values)
    points = grid_points_gdf.copy()
    points["dist_land_m"] = points.geometry.apply(lambda geom: geom.distance(land_union))
    dist_grid = (
        points.pivot(index="row", columns="col", values="dist_land_m")
        .sort_index(axis=0)
        .sort_index(axis=1)
        .values
    )
    return dist_grid, points


def build_ice_layers_for_site(
    ds,
    windows: Sequence[tuple[int, int]],
    months: Sequence[int],
    origin_x: float,
    origin_y: float,
    cols: int,
    rows: int,
    cell_size_m: float = 5000.0,
    cell_size_km: float = 5.0,
    dist_land_km: np.ndarray | None = None,
    verbose: bool = False,
) -> tuple[dict[tuple[int, int], object], dict[tuple[int, int], pd.DataFrame]]:
    """Build xarray ice-index layers and tabular A_mean surfaces for one site."""

    ice5km = {}
    for window in windows:
        ds_win = subset_ds_by_window_and_months(ds, window, months)
        if ds_win.sizes.get("time", 0) == 0:
            continue
        ice5km[window] = regrid_ice_to_5km(
            ds_win,
            origin_x=origin_x,
            origin_y=origin_y,
            cols=cols,
            rows=rows,
            cell_size=cell_size_m,
        )

    ice_index = build_ice_index_for_all_windows(
        ice5km,
        cell_size_km=cell_size_km,
        dist_land_km=dist_land_km,
        verbose=verbose,
    )
    return ice_index, build_ice_surfaces_from_index_dict(ice_index)


def add_total_biomass(
    df: pd.DataFrame,
    species_cols: Sequence[str] = DEFAULT_SPECIES_COLS,
) -> pd.DataFrame:
    """Add a total biomass column to a prey-surface DataFrame."""

    missing = [col for col in species_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing species columns: {missing}")
    out = df.copy()
    out["total_biomass"] = out[list(species_cols)].sum(axis=1)
    return out


def normalize_ice_keyed_dict(
    ice_surfaces_dict: Mapping[tuple[int, int] | str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Allow either tuple keys or 'YYYY_YYYY' keys for ice surfaces."""

    normalized = {}
    for key, value in ice_surfaces_dict.items():
        if isinstance(key, tuple):
            normalized[f"{key[0]}_{key[1]}"] = value
        else:
            normalized[str(key)] = value
    return normalized


def load_and_merge_surface(
    site: str,
    window_label: str,
    ice_surfaces_dict: Mapping[tuple[int, int] | str, pd.DataFrame],
    base_dir: str | Path,
    species_cols: Sequence[str] = DEFAULT_SPECIES_COLS,
) -> pd.DataFrame:
    """Read one prey surface and merge its A_mean ice surface."""

    ice_dict = normalize_ice_keyed_dict(ice_surfaces_dict)
    if window_label not in ice_dict:
        raise KeyError(f"No ice surface for {site} window {window_label}")

    surface_path = Path(base_dir) / f"{site}_surface_{window_label}.csv"
    df_biomass = pd.read_csv(surface_path)
    df_biomass = add_total_biomass(df_biomass, species_cols)
    df_ice = ice_dict[window_label]

    merged = df_biomass.merge(
        df_ice[["y_idx", "x_idx", "A_mean"]],
        left_on=["grid_y", "grid_x"],
        right_on=["y_idx", "x_idx"],
        how="left",
    )
    merged["A_mean"] = merged["A_mean"].fillna(0.0).clip(0.0, 1.0)
    return merged


def apply_beta_to_surface(
    df_raw: pd.DataFrame,
    beta: float,
    species_cols: Sequence[str] = DEFAULT_SPECIES_COLS,
) -> pd.DataFrame:
    """Apply the soft ice-accessibility biomass penalty to one surface."""

    df = df_raw.copy()
    df["A_beta"] = np.clip(1.0 - beta * (1.0 - df["A_mean"]), 0.0, 1.0)
    for col in species_cols:
        df[f"{col}_beta"] = df[col] * df["A_beta"]
    df["total_biomass_beta"] = df["total_biomass"] * df["A_beta"]
    return df


def build_biomass_with_ice_beta(
    site: str,
    ice_surfaces_dict: Mapping[tuple[int, int] | str, pd.DataFrame],
    base_dir: str | Path,
    beta: float,
    species_cols: Sequence[str] = DEFAULT_SPECIES_COLS,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build all beta-adjusted biomass surfaces for one site."""

    beta_surfaces: dict[str, pd.DataFrame] = {}
    rows = []
    for window_label in sorted(normalize_ice_keyed_dict(ice_surfaces_dict)):
        merged = load_and_merge_surface(site, window_label, ice_surfaces_dict, base_dir, species_cols)
        beta_df = apply_beta_to_surface(merged, beta, species_cols)
        total_before = float(beta_df["total_biomass"].sum())
        total_after = float(beta_df["total_biomass_beta"].sum())
        beta_surfaces[window_label] = beta_df
        start_year, end_year = map(int, window_label.split("_"))
        rows.append(
            {
                "site": site,
                "window": window_label,
                "start_year": start_year,
                "end_year": end_year,
                "beta": beta,
                "total_biomass": total_before,
                "ice_adjusted_biomass": total_after,
                "mean_biomass_loss": 1.0 - total_after / total_before if total_before > 0 else np.nan,
            }
        )
    return beta_surfaces, pd.DataFrame(rows)


def calculate_single_beta_fraction(df_merged: pd.DataFrame, beta: float) -> dict[str, float]:
    """Calculate total pure and ice-adjusted biomass for one beta value."""

    a_beta = 1.0 - beta * (1.0 - df_merged["A_mean"].to_numpy())
    pure = df_merged["total_biomass"].to_numpy()
    adjusted = pure * a_beta
    total_pure = float(np.sum(pure))
    total_adjusted = float(np.sum(adjusted))
    return {
        "beta": beta,
        "total_pure": total_pure,
        "total_adjusted": total_adjusted,
        "max_attainable_fraction": total_adjusted / total_pure if total_pure > 0 else np.nan,
    }


def run_maf_sensitivity_analysis(
    sites_config: Sequence[dict[str, object]],
    beta_list: Sequence[float] = (0.3, 0.5, 0.8),
) -> pd.DataFrame:
    """Compute maximum attainable fractions for all sites, windows, and betas."""

    records = []
    for config in sites_config:
        site = str(config["name"])
        base_dir = Path(str(config["dir"]))
        ice_dict = normalize_ice_keyed_dict(config["ice_dict"])  # type: ignore[arg-type]
        for window_label in sorted(ice_dict):
            merged = load_and_merge_surface(site, window_label, ice_dict, base_dir)
            start_year, end_year = map(int, window_label.split("_"))
            for beta in beta_list:
                result = calculate_single_beta_fraction(merged, beta)
                records.append(
                    {
                        "Site": site,
                        "Window": window_label,
                        "Start_Year": start_year,
                        "End_Year": end_year,
                        "Beta": beta,
                        "Total_Pure_Biomass": result["total_pure"],
                        "Total_Adjusted_Biomass": result["total_adjusted"],
                        "Max_Attainable_Fraction": result["max_attainable_fraction"],
                    }
                )
    return pd.DataFrame(records)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="print module purpose and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.describe:
        print("Builds sea-ice A_ice layers and beta-adjusted functional biomass surfaces.")
        print("Requires external NetCDF ice products and prey-surface CSV files to run.")
    else:
        print("Import this module and call build_ice_layers_for_site or build_biomass_with_ice_beta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
