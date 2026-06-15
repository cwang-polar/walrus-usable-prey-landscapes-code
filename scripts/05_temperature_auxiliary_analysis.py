"""Auxiliary temperature and stratification analyses.

This module supports Appendix S1 Figures S3, S4, S9, and Table S4. It links
station-level prey biomass to summer ocean temperature fields and summarizes a
stratification proxy:

    DeltaT = surface temperature - bottom temperature
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def load_prey_station_data(csv_path: str | Path, site_name: str) -> pd.DataFrame:
    """Load processed station-level prey biomass records for one DBO site."""

    df = pd.read_csv(csv_path)
    required = [
        "GridCellNme",
        "GridCellID",
        "Latitude_Center",
        "Longitude_Center",
        "DataYear",
        "Point_Count_1",
        "sum_gc_Polychaeta_1",
        "sum_gc_Bivalvia_1",
        "sum_gc_Sipuncula_1",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Station table missing required columns: {missing}")

    df["biomass_gc_total"] = (
        df["sum_gc_Polychaeta_1"].fillna(0)
        + df["sum_gc_Bivalvia_1"].fillna(0)
        + df["sum_gc_Sipuncula_1"].fillna(0)
    )
    df["biomass_gc_per_grab"] = df["biomass_gc_total"] / df["Point_Count_1"].replace(0, np.nan)

    station = df[
        [
            "GridCellNme",
            "GridCellID",
            "Latitude_Center",
            "Longitude_Center",
            "DataYear",
            "biomass_gc_total",
            "biomass_gc_per_grab",
            "Point_Count_1",
        ]
    ].copy()
    station["station_year_id"] = station["GridCellNme"].astype(str) + "_" + station["DataYear"].astype(str)
    station["site"] = site_name
    return station.rename(
        columns={
            "Latitude_Center": "lat",
            "Longitude_Center": "lon",
            "DataYear": "year",
        }
    )


def build_bottom_temperature_for_stations(
    station_df: pd.DataFrame,
    ds_phy,
    months: Sequence[int] = (6, 7, 8, 9, 10),
    output_col: str = "bottomT_summer_6_10",
) -> pd.DataFrame:
    """Add nearest-grid mean summer bottom temperature to each station-year row."""

    years = ds_phy["time"].dt.year
    month_values = ds_phy["time"].dt.month

    def one_station(lat: float, lon: float, year: int) -> float:
        mask = (years == year) & month_values.isin(list(months))
        ds_year = ds_phy.sel(time=mask)
        if ds_year.sizes.get("time", 0) == 0:
            return np.nan
        point = ds_year.sel(latitude=lat, longitude=lon, method="nearest")
        values = point["bottomT"].values
        return float(np.nanmean(values)) if not np.all(np.isnan(values)) else np.nan

    out = station_df.copy()
    out[output_col] = [
        one_station(float(row["lat"]), float(row["lon"]), int(row["year"]))
        for _, row in out.iterrows()
    ]
    return out


def add_vertical_temperature_structure(
    df_reg: pd.DataFrame,
    ds_phy,
    months: Sequence[int] = (6, 7, 8, 9, 10),
    max_depth_for_column: float = 50.0,
) -> pd.DataFrame:
    """Add surface, column-mean, and DeltaT temperature variables."""

    from scipy.spatial import cKDTree

    ds_summer = ds_phy.sel(time=ds_phy["time"].dt.month.isin(list(months)))
    surface_field = ds_summer["thetao"].isel(depth=0).mean("time")
    depth_mask = ds_phy["depth"].values <= max_depth_for_column
    column_field = ds_summer["thetao"].isel(depth=np.where(depth_mask)[0]).mean(["time", "depth"])

    lats = ds_phy["latitude"].values
    lons = ds_phy["longitude"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    tree = cKDTree(np.column_stack([lat_grid.ravel(), lon_grid.ravel()]))

    def extract(field_da, lat: float, lon: float) -> float:
        _dist, idx = tree.query([lat, lon])
        iy, ix = np.unravel_index(idx, field_da.shape)
        return float(field_da.values[iy, ix])

    out = df_reg.copy()
    out["T_surface_summer"] = [extract(surface_field, float(r["lat"]), float(r["lon"])) for _, r in out.iterrows()]
    out["T_column_mean_summer"] = [extract(column_field, float(r["lat"]), float(r["lon"])) for _, r in out.iterrows()]
    out["DeltaT_summer"] = out["T_surface_summer"] - out["bottomT_summer_6_10"]
    return out


def prepare_regression_df(
    station_df: pd.DataFrame,
    ds_phy,
    months: Sequence[int] = (6, 7, 8, 9, 10),
) -> pd.DataFrame:
    """Prepare the regression table used for Appendix temperature plots."""

    df_reg = station_df[
        [
            "GridCellID",
            "year",
            "lat",
            "lon",
            "biomass_gc_per_grab",
            "bottomT_summer_6_10",
        ]
    ].dropna(subset=["bottomT_summer_6_10", "biomass_gc_per_grab"]).copy()
    df_reg["StationID"] = df_reg["GridCellID"].astype(str)
    df_reg = add_vertical_temperature_structure(df_reg, ds_phy=ds_phy, months=months)
    df_reg = df_reg.dropna(
        subset=[
            "biomass_gc_per_grab",
            "bottomT_summer_6_10",
            "T_surface_summer",
            "T_column_mean_summer",
            "DeltaT_summer",
        ]
    ).copy()
    df_reg["log_biomass_gc"] = np.log(df_reg["biomass_gc_per_grab"] + 0.1)
    return df_reg


def fit_ols_temperature_model(
    df: pd.DataFrame,
    x_var: str,
    y_var: str = "log_biomass_gc",
):
    """Fit an OLS model used for Appendix temperature-biomass panels."""

    import statsmodels.formula.api as smf

    clean = df.dropna(subset=[x_var, y_var]).copy()
    if clean.empty:
        raise ValueError(f"No rows available for {y_var} ~ {x_var}")
    return smf.ols(f"{y_var} ~ {x_var}", data=clean).fit()


def plot_temp_vs_log_biomass(
    df: pd.DataFrame,
    x_var: str,
    x_label: str,
    site_label: str,
    y_var: str = "log_biomass_gc",
    y_label: str = "log(Biomass per grab + 0.1)",
    savepath: str | Path | None = None,
    ci_level: float = 95,
):
    """Plot scatter, OLS line, confidence band, R2, and p-value."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    model = fit_ols_temperature_model(df, x_var=x_var, y_var=y_var)
    clean = df.dropna(subset=[x_var, y_var]).copy()
    x_grid = np.linspace(clean[x_var].min(), clean[x_var].max(), 100)
    pred = model.get_prediction(pd.DataFrame({x_var: x_grid}))
    ci = pred.conf_int(alpha=1 - ci_level / 100.0)

    plt.figure(figsize=(6, 4.5))
    sns.scatterplot(data=clean, x=x_var, y=y_var, alpha=0.7, edgecolor=None)
    plt.plot(x_grid, pred.predicted_mean, color="red", linewidth=2)
    plt.fill_between(x_grid, ci[:, 0], ci[:, 1], color="red", alpha=0.18, linewidth=0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{site_label}: {x_label}")
    plt.text(
        0.98,
        0.02,
        f"$R^2$ = {model.rsquared:.3f}\n$p$ = {model.pvalues[x_var]:.3f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
    )
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=600, bbox_inches="tight", pad_inches=0.02)
    return model


TEMPERATURE_PANEL_LABELS: Mapping[str, str] = {
    "bottomT_summer_6_10": "Bottom temperature (C)",
    "T_surface_summer": "Surface temperature (C)",
    "T_column_mean_summer": "Column-mean temperature (C)",
    "DeltaT_summer": "DeltaT (surface - bottom, C)",
}


def _plot_temperature_biomass_panel_set(
    df_reg: pd.DataFrame,
    site_label: str,
    output_dir: str | Path | None = None,
    file_prefix: str | None = None,
) -> dict[str, object]:
    """Create the temperature-biomass regression panels for one appendix figure."""

    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    prefix = file_prefix or site_label.lower().replace(" ", "_")

    models = {}
    for x_var, x_label in TEMPERATURE_PANEL_LABELS.items():
        savepath = None if out_dir is None else out_dir / f"{prefix}_{x_var}.pdf"
        models[x_var] = plot_temp_vs_log_biomass(
            df=df_reg,
            x_var=x_var,
            x_label=x_label,
            site_label=site_label,
            savepath=savepath,
        )
    return models


def plot_figure_s3_dbo1_temperature_biomass(
    df_reg1: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Appendix Figure S3: DBO1 temperature-biomass regression panels."""

    return _plot_temperature_biomass_panel_set(
        df_reg=df_reg1,
        site_label="DBO1",
        output_dir=output_dir,
        file_prefix="figure_s3_dbo1",
    )


def plot_figure_s4_dbo4_temperature_biomass(
    df_reg4: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Appendix Figure S4: DBO4 temperature-biomass regression panels."""

    return _plot_temperature_biomass_panel_set(
        df_reg=df_reg4,
        site_label="DBO4",
        output_dir=output_dir,
        file_prefix="figure_s4_dbo4",
    )


def run_lme_bottom_and_delta(df_reg_clean: pd.DataFrame):
    """Mixed-effects model: log biomass ~ bottom temperature + DeltaT + (1|StationID)."""

    from statsmodels.regression.mixed_linear_model import MixedLM

    model = MixedLM.from_formula(
        "log_biomass_gc ~ bottomT_summer_6_10 + DeltaT_summer",
        groups="StationID",
        data=df_reg_clean,
    )
    return model.fit()


def summarize_stratification(
    df_reg1: pd.DataFrame,
    df_reg4: pd.DataFrame,
    label1: str = "DBO1 (1993-2020)",
    label4: str = "DBO4 (2007-2020)",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return combined DeltaT data and descriptive statistics for Table S4."""

    cols = ["DeltaT_summer", "T_surface_summer", "bottomT_summer_6_10"]
    d1 = df_reg1.copy()
    d4 = df_reg4.copy()
    d1["Region"] = label1
    d4["Region"] = label4
    combined = pd.concat([d1[["Region", *cols]], d4[["Region", *cols]]], ignore_index=True)
    summary = combined.groupby("Region")[cols].agg(["mean", "std", "min", "max"])
    summary = summary.rename(
        columns={"mean": "Mean", "std": "SD", "min": "Min", "max": "Max"},
        level=1,
    ).round(2)
    return combined, summary


def build_table_s4_temperature_stratification_stats(
    df_reg1: pd.DataFrame,
    df_reg4: pd.DataFrame,
    label1: str = "DBO1 (1993-2020)",
    label4: str = "DBO4 (2007-2020)",
) -> pd.DataFrame:
    """Appendix Table S4: temperature and stratification descriptive statistics."""

    _combined, summary = summarize_stratification(
        df_reg1=df_reg1,
        df_reg4=df_reg4,
        label1=label1,
        label4=label4,
    )
    return summary


def plot_stratification_boxplot(
    combined: pd.DataFrame,
    savepath: str | Path | None = None,
):
    """Plot Appendix S9 stratification proxy distribution."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=combined, x="Region", y="DeltaT_summer", palette="Set2", ax=ax)
    ax.set_ylabel("Stratification proxy (DeltaT = surface - bottom)")
    ax.set_xlabel("")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=600, bbox_inches="tight")
    return fig, ax


def plot_figure_s9_stratification_proxy(
    combined: pd.DataFrame,
    savepath: str | Path | None = None,
):
    """Appendix Figure S9: stratification proxy distribution."""

    return plot_stratification_boxplot(combined=combined, savepath=savepath)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="print module purpose and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.describe:
        print("Builds auxiliary temperature regressions, mixed models, and stratification summaries.")
        print("Requires processed station data and CMEMS/GLORYS temperature NetCDF files.")
    else:
        print("Import this module and call prepare_regression_df, plot_temp_vs_log_biomass, or summarize_stratification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
