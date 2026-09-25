"""Reviewer orientation for the walrus foraging-capacity code package."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    ("01_prepare_benthic_surfaces.py", "Benthic prey processing, 5-km grids, kriging surfaces"),
    ("02_build_ice_accessibility.py", "GLORYS12/neXtSIM-F regridding and ice-accessibility index"),
    ("03_build_ship_stdi.py", "AIS processing and Ship Traffic Disturbance Index"),
    ("04_run_capacity_optimization.py", "Minimum-area optimization and sensitivity analyses"),
    ("05_temperature_auxiliary_analysis.py", "Auxiliary temperature regressions and mixed-effects models"),
    ("06_make_figures_and_tables.py", "Manuscript figure and appendix table generation map"),
    ("07_kriging_leave_one_grid_cell_out.py", "Kriging LOOCV and Appendix Table S7"),
    ("08_ice_internal_parameter_surfaces.py", "Six internal ice-accessibility scenarios and Table S5"),
    ("09_ice_internal_parameter_optimization.py", "CPLEX sensitivity curves and Table S6"),
    ("10_verify_reported_results.py", "Data-free consistency checks for revision results"),
]


def main() -> None:
    print("Walrus foraging-capacity code package")
    print(f"Repository root: {ROOT}")
    print()
    print("Modular analysis scripts:")
    for filename, description in MODULES:
        print(f"  - scripts/{filename}: {description}")
    print()
    print("See figure_table_code_map.md for final manuscript figure/table mapping.")
    print("Data files are not included in this peer-review code repository.")
    print("Compact non-coordinate revision summaries are included in results/.")
    print("Figures 1 and 2 were generated outside Python (GIS and PowerPoint).")


if __name__ == "__main__":
    main()
