# Numbered analysis modules

The numbered files in this folder expose the custom analysis logic used for the
manuscript in a reviewer-readable form.

They are not merely line-number guides. Each file contains executable Python functions for one stage of the workflow:

- `01_prepare_benthic_surfaces.py`: station filtering, 5 km prey grids, kriging, biomass time series, and prey Gini.
- `02_build_ice_accessibility.py`: sea-ice regridding, A_ice, land-distance support, and beta-adjusted prey biomass.
- `03_build_ship_stdi.py`: AIS-to-grid assignment, seasonal STDI, and five-year ship windows.
- `04_run_capacity_optimization.py`: CPLEX minimum-area optimization, pure/ice/ice+ship curves, Table S2 logic, and SI sensitivity variant.
- `05_temperature_auxiliary_analysis.py`: auxiliary temperature regressions, MixedLM, and stratification summaries.
- `06_make_figures_and_tables.py`: Python-generated manuscript figures and Appendix tables from prepared results.

Reviewer-facing output entry points use manuscript labels in their function
names, for example `plot_figure_5_upper_min_area_curves`,
`plot_figure_s7_dbo1_ice_ship_sensitivity_matrix`, and
`build_table_s2_shipping_accessibility_loss`. The complete mapping is in
`../figure_table_code_map.md` and in `MANUSCRIPT_OUTPUT_CODE_INDEX` inside
`06_make_figures_and_tables.py`.

Because this peer-review repository does not include processed input data, the scripts do not claim one-command reproduction of all manuscript results at review stage.

Use `00_reviewer_orientation.py` for a quick overview.
