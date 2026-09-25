# Figure and table to code map

Primary code locations refer to the modular scripts in `scripts/`. Function
names are reviewer-facing entry points for the corresponding manuscript output.

## Main-text figures

| Manuscript output | Analysis/code location | Notes |
| --- | --- | --- |
| Figure 1. Study system and multi-stressor foraging-capacity framework | `scripts/06_make_figures_and_tables.py::external_figure_notes` | Generated in GIS software; Alynne is acknowledged for this component. Python grid definitions are in `scripts/01_prepare_benthic_surfaces.py`. |
| Figure 2. Workflow chart | `scripts/06_make_figures_and_tables.py::external_figure_notes` | Drawn in PowerPoint; not recreated by Python. |
| Figure 3. Sliding-interval prey biomass summaries | `scripts/06_make_figures_and_tables.py::plot_figure_3_sliding_interval_biomass` | Source summaries are produced in `scripts/01_prepare_benthic_surfaces.py`; figure code covers DBO1/DBO4 total biomass trajectories and bootstrap uncertainty. |
| Figure 4. Ice accessibility and ice-adjusted prey landscapes | `scripts/06_make_figures_and_tables.py::plot_figure_4_aice_early_late_maps`; `plot_figure_4_ice_adjusted_biomass_maps`; `summarize_figure_4_ice_accessibility`; `plot_figure_4_accessibility_timeseries` | Ice-accessibility inputs are produced in `scripts/02_build_ice_accessibility.py`. |
| Figure 5. Minimum-area curves and functional ceilings | `scripts/06_make_figures_and_tables.py::plot_figure_5_upper_min_area_curves`; `plot_figure_5_lower_ship_curves` | Optimization curves are produced in `scripts/04_run_capacity_optimization.py`; main model uses `min_cost_noconn`. The upper panels show pure-biomass versus ice-adjusted minimum-area curves; the lower panels show pure-biomass, ice-only, and ice+ship curves for ship-supported intervals. |
| Figure 6. Shipping exposure and overlap with usable prey | `scripts/06_make_figures_and_tables.py::plot_figure_6_ship_exposure_maps`; `plot_figure_6_ice_adjusted_biomass_maps`; `plot_figure_6_stdi_violin_distribution` | STDI layers are produced in `scripts/03_build_ship_stdi.py`; ice-adjusted biomass inputs are produced in `scripts/02_build_ice_accessibility.py`. |

## Main-text tables

| Manuscript output | Analysis/code location | Notes |
| --- | --- | --- |
| Table 1. Focal-interval prey, accessibility, and capacity metrics | `scripts/01_prepare_benthic_surfaces.py::compute_total_biomass_timeseries`; `scripts/02_build_ice_accessibility.py::calculate_single_beta_fraction`; `scripts/04_run_capacity_optimization.py::compute_area_inflation_decomposition`; `compute_exact_ship_ceilings` | Consolidates previously plotted prey, ice-only, and ice+ship metrics for the eight focal intervals. |
| Table 2. Robustness across internal ice-accessibility scenarios | `scripts/09_ice_internal_parameter_optimization.py::main`; `results/ice_internal_parameter_key_results.csv`; `results/ice_internal_parameter_key_window_ranges.csv` | Baseline values and ranges across the six scenarios defined in Table S5. |

## Appendix S1 figures

| Appendix output | Analysis/code location | Notes |
| --- | --- | --- |
| Figure S1. Baseline spatial efficiency | `scripts/06_make_figures_and_tables.py::plot_figure_s1_baseline_spatial_efficiency` | Pure-biomass baseline curves are produced in `scripts/04_run_capacity_optimization.py`. |
| Figure S2. Benthic prey-biomass Gini | `scripts/06_make_figures_and_tables.py::plot_figure_s2_prey_biomass_gini` | Gini source summaries are produced in `scripts/01_prepare_benthic_surfaces.py`. |
| Figure S3. Temperature-biomass relationships, DBO1 | `scripts/05_temperature_auxiliary_analysis.py::plot_figure_s3_dbo1_temperature_biomass` | Auxiliary environmental analysis. |
| Figure S4. Temperature-biomass relationships, DBO4 | `scripts/05_temperature_auxiliary_analysis.py::plot_figure_s4_dbo4_temperature_biomass` | Auxiliary environmental analysis. |
| Figure S5. Annual ship-traffic intensity | `scripts/06_make_figures_and_tables.py::plot_figure_s5_annual_ship_traffic_intensity` | Annual STDI summaries are produced in `scripts/03_build_ship_stdi.py`. |
| Figure S6. Ice-sensitivity analysis | `scripts/06_make_figures_and_tables.py::plot_figure_s6_ice_sensitivity` | Evaluates beta values 0.3, 0.5, and 0.8. SI sensitivity uses an adjacency-penalty variant from `scripts/04_run_capacity_optimization.py`. |
| Figure S7. Ice + ship sensitivity matrix, DBO1 | `scripts/06_make_figures_and_tables.py::plot_figure_s7_dbo1_ice_ship_sensitivity_matrix` | Evaluates beta and alpha combinations for ship-supported intervals. |
| Figure S8. Ice + ship sensitivity matrix, DBO4 | `scripts/06_make_figures_and_tables.py::plot_figure_s8_dbo4_ice_ship_sensitivity_matrix` | DBO4 counterpart to Figure S7. |
| Figure S9. Stratification proxy distribution | `scripts/05_temperature_auxiliary_analysis.py::plot_figure_s9_stratification_proxy` | Auxiliary temperature-structure summary. |

## Appendix S1 tables

| Appendix output | Analysis/code location | Notes |
| --- | --- | --- |
| Table S1. Sensitivity of functional biomass and attainable fraction to beta | `scripts/06_make_figures_and_tables.py::build_table_s1_ice_sensitivity` | Generates total biomass, ice-adjusted biomass, and maximum attainable fraction for beta values. |
| Table S2. Shipping-induced accessibility loss and incremental impacts | `scripts/04_run_capacity_optimization.py::build_table_s2_shipping_accessibility_loss` | Computes average differences among prey-only, prey+ice, and prey+ice+ship curves where paired curves have valid data. |
| Table S3. Ship-traffic statistics | `scripts/03_build_ship_stdi.py::build_table_s3_ship_traffic_statistics` | Summarizes total STDI, mean STDI, and active grid counts. |
| Table S4. Temperature and stratification descriptive statistics | `scripts/05_temperature_auxiliary_analysis.py::build_table_s4_temperature_stratification_stats` | Produces descriptive statistics for temperature variables and stratification proxy. |
| Table S5. Internal ice-accessibility scenarios | `scripts/08_ice_internal_parameter_surfaces.py::SCENARIOS`; `results/ice_internal_parameter_scenario_definitions.csv` | Defines the baseline plus five targeted parameter alternatives. |
| Table S6. Detailed internal ice-accessibility sensitivity results | `scripts/08_ice_internal_parameter_surfaces.py`; `scripts/09_ice_internal_parameter_optimization.py`; `results/ice_internal_parameter_key_results.csv` | Rebuilds accessibility surfaces and reruns the CPLEX minimum-area model at beta = 0.5 for eight focal site-interval combinations. |
| Table S7. Kriging leave-one-grid-cell-out validation | `scripts/07_kriging_leave_one_grid_cell_out.py::build_table_s7_summary`; `results/kriging_loocv_table_s7.csv` | Reports taxon-specific and summed-prey validation statistics across all 17 site-interval combinations. |

## Notes for reviewers

This map follows the final manuscript numbering.
