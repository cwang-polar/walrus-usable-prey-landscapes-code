# Walrus foraging-capacity analysis code

This repository provides the custom Python analysis code for the manuscript:

**Hidden loss of foraging capacity: Sea ice loss and Arctic shipping reshape usable Pacific walrus prey landscapes**

Manuscript submitted to *Ecological Applications* as EAP26-0464.

Corresponding author: Chao Wang, Chesapeake Biological Laboratory, University of Maryland Center for Environmental Science.  
Email: cwang@umces.edu

## Purpose of this repository

This is a peer-review code repository prepared to satisfy the ESA Open Research requirement that novel code be available during review. It contains refactored Python modules exposing the custom code used to prepare benthic prey surfaces, construct sea-ice and ship-disturbance accessibility layers, run minimum-area optimization, perform sensitivity analyses, and generate manuscript figures and appendix tables.

Raw observations and processed spatial input files are not included in this peer-review repository. Original public data sources are cited in the manuscript and Appendix S1. Compact, non-coordinate result summaries supporting the revision analyses are included in `results/`. Processed input tables, derived gridded products, and complete model outputs will be archived in a permanent repository upon manuscript acceptance.

## Repository contents

```text
.
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
|-- VERIFY.md
|-- data_availability_note.md
|-- figure_table_code_map.md
|-- scripts/
|   |-- 00_reviewer_orientation.py
|   |-- 01_prepare_benthic_surfaces.py
|   |-- 02_build_ice_accessibility.py
|   |-- 03_build_ship_stdi.py
|   |-- 04_run_capacity_optimization.py
|   |-- 05_temperature_auxiliary_analysis.py
|   |-- 06_make_figures_and_tables.py
|   |-- 07_kriging_leave_one_grid_cell_out.py
|   |-- 08_ice_internal_parameter_surfaces.py
|   |-- 09_ice_internal_parameter_optimization.py
|   |-- 10_verify_reported_results.py
|   `-- revision_support.py
`-- results/
    |-- README.md
    |-- ice_internal_parameter_scenario_definitions.csv
    |-- ice_internal_parameter_key_results.csv
    |-- ice_internal_parameter_key_window_ranges.csv
    |-- ice_internal_parameter_cplex_curve_points.csv
    |-- kriging_loocv_taxon_summary.csv
    `-- kriging_loocv_table_s7.csv
```

## Which code files should reviewers read?

The primary reviewer-facing code is in:

```text
scripts/
```

The numbered scripts are modular analysis files, not just pointers. They contain
the reviewer-facing implementation for each analysis stage.

## Major analysis components

- Benthic prey station processing and 5-km gridded prey surfaces
- Taxon-specific ordinary kriging using PyKrige
- Sea-ice accessibility surfaces from GLORYS12 and neXtSIM-F products
- Ship Traffic Disturbance Index (STDI) from public s-AIS data
- Minimum-area optimization using IBM ILOG CPLEX through the Python API
- Sensitivity analyses for ice and ship penalty parameters
- Auxiliary water-temperature regressions and mixed-effects models
- Leave-one-grid-cell-out validation of taxon-specific and summed prey surfaces
- Six targeted internal ice-accessibility parameter scenarios
- Main-text and Appendix S1 figure/table generation

Figures 1 and 2 are exceptions: Figure 1 was prepared in GIS software, and Figure 2 was drawn in PowerPoint. The Python code documents these outputs but does not recreate them.

For Figure 3, interval-level uncertainty was calculated from 200 bootstrap
resamples of station observations within each analysis interval, using random
seed 123. Taxon-specific prey surfaces were reconstructed for each resample,
and the 2.5th and 97.5th percentiles of integrated total prey biomass define
the plotted 95% bootstrap confidence intervals.

## Dependencies

The analysis was implemented in Python 3.10. Main Python packages are listed in `requirements.txt`.

The minimum-area optimization uses IBM ILOG CPLEX through the Python API. CPLEX is commercial/proprietary software and must be installed separately according to IBM's license terms.

## Data status

The peer-review code repository does not include input data files. It does include compact revision-result summaries that contain no station coordinates or cell-level observations. See `data_availability_note.md` for the data-access plan and `figure_table_code_map.md` for how analysis code maps to manuscript outputs.

## Figure and table index

Every manuscript figure and table is mapped to a reviewer-facing code entry
point in `figure_table_code_map.md`. The same output index is also stored as
`MANUSCRIPT_OUTPUT_CODE_INDEX` in `scripts/06_make_figures_and_tables.py`.
Figures 1 and 2 are explicitly marked as non-Python outputs.

## Revision analyses

The analyses added in response to peer review are exposed directly in:

```text
scripts/07_kriging_leave_one_grid_cell_out.py
scripts/08_ice_internal_parameter_surfaces.py
scripts/09_ice_internal_parameter_optimization.py
```

The ice-scenario workflow must be run in numerical order because script 09 uses
the scenario surfaces produced by script 08. The scripts read processed inputs
from `data/` and write generated files to `outputs/` by default. Alternative
locations can be supplied with the `WALRUS_INPUT_DIR` and
`WALRUS_OUTPUT_DIR` environment variables.

## Verification

See `VERIFY.md` for checks that can be run on the repository code package. In
particular, `python scripts/10_verify_reported_results.py` independently checks
the internal consistency of the committed revision summaries. Full reproduction
from source products requires processed input data and external source products
that are not included in this peer-review repository.
