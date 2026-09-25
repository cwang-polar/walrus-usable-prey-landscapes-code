# Verification guide

These checks verify the structure, readability, and committed revision results
in the peer-review code package. Full reproduction from source products requires
data files that are not included in this repository.

## 1. Syntax check

From the repository root, run:

```bash
python -m py_compile scripts/*.py
```

The modular scripts in `scripts/` are the reviewer-facing code files.

## 2. Orientation check

Run:

```bash
python scripts/00_reviewer_orientation.py
```

This prints the analysis modules and notes that Figures 1 and 2 were generated outside Python.

## 3. Figure/table mapping check

Open `figure_table_code_map.md` and confirm that each manuscript output has a
named code entry point. The same mapping is stored as
`MANUSCRIPT_OUTPUT_CODE_INDEX` in `scripts/06_make_figures_and_tables.py`.

Each numbered module also supports a short description check:

```bash
python scripts/01_prepare_benthic_surfaces.py --describe
python scripts/02_build_ice_accessibility.py --describe
python scripts/03_build_ship_stdi.py --describe
python scripts/04_run_capacity_optimization.py --describe
python scripts/05_temperature_auxiliary_analysis.py --describe
python scripts/06_make_figures_and_tables.py --describe
python scripts/07_kriging_leave_one_grid_cell_out.py --describe
python scripts/08_ice_internal_parameter_surfaces.py --describe
python scripts/09_ice_internal_parameter_optimization.py --describe
```

## 4. Revision-result consistency check

Run:

```bash
python scripts/10_verify_reported_results.py
```

The check recalculates minimum-area inflation from all committed CPLEX curve
points, recalculates scenario ranges, checks the number of sensitivity and
cross-validation results, and verifies the two summed-prey validation summaries
reported in Appendix S1. Its final line must be:

```text
ALL CHECKS PASSED
```

## 5. Dependency check

Install the packages listed in `requirements.txt`. IBM ILOG CPLEX must be installed separately according to IBM's license terms.

## 6. Full revision-analysis execution

With the processed input bundle available, set its location and an output
directory before running the revision scripts:

```bash
export WALRUS_INPUT_DIR=/path/to/processed_inputs
export WALRUS_OUTPUT_DIR=/path/to/generated_outputs
python scripts/07_kriging_leave_one_grid_cell_out.py
python scripts/08_ice_internal_parameter_surfaces.py
python scripts/09_ice_internal_parameter_optimization.py
```

Script 09 requires the IBM ILOG CPLEX Python API and must follow script 08.

## 7. Full analysis status

Full reproduction of manuscript figures and tables requires processed input data and external source products. These files are not included in this peer-review repository and will be archived upon manuscript acceptance.
