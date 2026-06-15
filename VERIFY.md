# Verification guide

These checks verify the structure and basic readability of the peer-review code package. Full reproduction of manuscript results requires data files that are not included in this repository.

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
```

## 4. Dependency check

Install the packages listed in `requirements.txt`. IBM ILOG CPLEX must be installed separately according to IBM's license terms.

## 5. Full analysis status

Full reproduction of manuscript figures and tables requires processed input data and external source products. These files are not included in this peer-review repository and will be archived upon manuscript acceptance.
