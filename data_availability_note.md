# Data availability note for peer review

This peer-review repository contains custom analysis code and compact,
non-coordinate summaries of the revision analyses. It does not include raw
data, processed spatial input tables, derived gridded products, station
coordinates, or cell-level cross-validation predictions.

## Data sources cited in the manuscript

- Station-level benthic biomass data: NSF Arctic Data Center / Distributed Biological Observatory data portal. The manuscript cites the dataset-specific temporary DOI for Grebmeier & Cooper (2026).
- Ship-traffic data: public satellite AIS products cited in the manuscript and Appendix S1.
- Sea-ice and oceanographic products: GLORYS12 and neXtSIM-F products cited in the manuscript and Appendix S1.
- Land boundary data: Natural Earth land polygon data used for land-distance calculations.

## Peer-review repository scope

The code repository is provided to allow editors and reviewers to inspect the custom code used to conduct analyses and generate manuscript results, tables, and figures.

The `results/` directory contains the six internal ice-parameter definitions,
the associated focal-interval optimization summaries and curve points, and
aggregate kriging cross-validation statistics reported in Appendix S1: Tables
S5-S7. These files are sufficient to audit the reported summary calculations
without redistributing the processed station-level input data.

The repository does not promise one-command reproduction of all results during peer review because the processed inputs and derived gridded products are not included at this stage.

## Planned final archive

Upon manuscript acceptance, processed input tables, derived gridded prey surfaces, ice-accessibility fields, ship-exposure fields, stressor-adjusted biomass layers, model outputs, and publication-ready code will be archived in a permanent public repository with a persistent identifier.
