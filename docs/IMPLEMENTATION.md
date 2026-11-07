# Implementation and reproduction

## Source-table workflow

1. `scripts/selection_analysis.py` reads the paper selection configuration, recomputes
   matched-cluster marker precision/recall, endpoint agreement, cross-task losses
   and leave-study-family-out summaries.
2. `scripts/plot_paper_figures.py` renders Figures 2–5 from these summaries and the
   supplied depth, control and spatial-readout tables.
3. `scripts/reproduce_paper.py` verifies the input hashes, compares eight output tables
   and all plotted numerical matrices with `reference/`, and records the runtime
   environment and verification results.

Selection ties are averaged. Range-normalized losses are undefined when the
target's candidate range is zero. Study-family transfer keeps related datasets
together; it is a retrospective analysis of the recorded outcomes.

## Matrix-level implementation

| Component | Source |
|---|---|
| Gene-assignment rules and matched controls | `spatialgabench/gene_activity/` |
| Fragment filtering, peak/bin counting | `spatialgabench/representations/` |
| Paired-depth RNA evaluation | `spatialgabench/nm_paired_depth_accuracy.py` |
| Frozen-peak recovery | `spatialgabench/nm_frozen_peak_rescue.py` |
| Spatial programs | `spatialgabench/nm_upgrade_completion.py` |
| Spatial domains, markers and Moran statistics | `spatialgabench/nm_ds1_downstream.py`, `nm_ds3_downstream.py` |
| Mechanistic and empirical simulations | `spatialgabench/simulation/` |
| Endpoint rankings and study-family effects | `spatialgabench/nm_xend1.py`, `nm_rank2.py`, `nm_fam1.py` |

The implementation snapshot retains the development modules and their configuration
files. A matrix-level run requires the corresponding data, annotations and external
tools. The source-table workflow is self-contained for numerical reproduction of
the paper analyses. The release verification record distinguishes these scopes.

## Provenance

`docs/source_manifest.json` maps exported implementation and analysis inputs to
their source-snapshot entries and hashes. The public layout changes only paths and
font discovery in the plotting entry point. The analysis formulas and numerical
reference values are retained. Internal transport scripts, environments, account
credentials and manuscript files are outside this repository.
