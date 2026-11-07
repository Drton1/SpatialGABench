# SpatialGABench

SpatialGABench benchmarks gene activity construction from spatial chromatin
accessibility. It separates the contributions of genomic input representation,
feature-to-gene assignment and downstream evaluation task.

This repository provides the benchmark implementation and the reproducible
analyses accompanying **SpatialGABench dissects representation and depth effects
in spatial ATAC gene activity**.

## What the benchmark evaluates

- Peak and fixed-bin representations across retained sequencing depths.
- Promoter, gene-body and distal assignment rules, with matched mapping controls.
- RNA concordance and effective gene coverage.
- Spatial domains, matched-cluster markers, spatially variable genes and held-out RNA programs.
- Frozen-peak recovery, task-choice losses and retrospective study-family transfer.

## Reproduce the paper analyses

Use Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduce.txt
python scripts/reproduce_paper.py --config configs/paper_reproduction.json
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The command recomputes eight analysis tables, renders the 24 result panels in
Figures 2–5, and compares their numerical values with the supplied references.
Outputs and a machine-readable verification report are written to `results/paper/`.
This workflow has passed on Linux and Windows; see [the verification record](docs/REPRODUCTION.md).
The included source tables are sufficient for this workflow. Full sequencing
inputs are obtained separately from GEO; see [the data guide](docs/DATA.md).

For analysis tables only:

```bash
python scripts/reproduce_paper.py --config configs/paper_reproduction.json --skip-figures
```

Plots use Arial when installed, then Liberation Sans, then DejaVu Sans. Set
`SPATIALGABENCH_FONT` to choose a local font. Numerical verification is independent
of font rendering. Figure 5E labels express absolute losses in units of 0.001;
the source tables retain unscaled values.

## Repository layout

| Directory | Contents |
|---|---|
| `spatialgabench/` | Gene activity, representations, evaluation, simulation and command-line implementation |
| `configs/` | Benchmark configuration snapshots and paper-analysis settings |
| `registries/` | Dataset, method and task metadata |
| `scripts/` | Paper analysis, plotting and reproduction entry points |
| `data/paper/` | Source tables and spatial-map coordinates used by the result figures |
| `reference/` | Expected analysis tables and plotted numerical matrices |
| `docs/` | Data access, workflow and provenance documentation |

To install the full benchmark package and its matrix-analysis dependencies:

```bash
python -m pip install -e .
python -m spatialgabench.cli.main --help
```

Raw-data runs use the registered inputs and configuration files described in
[the implementation guide](docs/IMPLEMENTATION.md). The table-reproduction command
above is the entry point for regenerating the paper's numerical comparisons.

## Citation and license

Software citation metadata are provided in [CITATION.cff](CITATION.cff).
The project code is distributed under the [MIT License](LICENSE). Dependencies
and original datasets retain their respective licenses and data-use terms.
