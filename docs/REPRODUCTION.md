# Verified reproduction

The public source-table workflow was executed on Windows and Linux on
7 November 2026. Both runs passed the following checks:

| Check | Result |
|---|---|
| Input integrity | 20 input files matched their SHA-256 hashes |
| Recomputed analyses | All 8 TSV tables matched the reference values |
| Plotted numerical matrices | All 12 matrices matched the reference values |
| Figure output | Figures 2–5 rendered to PDF, SVG and PNG |

Numerical comparisons use an absolute and relative tolerance of `1e-12`, with
matching missing values. Table headers, row order and dimensions are checked.
The plotted-matrix checks cover the recorded heatmap and agreement matrices;
the remaining panels are rendered from the checksum-verified source tables.

| Environment | Python | NumPy | Matplotlib |
|---|---|---|---|
| Linux | 3.11.15 | 2.4.6 | 3.11.0 |
| Windows | 3.14.3 | 2.5.4 | 3.11.2 |

`requirements-reproduce.txt` specifies the tested Linux analysis versions.
Each reproduction run writes its own detailed report and logs to `results/paper/`.
Figure bytes can vary with fonts and rendering libraries; numerical values are
checked independently of PDF or image hashes.

The full benchmark command-line entry point and its self-contained toy benchmark
also completed in the Linux scientific environment:

```bash
python -m spatialgabench.cli.main --help
python -m spatialgabench.cli.main run-toy-example --outdir results/toy_smoke
```

These checks validate reanalysis of the supplied paper tables, figure generation
and the toy execution path. They do not represent a new run of the full raw-data
benchmark. Dataset accessions and the matrix-level workflow are described in
[DATA.md](DATA.md) and [IMPLEMENTATION.md](IMPLEMENTATION.md).
