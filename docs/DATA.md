# Data and analysis inputs

The paper cohort contains seven datasets from four study families:

| GEO accession | Analysis samples | Reference |
|---|---|---|
| [GSE263333](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE263333) | 5M mouse tissue | Paired RNA |
| [GSE308623](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE308623) | P2S1, P2S2, P2S3 | Paired RNA |
| [GSE205052](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205052) | Mouse brain | Paired RNA |
| [GSE205055](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055) | Human brain | Paired RNA |
| [GSE214986](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE214986) | Author-assembled embryo context | Author-defined domains |

P2S1–P2S3 are grouped within one study family. The mouse and human datasets from
Zhang 2023 are grouped within another. Complete sample counts and analysis scope
are listed in `data/paper/evidence/completion/Fig1b_analysis_cohort.tsv`.

`data/paper/` contains benchmark summaries, gene-level evaluation values and spatial
map coordinates needed for the paper-analysis workflow. Source counts, unavailable
conditions and non-estimable endpoints are retained. The feature-occupancy table's
provenance-only matrix path column is made project-relative; numerical columns
are unchanged. `data/paper/input_checksums.json` records every supplied input.

Original sequencing files and matrices are obtained from the accession records.
Follow the source study's licensing, attribution and data-use conditions. Place
inputs at the relative paths specified by the selected benchmark configuration,
or update that configuration to point to your local data.

The domain/marker/SVG display comparisons use their documented primary seed.
Ranking and study-family transfer use the seed-median domain scores. The marker
correction step recomputes precision and recall from matched clusters; it preserves
the earlier input and writes an explicit correction audit.
