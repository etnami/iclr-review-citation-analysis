# ICLR Peer Review Patterns and Citation Impact: A PySpark Big Data Analysis

MSc Data Science coursework (University of Sheffield), Grade: 79 (Distinction). Analyzes 55,906 ICLR paper submissions (2017-2026) using PySpark on Azure Databricks: acceptance patterns, whether reviewer scores predict acceptance, keyword trends, research community clustering, and whether review scores predict real-world citation impact by joining against OpenAlex.

> **Reproducibility note:** this notebook was run on Azure Databricks against two source files (an ICLR submissions parquet file and an OpenAlex JSON export) that aren't included here, so it can't be rerun end-to-end from this repo. All figures below come from the notebook's stored cell outputs from the original run.

## Key results

**Q1, acceptance/rejection/withdrawal (Pending submissions with no decision yet excluded):**

| Outcome | Papers |
|---|---|
| Accepted | 11,349 |
| Rejected | 17,290 |
| Withdrawn | 7,594 |
| (Pending, excluded from rate) | 19,673 |

Overall acceptance rate (accepted / (accepted + rejected)): **39.6%**. By year, the rate fell from 46.3% (2018) to a low of 30.9% (2020) as submission volume grew, then stabilized around 40% from 2022 onward (39.0%-42.5%).

**Q2, do reviewer scores predict acceptance?** 1,596 accepted papers (14.1% of all accepted) had at least one reviewer score under 5, meaning a paper can succeed despite one skeptical reviewer.

A logistic regression (mean score, score standard deviation, number of reviewers → accept/reject) predicts acceptance with **87.1% accuracy** and an **AUC of 0.9418**. Mean score is by far the dominant coefficient (3.225); score spread and reviewer count matter far less. 413 accepted-but-flagged-as-rejected cases in the test set (413 false negatives) are the main source of error.

**Q3, top-scoring paper and score/acceptance relationship:** the single highest-scoring paper (`u1cQYxRI1H`, mean score 10.0 from 4 reviewers, 2025, Accept Oral) was manually cross-checked against the raw data and confirmed. Accepted papers average a mean score of 6.494 (SD 1.086) vs. 4.676 (SD 1.222) for rejected papers. Acceptance rate climbs sharply by score band: <4 → 0.2%, 4-5 → 1.8%, 5-6 → 21.3%, 6-7 → 80.7%, 7+ → 98.5%.

**Q4, explosively growing keywords:** only 4 keywords went from under 5 mentions to over 50 the following year: `test-time scaling` (3 → 87, 2025 → 2026), `llm` (4 → 76, 2023 → 2024), `large reasoning models` (1 → 57, 2025 → 2026), `spatial reasoning` (3 → 57, 2025 → 2026).

TF-IDF + k-means clustering of papers by keyword profile (k=3, chosen by silhouette score: 0.0581 for k=3 vs. 0.0544 for k=4 and negative for k=5-7) found three research communities: a large general ML/RL/LLM cluster (47,102 papers), an adversarial robustness cluster (3,525 papers, 37.5% acceptance), and a reasoning/LLM cluster (3,148 papers, **44.2% acceptance**, the highest of the three and the fastest-growing).

**Q5, matching ICLR papers to OpenAlex citation data:** two different join strategies give two different counts, and the notebook's own write-up and its own sanity checks disagree on which to treat as final. See Verification note below.

- Basic join, raw titles, no dedup: **1,593 matched papers** (2.9% of all 55,906 ICLR papers)
- Normalized (lowercased/trimmed) + deduplicated join: **1,602 matched papers**

Either way, ~70.9% of OpenAlex's own title list matches something in the ICLR data. Match counts drop off sharply after 2021, consistent with newer papers not yet being fully indexed by OpenAlex rather than a data quality problem.

**Citation analysis (Q5 extension):** rejected papers actually average *more* citations (113.0) than accepted papers (78.2), though medians are much closer (25 vs. 22), suggesting the average is skewed by a handful of highly-cited outliers rather than a systematic pattern. The correlation between a paper's mean reviewer score and its eventual citation count is essentially zero (r = 0.0453).

**Q6, low-scored but highly-cited papers:** same dual-answer situation as Q5. See Verification note.

- Using OpenAlex's own pre-computed citation percentile field, joined on raw titles: **13 papers**
- Using a locally-computed 95th percentile threshold (≥ 284 citations) on the deduplicated, normalized join: **8 papers**

Only 4 papers appear on both lists. Notable examples either way include "Visualizing the Loss Landscape of Neural Nets" (mean score 4.67, several hundred citations) and "GradNorm" (mean score 4.67, 313-444 citations depending on which citation count is used).

A linear regression predicting citation count from review features (mean score, score spread, reviewer count, paper age) explains only **5.5% of citation variance** (R² = 0.0545, RMSE = 169.50). Paper age is by far the strongest predictor (coefficient -57.97, i.e. older papers have accumulated more citations), not review quality.

**The headline finding the coursework is built around:** reviewer scores predict *acceptance* very well (87.1% accuracy) but predict *citation impact* very poorly (review features plus paper age explain 5.5% of variance in the ~1,600 papers matched to OpenAlex). The review process is internally consistent, agreeing with itself about what to accept, but that consistency doesn't carry over to identifying the work the field will end up valuing most.

## Methods & tools

- **Platform:** PySpark on Databricks
- **Data:** an ICLR submissions dataset (55,906 papers, 2017-2026: decision, reviewer scores, keywords) joined against an OpenAlex export (2,245 records: citation counts and percentiles)
- **Q1-Q6 core analysis:** `groupBy`/`pivot` aggregations, `explode` for array columns (scores, keywords), self-joins for year-over-year keyword growth, title-based joins (raw and normalized) against OpenAlex
- **ML extensions:** logistic regression (`pyspark.ml.classification`) for acceptance prediction, k-means clustering with TF-IDF features (`HashingTF` + `IDF`) for research-community detection, linear regression (`pyspark.ml.regression`) for citation prediction
- **Model evaluation:** AUC-ROC and confusion matrices (classification), silhouette score (clustering), R²/RMSE (regression)

## Repo structure

```
iclr-review-citation-analysis/
├── README.md                        ← you are here
└── iclr_bigdata_analysis.ipynb      ← full notebook, including stored outputs from an actual run
```

## How to run

1. Requires a Spark environment (this was run on Databricks) with `pyspark.ml` available.
2. Two source files are expected at Databricks Volume paths (`iclr26v1.parquet` and `openalex.json`, referenced near the top of the notebook) that aren't included in this repo; see Data, below.
3. The notebook follows Q1-Q6, and each question also has an 'EXTENSION' cell that goes further.

## Data

The ICLR submissions dataset and the OpenAlex export are not redistributed here. The notebook's own outputs (visible directly in the `.ipynb` file) are the record of what running it actually produced, standing in for a live rerun.

## Limitations

- Q5 and Q6 each have two different, non-trivial implementations in the notebook that produce different answers (1,593 vs. 1,602 for Q5; 13 vs. 8 for Q6). Per a deliberate choice made when building this repo, both are presented side by side above rather than picking one as "the" answer, since the underlying methodological choices (which titles get treated as duplicates, whether to trust OpenAlex's own percentile field or recompute one locally) are each reasonable and the coursework's own write-up and its own sanity-check cells don't agree with each other on which to prefer. In practice I would choose the deduplicated join as the more defensible method and treat the other as a sensitivity check, but both were required to be reported for the project and the deduplication was treated as an extension.
- The clustering (Q4 extension) has a fairly weak silhouette score even at its best (k=3, 0.0581), meaning the three "research community" clusters are real but not sharply separated, consistent with ML/AI research topics overlapping heavily rather than falling into clean categories.
- The citation-prediction linear regression (Q6 extension) explains very little variance (R² = 0.0545); this is reported as a substantive finding (review scores are a poor predictor of citation impact), not a modeling failure to be fixed.
- All numbers here come from reading the notebook's own stored outputs from a real prior run, not from independently re-executing the code, since the underlying data and a Spark environment weren't available to verify from scratch.

## Verification note

Q5 and Q6 each have two implementations in the notebook that give different answers (1,593 vs 1,602 matched papers; 13 vs 8 low-scored, highly-cited papers), and the coursework write-up and the notebook's own sanity checks disagree on which to treat as final. Both are reported above. I'd treat the normalised, deduplicated join as the more defensible method and the other as a sensitivity check. All other figures had a single implementation and matched the write-up.
