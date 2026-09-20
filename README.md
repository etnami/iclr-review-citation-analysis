# ICLR Peer Review Patterns and Citation Impact: A PySpark Big Data Analysis

MSc Data Science coursework (University of Sheffield), Grade: 79 (Distinction). Analyses 55,906 ICLR paper submissions (2017-2026) using PySpark on Databricks serverless: acceptance patterns, whether reviewer scores predict acceptance, keyword trends, keyword-based clustering, and whether review scores predict real-world citation impact by joining against OpenAlex.

> **Reproducibility note:** this notebook was run on Databricks against two source files (an ICLR submissions parquet file and an OpenAlex JSON export) that aren't included here, so it can't be rerun end-to-end from this repo. All figures below come from the notebook's stored cell outputs from the original run.

## Key results

**Q1, acceptance/rejection/withdrawal (Pending submissions with no decision yet excluded):**

| Outcome | Papers |
|---|---|
| Accepted | 11,349 |
| Rejected | 17,290 |
| Withdrawn | 7,594 |
| (Pending, excluded from rate) | 19,673 |

Overall acceptance rate (accepted / (accepted + rejected)): **39.6%**. By year, the rate fell from 46.3% (2018) to a low of 30.9% (2020) as submission volume grew, then stabilised around 40% from 2022 onward (39.0%-42.5%).

**Q2, do reviewer scores predict acceptance?** 1,596 accepted papers (14.1% of all accepted) had at least one reviewer score under 5, meaning a paper can succeed despite one sceptical reviewer.

A logistic regression (mean score, score standard deviation, number of reviewers → accept/reject), trained on the 28,476 accepted or rejected papers with reviewer scores (80/20 split, seed 42), predicts acceptance on the 5,668-paper test set with **87.1% accuracy** and an **AUC of 0.9418**; always predicting "rejected" would score 59.2%. Mean score carries the largest weight (coefficient 3.225, against -0.057 for score spread and -0.153 for reviewer count; features unstandardised). The main source of error is 413 accepted papers predicted as rejected (false negatives).

**Q3, top-scoring paper and score/acceptance relationship:** the single highest-scoring paper (`u1cQYxRI1H`, mean score 10.0 from 4 reviewers, 2025, Accept Oral) was manually cross-checked against the raw data and confirmed. Accepted papers average a mean score of 6.494 (SD 1.086) vs. 4.676 (SD 1.222) for rejected papers. Acceptance rate climbs sharply by score band: <4 → 0.2%, 4-5 → 1.8%, 5-6 → 21.3%, 6-7 → 80.7%, 7+ → 98.5%.

**Q4, explosively growing keywords:** only 4 keywords went from under 5 mentions to over 50 the following year: `test-time scaling` (3 → 87, 2025 → 2026), `llm` (4 → 76, 2023 → 2024), `large reasoning models` (1 → 57, 2025 → 2026), `spatial reasoning` (3 → 57, 2025 → 2026).

TF-IDF + k-means clustering of papers by keyword profile (k=3, chosen by silhouette score: 0.0581 for k=3 vs. 0.0544 for k=4 and negative for k=5-7) found three approximate topic groups: a large general ML/RL/LLM cluster 47,102 papers; 53,775 papers have keywords in total), an adversarial robustness cluster (3,525 papers, 37.5% acceptance), and a reasoning/LLM cluster (3,148 papers, **44.2% acceptance**, the highest of the three and the fastest-growing).

**Q5, matching ICLR papers to OpenAlex citation data:** two different join strategies give two different counts, and the notebook's own write-up and its own sanity checks disagree on which to treat as final. See Verification note below.

- Basic join, raw titles, no dedup: **1,593 matched papers** (2.9% of all 55,906 ICLR papers)
- Normalised (lowercased/trimmed) + deduplicated join: **1,602 matched papers**

Either way, ~70.9% of OpenAlex's own title list matches something in the ICLR data. Match counts drop off sharply after 2021, consistent with newer papers not yet being fully indexed by OpenAlex rather than a data quality problem.

**Citation analysis (Q5 extension):** rejected papers actually average *more* citations (113.0, n = 241) than accepted papers (78.2, n = 1,315), though medians are much closer (25 vs. 22), suggesting the average is skewed by a handful of highly-cited outliers rather than a systematic pattern. The correlation between a paper's mean reviewer score and its eventual citation count is essentially zero (r = 0.0453).

**Q6, low-scored but highly-cited papers:** same dual-answer situation as Q5. See Verification note.

- Using OpenAlex's own pre-computed citation percentile field, joined on raw titles: **13 papers**
- Using a locally-computed 95th percentile threshold (≥ 284 citations) on the deduplicated, normalised join: **8 papers**

Only 4 papers appear on both lists. Notable examples either way include "Visualising the Loss Landscape of Neural Nets" (mean score 4.67, several hundred citations) and "GradNorm" (mean score 4.67, 313-444 citations depending on which citation count is used).

A linear regression predicting citation count from review features (mean score, score spread, reviewer count) plus submission year (years since 2017), fitted on 1,589 matched papers with an 80/20 split, explains only **5.5% of citation variance** on the test set (R² = 0.0545, RMSE = 169.50). Submission year has the largest coefficient (-57.97: each later year goes with about 58 fewer citations, because newer papers have had less time to be cited), while each extra point of mean score goes with only about 8 more citations (features are unstandardised, so coefficient sizes are indicative).

**Headline finding:** reviewer scores predict *acceptance* well (87.1% accuracy) but predict *citation impact* poorly (review features plus submission year explain 5.5% of variance in the 1,589 papers matched to OpenAlex). The review process is internally consistent about what to accept, but in this sample that consistency does not reliably identify the work that goes on to be most cited. Citations are only one measure of value, and the matched papers are a small, likely more-cited-than-average subset (see Limitations).

## Methods & tools

- **Platform:** PySpark on Databricks
- **Data:** an ICLR submissions dataset (55,906 papers, 2017-2026: decision, reviewer scores, keywords) joined against an OpenAlex export (2,245 records: citation counts and percentiles)
- **Q1-Q6 core analysis:** `groupBy`/`pivot` aggregations, `explode` for array columns (scores, keywords), self-joins for year-over-year keyword growth, title-based joins (raw and normalised) against OpenAlex
- **ML extensions:** logistic regression (`pyspark.ml.classification`) for acceptance prediction, k-means clustering with TF-IDF features (`HashingTF` + `IDF`) for topic grouping, linear regression (`pyspark.ml.regression`) for citation prediction
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
4. The notebook was also checked against the 2024 and 2025 versions of the ICLR dataset (iclr24v2, iclr25v2), where all six questions matched the expected answers.

## Data

The ICLR submissions dataset and the OpenAlex export are not redistributed here. The notebook's own outputs (visible directly in the `.ipynb` file) are the record of what running it actually produced, standing in for a live rerun.

## Limitations

- Q5 and Q6 each have two implementations that give different answers (1,593 vs. 1,602 matched papers; 13 vs. 8 low-scored, highly-cited papers); see the Verification note. Both are reported.
- Only 2.9% of ICLR submissions have OpenAlex citation data, and papers indexed in OpenAlex are likely to be more cited than average, so the citation results (Q5 extension, Q6 and the regression) describe a small, non-representative subset, not all submissions.
- Citation counts are not normalised by year or field, so older papers have a built-in advantage; the regression includes submission year to partly account for this, but the relationship is unlikely to be linear.
- The clustering (Q4 extension) has a weak silhouette score even at its best (k=3, 0.0581), so the data does not form clearly separable clusters; the three groups are approximate topic groupings, not definitive research communities, consistent with ML/AI topics overlapping heavily.
- The citation-prediction linear regression (Q6 extension) explains very little variance (test R² = 0.0545, on roughly 320 held-out papers); I report this as a finding (review scores are a poor predictor of citation count), not a modelling failure to be fixed.

## Verification note

Q5 and Q6 each have two implementations in the notebook: the basic version that answers the question as set (1,593 matched papers; 13 low-scored, highly-cited papers, using OpenAlex's own percentile field) and a more careful version with normalised titles and deduplication (1,602; 8, using a locally computed 95th-percentile threshold). The counts differ because of those method choices, so both are reported. I'd treat the deduplicated version as more defensible; the citation regression uses it. All other figures had a single implementation and match my write-up.
