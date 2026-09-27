# Data

Read before downloading anything or accepting terms. Nothing in this file has been downloaded. Accepting a license or gated-access terms requires asking Sathvik first.

Licenses below are as recorded in the 2026-09-26 handoff (`project-c.md` §9). They were not re-checked in the session that created this file. Re-read the source page before download and correct this file if the terms differ.

## Defaults

| Dataset | Role | License as recorded | Source | Status |
|---|---|---|---|---|
| arXiv metadata | Default T1 | CC0. About 1.7M+ papers, 1991 onward; title, abstract, categories, version dates; about 1–4 GB compressed | kaggle.com/datasets/Cornell-University/arxiv | not downloaded |
| HuffPost News Category | Default T2 | About 210K headlines, 2012–2022, 42 categories. CC BY 4.0 per the paper. Wild-Time lists its slice as CC0; use the stricter terms | kaggle.com/datasets/rmisra/news-category-dataset ; arxiv.org/abs/2209.11429 | not downloaded |
| Wild-Time (NeurIPS 2022 D&B) | Split reference, not a third task | Code MIT. Temporal-shift benchmark with arXiv and HuffPost splits by year | github.com/huaxiuyao/Wild-Time ; arxiv.org/abs/2211.14238 | not downloaded |

## Optional, not in the core

| Dataset | Role | License as recorded | Source | Status |
|---|---|---|---|---|
| CivilComments | Optional | CC0. About 2M comments, 2015–2017 only; toxicity scores | huggingface.co/datasets/google/civil_comments | not downloaded |
| WildChat-1M | Optional, for a later agent or chat addition | ODC-BY. Timestamps span about one year; toxicity and language labels | huggingface.co/datasets/allenai/WildChat-1M | not downloaded |

An optional dataset is an addition under `PROJECT_SPEC.md`. It does not replace T1 or T2.

## Do not download

- Stack Exchange dumps. The 2024 terms forbid LLM-training use.
- LMSYS-Chat-1M. Gated custom agreement.
- Yelp and Amazon review data. Non-commercial terms.
- MIMIC. Credentialed access.

## Before a download is allowed

1. Ask Sathvik to accept the terms.
2. Confirm the license line in this file against the source, and correct it if needed.
3. Record the download date, the exact file or revision, and where it lives on disk. Dataset bytes do not belong in git.
