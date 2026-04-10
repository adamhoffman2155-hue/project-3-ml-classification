# Project 3: ML Drug Response Prediction (GDSC)

**Research question:** Can genomic features predict drug sensitivity across cancer cell lines?

This is the third project in a [computational biology portfolio](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio). After Projects 1–2 identified transcriptomic and immune signatures in GEA, this project asks whether those genomic features can actually predict drug response — moving from descriptive biology to predictive modeling using the GDSC2 pharmacogenomics dataset.

## What It Does

Benchmarks multiple ML models on GDSC2 cell line data to predict IC50 drug sensitivity from genomic features:

1. **Data loading** — GDSC2 IC50 values + genomic feature matrices
2. **Feature engineering** — Mutation profiles, copy number, pathway scores
3. **Model training** — Random Forest, XGBoost, ElasticNet with stratified 5-fold CV
4. **Feature inspection** — SHAP values for model interpretability
5. **Evaluation** — ROC-AUC, precision-recall, confusion matrices
6. **Model comparison** — Benchmark across all approaches

The pipeline explores which genomic features correlate with drug sensitivity, with particular focus on BRCA-pathway drugs relevant to the DDR biology from my thesis.

## Methods & Tools

| Category | Tools |
|----------|-------|
| ML Models | Random Forest, XGBoost, ElasticNet (scikit-learn) |
| Interpretability | SHAP |
| Validation | Stratified 5-fold cross-validation |
| Data | GDSC2 pharmacogenomics |
| Visualization | matplotlib, seaborn |
| Environment | Docker, Conda |

## Project Structure

```
project-3-ml-classification/
├── README.md
├── Dockerfile
├── environment.yml
├── requirements.txt
├── config/
│   └── model_config.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   ├── models.py
│   ├── neural_net.py
│   └── utils.py
├── scripts/
│   └── poc/
│       └── run_poc.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
└── results/
    └── poc/
```

## Quick Start

```bash
git clone https://github.com/adamhoffman2155-hue/project-3-ml-classification.git
cd project-3-ml-classification

# Using Docker
docker build -t ml-classification .
docker run -it -v $(pwd):/workspace ml-classification bash

# Or Conda
conda env create -f environment.yml
conda activate ml-classification
```

## Proof of Concept

A minimal end-to-end ML run on GDSC-derived cell line data, predicting binarized drug sensitivity from mutation features.

**Dataset:** GDSC binary mutation matrix + IC50 values (878 cell lines × 534 genes), accessed via a published GitHub mirror (`hanjunwei-lab/DeepCCDS`) which re-distributes the Sanger CancerRxGene data. The original `cog.sanger.ac.uk` release-8.5 endpoints were unreachable from the reproducibility sandbox.

**Task:** Binarize Olaparib (PARP inhibitor) IC50 into sensitive (bottom 25%) vs resistant (top 25%) classes, drop the middle 50%, then predict class from a 39-gene mutation panel (DDR + HRD + pan-cancer drivers) using stratified 5-fold CV.

**Reproduce:**
```bash
pip install pandas numpy scikit-learn shap matplotlib
python scripts/poc/run_poc.py
```

**Results (actual 5-fold CV AUCs):**

| Model | Mean AUC | Std |
|---|---|---|
| LogisticRegression | **0.5949** | 0.0894 |
| GradientBoosting | 0.5785 | 0.0882 |
| RandomForest | 0.5623 | 0.0750 |

| Cohort | Value |
|---|---|
| N cell lines after filtering | 432 |
| Class balance | 216 sensitive / 216 resistant |
| Panel genes available | 39 of 49 |

**Top 10 SHAP features (Logistic Regression, best model):**

| Feature | mean \|SHAP\| |
|---|---|
| TP53 | 0.368 |
| RB1 | 0.290 |
| MSH2 | 0.280 |
| FANCA | 0.220 |
| BRAF | 0.201 |
| MYC | 0.199 |
| KRAS | 0.192 |
| FANCC | 0.188 |
| RAD51B | 0.159 |
| MSH6 | 0.149 |

**Honest assessment:** Best CV AUC = **0.595** — a **weak** signal. This is consistent with published literature: individual mutation markers (including BRCA1/2) are weak Olaparib predictors in cancer cell lines. HRD scar scores and transcriptomic features typically outperform mutation-only features. The SHAP ranking does show DDR/MMR genes (MSH2, FANCA, FANCC, RAD51B, MSH6) in the top 10 alongside pan-cancer drivers (TP53, RB1, KRAS, MYC), which is the expected biology.

**Limits:**
- Small effective sample (~432 cell lines after top/bottom quartile filtering)
- Binarization by quartile is arbitrary vs. clinical cut-offs
- Mutation-only features ignore copy number, methylation, and expression
- Pan-cancer pooled — not stratified by tissue type
- BRCA1/BRCA2 are present in the panel but not in the top 10 SHAP list, which is consistent with published findings that they are weak cell-line predictors without additional HRD-scar evidence
- The original `cog.sanger.ac.uk` release 8.5 files are unreachable from the sandbox; the POC uses a published GDSC-derived mirror on GitHub (documented in the script docstring)

## My Role

I chose the GDSC2 dataset and DDR/biomarker framing based on my thesis work, and evaluated whether SHAP outputs matched expected biology for BRCA-pathway drugs. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 3 of 7**. It marks the transition from descriptive transcriptomics (Projects 1–2) to predictive modeling — asking whether the biology I identified can actually predict drug response. The pharmacogenomics approach here is extended in Project 4 with DDR-specific biomarkers. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
