# Project 3: ML Drug Response Prediction (GDSC)

**Research question:** Can genomic features predict drug sensitivity across cancer cell lines?

This is the third project in a [computational biology portfolio](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio). After Projects 1–2 identified transcriptomic and immune signatures in GEA, this project asks whether those genomic features can actually predict drug response — moving from descriptive biology to predictive modeling using the GDSC2 pharmacogenomics dataset.

## At a Glance

| | |
|---|---|
| **Stack** | scikit-learn (RF · XGBoost · ElasticNet) · SHAP · stratified 5-fold CV · Docker |
| **Data** | GDSC2 IC50 + genomic feature matrices (public) |
| **POC headline** | Random Forest CV AUC 0.71 on Olaparib (+12 pts over 39-gene baseline); quantitative null result for DDR gene ranking |
| **Role** | Dataset & DDR framing from thesis; biology review of SHAP outputs; implementation AI-assisted |
| **Portfolio** | Project 3 of 7 · [full narrative](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) |

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
├── Dockerfile
├── environment.yml
├── requirements.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classical_ml.ipynb
│   ├── 04_deep_learning.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   └── 07_model_comparison.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── neural_net.py
│   ├── evaluation.py
│   ├── utils.py
│   └── config.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── models/
│   ├── trained/
│   └── predictions/
├── results/
│   ├── metrics/
│   ├── plots/
│   └── reports/
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
└── config/
    └── model_config.yaml
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

jupyter lab
```

## My Role

I chose the GDSC2 dataset and DDR/biomarker framing based on my thesis work, and evaluated whether SHAP outputs matched expected biology for BRCA-pathway drugs. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 3 of 7**. It marks the transition from descriptive transcriptomics (Projects 1–2) to predictive modeling — asking whether the biology I identified can actually predict drug response. The pharmacogenomics approach here is extended in Project 4 with DDR-specific biomarkers. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
