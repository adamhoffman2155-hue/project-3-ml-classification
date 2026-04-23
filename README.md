# Project 3: ML Drug Response Prediction (GDSC)

[![CI](https://github.com/adamhoffman2155-hue/project-3-ml-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/adamhoffman2155-hue/project-3-ml-classification/actions/workflows/ci.yml)

**Research question:** Can genomic features predict drug sensitivity across cancer cell lines?

This is the third project in a [computational biology portfolio](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio). After Projects 1–2 identified transcriptomic and immune signatures in GEA, this project asks whether those genomic features can actually predict drug response — moving from descriptive biology to predictive modeling framed around the GDSC2 pharmacogenomics dataset.

## What It Does

A reusable classical-ML + neural-net benchmarking harness for binarised drug-response prediction from genomic feature matrices:

1. **Data loading / cleaning** (`src/data.py`) — generic CSV feature + label loader, missing-value imputation, constant / highly-correlated feature removal, `DataPreprocessor` (standard scaling)
2. **Feature engineering** (`src/features.py`) — mutation profile / CNV / pathway-score transformations
3. **Classical models** (`src/models.py`) — Logistic Regression, Random Forest, Gradient Boosting, and Elastic Net (via `SGDClassifier(penalty='elasticnet')`), all scikit-learn, with stratified 5-fold CV
4. **Deep-learning baseline** (`src/neural_net.py`) — feed-forward network
5. **Evaluation** (`src/evaluation.py`) — ROC-AUC, precision-recall, confusion matrices, SHAP-based feature importance

Data and labels are passed in as CSVs — the loader is dataset-agnostic. The project is framed around GDSC2 IC50 endpoints (with a focus on BRCA-pathway drugs relevant to the DDR biology from my thesis), but the actual harness runs on whatever feature / label CSVs are supplied.

## Methods & Tools

| Category | Tools |
|----------|-------|
| Classical ML | Logistic Regression, Random Forest, Gradient Boosting, Elastic Net (via SGD) — all scikit-learn |
| Deep learning | Feed-forward neural net |
| Interpretability | SHAP |
| Validation | Stratified 5-fold cross-validation |
| Intended data | GDSC2 pharmacogenomics (IC50 + genomic features) |
| Visualization | matplotlib, seaborn |
| Testing | pytest (`test_data.py`, `test_features.py`, `test_models.py`) |
| Environment | Docker, Conda, pip |

## Project Structure

```
project-3-ml-classification/
├── Dockerfile
├── environment.yml
├── requirements.txt
├── config/
│   └── model_config.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py              # Generic CSV loader + DataPreprocessor
│   ├── features.py          # Feature engineering transforms
│   ├── models.py            # sklearn: LogReg / RandomForest / GradientBoosting / SGD-ElasticNet
│   ├── neural_net.py        # FFN baseline
│   ├── evaluation.py        # CV, metrics, SHAP
│   └── utils.py
└── tests/
    ├── test_data.py
    ├── test_features.py
    └── test_models.py
```

Trained model artefacts, predictions, metrics, and plots are written under `data/`, `models/`, and `results/` at runtime and are gitignored.

## Quick Start

```bash
git clone https://github.com/adamhoffman2155-hue/project-3-ml-classification.git
cd project-3-ml-classification

# Choose one environment
docker build -t ml-classification . && docker run -it -v $(pwd):/workspace ml-classification bash
#   or
conda env create -f environment.yml && conda activate ml-classification
#   or
pip install -r requirements.txt

# Generate the synthetic-GDSC fixture so tests + integration runs have data
python scripts/generate_synthetic_gdsc.py

pytest
```

The synthetic fixture (`data/synthetic/features.csv`, `data/synthetic/labels.csv`)
is a 200-cell-line × 50-feature matrix with three planted "driver" features and
a balanced binary sensitivity label. `tests/test_integration.py` trains
Logistic Regression, Random Forest, and Gradient Boosting end-to-end on it and
asserts CV ROC-AUC > 0.6 — this is the repo's "it runs on a fresh clone"
acceptance test.

## My Role

I chose the GDSC2 dataset and DDR/biomarker framing based on my thesis work, and evaluated whether SHAP outputs matched expected biology for BRCA-pathway drugs. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 3 of 7**. It marks the transition from descriptive transcriptomics (Projects 1–2) to predictive modeling — asking whether the biology I identified can actually predict drug response. The pharmacogenomics approach here is extended in Project 4 with DDR-specific biomarkers (where the GDSC2 data is actually loaded and benchmarked). See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
