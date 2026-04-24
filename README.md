# Project 3: ML Drug Response Prediction (GDSC)

![CI](https://github.com/adamhoffman2155-hue/project-3-ml-classification/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Repro](https://img.shields.io/badge/FAIR_DOME_CURE-11%2F14_%7C_5%2F7_%7C_4%2F4-brightgreen)

> **Branch status:** `main` holds the GDSC POC v2 (RandomForest CV AUC 0.711 ± 0.085, DDR quantitative null result, committed in `results/poc/`); this feature branch `claude/research-industry-standards-bQAWX` adds uniform polish (CI, pre-commit, sklearn 1.4 compat fix, REPRODUCIBILITY.md, LogReg/GBM/XGBoost benchmark, cross-project docs, SHAP-claim cleanup) on top of `master` (pre-POC). The intended path is to merge this feature branch's polish layer into `main` — the branches are additive, not independently authoritative.

**Research question:** Can genomic features predict drug sensitivity across cancer cell lines?

This is the third project in a [computational biology portfolio](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio). After Projects 1–2 identified transcriptomic and immune signatures in GEA, this project asks whether those genomic features can actually predict drug response — moving from descriptive biology to predictive modeling framed around the GDSC2 pharmacogenomics dataset.

## What It Does

A reusable classical-ML + neural-net benchmarking harness for binarised drug-response prediction from genomic feature matrices:

1. **Data loading / cleaning** (`src/data.py`) — generic CSV feature + label loader, missing-value imputation, constant / highly-correlated feature removal, `DataPreprocessor` (standard scaling)
2. **Feature engineering** (`src/features.py`) — mutation profile / CNV / pathway-score transformations
3. **Classical models** (`src/models.py`) — Random Forest, XGBoost, ElasticNet with stratified 5-fold CV
4. **Deep-learning baseline** (`src/neural_net.py`) — feed-forward network
5. **Evaluation** (`src/evaluation.py`) — ROC-AUC, precision-recall, confusion matrices, permutation/tree-based feature importance

Data and labels are passed in as CSVs — the loader is dataset-agnostic. The project is framed around GDSC2 IC50 endpoints (with a focus on BRCA-pathway drugs relevant to the DDR biology from my thesis), but the actual harness runs on whatever feature / label CSVs are supplied.

## Methods & Tools

| Category | Tools |
|----------|-------|
| Classical ML | Random Forest, XGBoost, ElasticNet (scikit-learn) |
| Deep learning | Feed-forward neural net |
| Interpretability | Permutation importance, sklearn `feature_importances_` |
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
│   ├── models.py            # RF / XGBoost / ElasticNet
│   ├── neural_net.py        # FFN baseline
│   ├── evaluation.py        # CV, metrics, permutation importance
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

pytest
```

## My Role

I chose the GDSC2 dataset and DDR/biomarker framing based on my thesis work, and evaluated whether feature-importance outputs matched expected biology for BRCA-pathway drugs (SHAP-based biomarker ranking is handled in Project 4, which is where the `shap` dependency actually runs). Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 3 of 7**. It marks the transition from descriptive transcriptomics (Projects 1–2) to predictive modeling — asking whether the biology I identified can actually predict drug response. The pharmacogenomics approach here is extended in Project 4 with DDR-specific biomarkers (where the GDSC2 data is actually loaded and benchmarked). See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

### Cross-project data flow

```
Project 1 (bulk RNA-seq DEGs) ──┐
                                │   (candidate features, narrative)
                                ▼
Project 3 (this one — broad ML harness)
                                │   drug-response predictions
                                ▼
Project 4 (DDR-specific biomarkers + SHAP) ──▶ Project 6 (survival)
```

- **Upstream** — DEG / pathway matrices from Project-1 as candidate features (conceptual).
- **Downstream** — Project-4 narrows the framing to the DDR biology and layers SHAP on top; Project-6 integrates the drug-response story into its survival covariate panel (narrative input).

## Benchmarks

| Benchmark | Output | Summary |
| --- | --- | --- |
| LogReg baseline vs sklearn GBM vs XGBoost | [`results/benchmark/leaderboard.md`](results/benchmark/leaderboard.md) | On a synthetic 500×30 binary task, GBM (0.923) and XGBoost (0.912) beat the LogReg baseline (0.816) by ~0.10 CV AUC — confirms the harness can report leaderboard-style numbers without running the full pipeline. |

Rebuild with `python scripts/benchmark_models.py`.

## Reproducibility

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the FAIR-BioRS / DOME / CURE self-scorecard (11/14 · 5/7 · 4/4).

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
