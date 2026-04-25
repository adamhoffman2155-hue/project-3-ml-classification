# Project 3: ML Drug Response Prediction (GDSC)

![CI](https://github.com/adamhoffman2155-hue/project-3-ml-classification/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Repro](https://img.shields.io/badge/FAIR_DOME_CURE-11%2F14_%7C_5%2F7_%7C_4%2F4-brightgreen)

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
├── README.md
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
├── scripts/
│   ├── benchmark_models.py  # LogReg vs GBM vs XGBoost benchmark
│   └── poc/
│       └── run_poc.py       # Proof-of-concept runner
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
└── results/
    └── poc/                 # POC outputs (committed)
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
