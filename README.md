# Project 3: ML Drug Response Prediction (GDSC)

**Research question:** Can genomic features predict drug sensitivity across cancer cell lines?

This is the third project in a [computational biology portfolio](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio). After Projects 1–2 identified transcriptomic and immune signatures in GEA, this project asks whether those genomic features can actually predict drug response — moving from descriptive biology to predictive modeling using the GDSC2 pharmacogenomics dataset.

## What It Does

Benchmarks multiple ML models on GDSC2 cell-line data to predict IC50 drug sensitivity from genomic features:

1. **Data loading** — GDSC2 IC50 values + genomic feature matrices (`src/data.py`)
2. **Feature engineering** — Mutation profiles, copy number, pathway scores (`src/features.py`)
3. **Model training** — Random Forest, XGBoost, ElasticNet with stratified 5-fold CV (`src/models.py`)
4. **Deep learning baseline** — Feed-forward network (`src/neural_net.py`)
5. **Evaluation** — ROC-AUC, precision-recall, confusion matrices, SHAP interpretability (`src/evaluation.py`)

The pipeline explores which genomic features correlate with drug sensitivity, with particular focus on BRCA-pathway drugs relevant to the DDR biology from my thesis.

## Methods & Tools

| Category | Tools |
|----------|-------|
| Classical ML | Random Forest, XGBoost, ElasticNet (scikit-learn) |
| Deep learning | Feed-forward neural net |
| Interpretability | SHAP |
| Validation | Stratified 5-fold cross-validation |
| Data | GDSC2 pharmacogenomics |
| Visualization | matplotlib, seaborn |
| Testing | pytest |
| Environment | Docker, Conda |

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
│   ├── data.py              # GDSC2 loaders
│   ├── features.py          # Feature engineering
│   ├── models.py            # RF / XGBoost / ElasticNet
│   ├── neural_net.py        # FFN baseline
│   ├── evaluation.py        # CV, metrics, SHAP
│   └── utils.py
└── tests/                   # pytest suite
```

Trained model artefacts, predictions, metrics, and plots are written under `data/`, `models/`, and `results/` at runtime and are gitignored.

## Quick Start

```bash
git clone https://github.com/adamhoffman2155-hue/project-3-ml-classification.git
cd project-3-ml-classification

# Using Docker
docker build -t ml-classification .
docker run -it -v $(pwd):/workspace ml-classification bash

# Or Conda / pip
conda env create -f environment.yml
conda activate ml-classification
# or: pip install -r requirements.txt

pytest
```

## My Role

I chose the GDSC2 dataset and DDR/biomarker framing based on my thesis work, and evaluated whether SHAP outputs matched expected biology for BRCA-pathway drugs. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 3 of 7**. It marks the transition from descriptive transcriptomics (Projects 1–2) to predictive modeling — asking whether the biology I identified can actually predict drug response. The pharmacogenomics approach here is extended in Project 4 with DDR-specific biomarkers. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
