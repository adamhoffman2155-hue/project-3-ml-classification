# Project 3: ML Drug Response Prediction (GDSC)

**Research question:** Can genomic features predict drug sensitivity across cancer cell lines?

This is the third project in a [computational biology portfolio](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio). After Projects 1–2 identified transcriptomic and immune signatures in GEA, this project asks whether those genomic features can actually predict drug response — moving from descriptive biology to predictive modeling using the GDSC pharmacogenomics dataset.

## At a Glance

| | |
|---|---|
| **Stack** | scikit-learn (LogReg · RF · GradBoost) · SHAP · stratified 5-fold CV · gdsctools · Docker |
| **Data** | GDSC v17 IC50 + genomic features (424 cell lines, 704 features post-quartile filter) |
| **POC headline** | Olaparib RF CV AUC **0.711 ± 0.085** on full 704-feature matrix; DDR genes rank median **400 / 704** (MWU p=0.844) — quantitative null result for mutation-only DDR hypothesis |
| **Status** | **Runnable POC** (reproducible via `scripts/poc/run_poc.py`); pipeline in `src/` is a scaffolding target |
| **Role** | Dataset & DDR framing from thesis; SHAP/DDR-enrichment interpretation; implementation AI-assisted |
| **Portfolio** | Project 3 of 7 · [full narrative](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) |

## What It Does

Benchmarks multiple ML models on GDSC cell-line data to predict Olaparib IC50 sensitivity from genomic features:

1. **Data loading** — GDSC v17 IC50 + genomic feature matrix (via `gdsctools`)
2. **Feature construction** — mutation indicators + copy-number gain/loss segments + tissue one-hots + MSI binary
3. **Label construction** — top-quartile IC50 = resistant, bottom-quartile = sensitive (balanced classes)
4. **Model training** — Logistic Regression, Random Forest, Gradient Boosting with stratified 5-fold CV
5. **Feature inspection** — SHAP values on best model for interpretability
6. **DDR-enrichment test** — Mann-Whitney U of DDR-gene feature ranks vs. background

The pipeline explores which genomic features correlate with Olaparib sensitivity, with particular focus on whether DDR-panel gene mutations carry the signal (spoiler: they don't — tissue context and copy-number segments dominate).

## Methods & Tools

| Category | Tools |
|----------|-------|
| ML Models | Random Forest, Gradient Boosting, Logistic Regression (scikit-learn) |
| Interpretability | SHAP (TreeExplainer, LinearExplainer) |
| Validation | Stratified 5-fold cross-validation |
| Data | GDSC v17 bundled in `gdsctools` PyPI package |
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
│   ├── config.py, data.py, features.py, models.py, evaluation.py, utils.py
├── scripts/
│   └── poc/
│       └── run_poc.py         # v2 POC: full 704-feature model + DDR enrichment
├── tests/
│   ├── test_data.py, test_features.py, test_models.py
└── results/
    └── poc/                   # committed run outputs
```

## Quick Start

```bash
git clone https://github.com/adamhoffman2155-hue/project-3-ml-classification.git
cd project-3-ml-classification

pip install -r requirements.txt
python scripts/poc/run_poc.py
```

## Proof of Concept (v2)

Full-feature GDSC Olaparib benchmark on cell-line data, with a quantitative test of the DDR-hypothesis.

**Dataset:** GDSC v17 (CancerRxGene / Sanger Institute) — bundled inside the `gdsctools` PyPI package. The canonical `cog.sanger.ac.uk` release-8.5 hosts aren't reachable from the reproducibility sandbox, so v17 (the gdsctools-bundled snapshot) is the accessible version.

**Feature matrix (after top/bottom-quartile IC50 filter):**
- 424 cell lines · 212 sensitive / 212 resistant (balanced)
- 704 features:
  - 270 mutation indicators (Hugo symbols)
  - 116 copy-number gain segments
  - 291 copy-number loss segments
  - TISSUE_FACTOR one-hot dummies
  - MSI_FACTOR binary

### Cross-validation ROC-AUC (5-fold stratified)

| Model | CV AUC | Std |
|---|---|---|
| **RandomForest** | **0.7106** | 0.0851 |
| GradientBoosting | 0.6879 | 0.0819 |
| LogisticRegression | 0.6207 | 0.0458 |

Best model: **Random Forest, AUC 0.711 ± 0.085**.

### Top 15 SHAP features (Random Forest)

Dominated by **tissue context and copy-number segments**, not DDR mutations:

| Rank | Feature | mean \|SHAP\| |
|---|---|---|
| 1 | TISSUE_FACTOR_leukemia | 0.0170 |
| 2 | gain_cnaPANCAN363_(ASXL1) | 0.0101 |
| 3 | TISSUE_FACTOR_lung_NSCLC | 0.0092 |
| 4 | TISSUE_FACTOR_lymphoma | 0.0076 |
| 5 | TISSUE_FACTOR_breast | 0.0055 |
| … | copy-number segments, TP53_mut | … |

### DDR-panel enrichment test (Mann-Whitney U)

Tests whether DDR-panel gene mutations rank higher (by mean |SHAP|) than background features:

| Metric | Value |
|---|---|
| DDR panel size | 26 genes queried |
| Present in v17 matrix | 7 / 26 (BRCA1, BRCA2, ATM, ATR, CHEK2, MLH1, MLH3) |
| DDR median rank (1 = top) | **400 / 704** |
| MWU one-sided p (DDR > bg) | **p = 0.844** |

| DDR feature | mean \|SHAP\| | Rank |
|---|---|---|
| BRCA2_mut | 0.0006 | 112 |
| ATM_mut | 0.0003 | 180 |
| CHEK2_mut | 0.0000 | 392 |
| MLH3_mut | 0.0000 | 400 |
| MLH1_mut | 0.0000 | 401 |
| ATR_mut | 0.0000 | 456 |
| BRCA1_mut | 0.0000 | 602 |

This is a **quantitative negative result for the mutation-only DDR hypothesis** in cell lines — consistent with published literature that transcriptomic and HRD-scar features outperform mutation panels.

### v1 vs v2 comparison

| Version | Feature matrix | Best model | CV AUC |
|---|---|---|---|
| v1 | 39-gene DDR panel, 432 cell lines | LogReg | 0.595 ± 0.089 |
| **v2** | **704-feature full GDSC v17, 424 cell lines** | **RandomForest** | **0.711 ± 0.085** |

+12 AUC points from including copy-number segments and tissue context — but the DDR-specific hypothesis (mutations alone) remains null.

### Reproduce

```bash
pip install pandas numpy scipy scikit-learn shap matplotlib gdsctools
python scripts/poc/run_poc.py
```

Outputs land in `results/poc/`:
- `cv_metrics.csv` — 5-fold CV AUC for all three models
- `feature_importance.csv` — top-30 SHAP features
- `ddr_feature_rank.csv` — DDR panel feature ranks
- `shap_summary.png` — top-15 bar plot
- `poc_summary.txt` — full run log

### Honest assessment

- Predicting Olaparib sensitivity from mutations/CNVs alone is a **known hard problem**. Published pan-cancer mutation-only benchmarks land at 0.55–0.65 AUC; transcriptomic and HRD-scar predictors do better.
- The v2 improvement over v1 comes primarily from copy-number segments and TISSUE_FACTOR — **not DDR mutations per se**.
- The Mann-Whitney test explicitly shows DDR mutations do **not** rank higher than background (p=0.844). This is a quantitative negative result, not a framing choice.
- All numbers are real 5-fold CV output on held-out folds (committed in `results/poc/cv_metrics.csv`).

## My Role

I chose the GDSC dataset and DDR/biomarker framing based on my thesis work, and evaluated whether SHAP outputs matched expected biology for BRCA-pathway drugs. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 3 of 7**. It marks the transition from descriptive transcriptomics (Projects 1–2) to predictive modeling — asking whether the biology I identified can actually predict drug response. The pharmacogenomics approach here is extended in Project 4 with DDR-specific biomarkers and MSI-aware analysis. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
