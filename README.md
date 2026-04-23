# Project 3: ML Drug Response Prediction (GDSC)

> **A machine-learning model that tries to predict which cancer cell lines respond to a cancer drug — from their DNA alone — and honestly reports where that approach breaks down.**

## The short version

**What this project does.** Trains three machine-learning models on 424 cancer cell lines to predict whether each one is sensitive or resistant to Olaparib (a targeted PARP inhibitor). Then quantitatively tests a specific hypothesis: that DNA-repair gene mutations are the feature that drives the prediction.

**The question behind it.** Olaparib is supposed to work best in tumors with DNA-repair defects. If that story holds up in cell lines, mutations in genes like BRCA1, BRCA2, and ATM should be the top features the model uses. Does the story hold?

**What the proof-of-concept shows.** The best model (Random Forest on 704 features) reaches **71% accuracy** (ROC-AUC 0.711 ± 0.085) at separating sensitive vs. resistant cell lines. **But** a direct statistical test shows the DNA-repair genes we'd expect to matter rank at median **position 400 out of 704** (Mann-Whitney p = 0.844) — they don't carry the signal. What actually does: tissue type and copy-number patterns.

**Why it matters.** This is a useful negative result. It agrees with published studies showing that mutation-only panels don't predict PARP-inhibitor response well in cell lines — you need transcriptomic data or HRD-scar scores. Showing where a hypothesis fails, with the numbers to back it, is as important as showing where it succeeds.

---

_The rest of this README is technical detail for bioinformaticians, recruiters doing a deep review, or anyone reproducing the work._

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

## Methods & Tools

| Category | Tools |
|----------|-------|
| ML Models | Random Forest, Gradient Boosting, Logistic Regression (scikit-learn) |
| Interpretability | SHAP (TreeExplainer, LinearExplainer) |
| Validation | Stratified 5-fold cross-validation |
| Data | GDSC v17 bundled in `gdsctools` PyPI package |
| Visualization | matplotlib, seaborn |
| Environment | Docker, Conda |

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
- 704 features: 270 mutation indicators + 116 copy-number gains + 291 copy-number losses + TISSUE_FACTOR dummies + MSI_FACTOR binary

### Cross-validation ROC-AUC (5-fold stratified)

| Model | CV AUC | Std |
|---|---|---|
| **RandomForest** | **0.7106** | 0.0851 |
| GradientBoosting | 0.6879 | 0.0819 |
| LogisticRegression | 0.6207 | 0.0458 |

### Top 5 SHAP features (Random Forest)

Dominated by **tissue context and copy-number segments**, not DDR mutations:

| Rank | Feature | mean \|SHAP\| |
|---|---|---|
| 1 | TISSUE_FACTOR_leukemia | 0.0170 |
| 2 | gain_cnaPANCAN363_(ASXL1) | 0.0101 |
| 3 | TISSUE_FACTOR_lung_NSCLC | 0.0092 |
| 4 | TISSUE_FACTOR_lymphoma | 0.0076 |
| 5 | TISSUE_FACTOR_breast | 0.0055 |

### DDR-panel enrichment test (Mann-Whitney U)

| Metric | Value |
|---|---|
| DDR panel size | 26 genes queried |
| Present in v17 matrix | 7 / 26 |
| DDR median rank (1 = top) | **400 / 704** |
| MWU one-sided p (DDR > bg) | **p = 0.844** |

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
- `shap_summary.png`, `poc_summary.txt`

### Honest assessment

- Predicting Olaparib sensitivity from mutations/CNVs alone is a **known hard problem**. Published pan-cancer mutation-only benchmarks land at 0.55–0.65 AUC; transcriptomic and HRD-scar predictors do better.
- The v2 improvement comes primarily from copy-number segments and TISSUE_FACTOR — **not DDR mutations per se**.
- The Mann-Whitney test explicitly shows DDR mutations do **not** rank higher than background (p=0.844). This is a quantitative negative result, not a framing choice.

## My Role

I chose the GDSC dataset and DDR/biomarker framing based on my thesis work, and evaluated whether SHAP outputs matched expected biology for BRCA-pathway drugs. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 3 of 7**. It marks the transition from descriptive transcriptomics (Projects 1–2) to predictive modeling — asking whether the biology I identified can actually predict drug response. The pharmacogenomics approach here is extended in Project 4 with DDR-specific biomarkers and MSI-aware analysis. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
