# Benchmark: LogReg baseline vs sklearn GBM vs XGBoost

Synthetic binary classification (500 samples × 30 features, 8 informative,
4 redundant, seed=42). 5-fold CV ROC-AUC, stratified implicitly by cross_val_score.

| Model | CV AUC (mean ± std) |
| --- | ---: |
| sklearn GradientBoosting | 0.923 ± 0.021 |
| XGBoost | 0.912 ± 0.028 |
| LogisticRegression (baseline) | 0.816 ± 0.026 |

## Interpretation

LogisticRegression provides the 'can a linear model even do this' baseline;
both tree ensembles are expected to beat it on a task with informative +
redundant feature interactions. This benchmark is the industry-standard
sanity-check when swapping ML models — it demonstrates that the harness
can report leaderboard-style numbers without re-running the full
pipeline.
