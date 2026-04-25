"""Benchmark: LogReg baseline vs sklearn GBM vs XGBoost on synthetic data.

Generates a small synthetic classification dataset (500 samples, 30 features)
and reports 5-fold CV ROC-AUC for each model. Additive — does not touch any
POC output. Writes ``results/benchmark/leaderboard.csv`` + markdown summary.

Run:
    python scripts/benchmark_models.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.datasets import make_classification  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "benchmark"
OUT_CSV = OUT_DIR / "leaderboard.csv"
OUT_MD = OUT_DIR / "leaderboard.md"

SEED = 42


def _xgboost_factory():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    def _make():
        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
        )

    return _make


def _score(model, x, y) -> tuple[float, float]:
    scores = cross_val_score(model, x, y, scoring="roc_auc", cv=5, n_jobs=-1)
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y = make_classification(
        n_samples=500,
        n_features=30,
        n_informative=8,
        n_redundant=4,
        random_state=SEED,
    )

    rows: list[dict[str, object]] = []

    mean, std = _score(LogisticRegression(max_iter=1000, random_state=SEED), x, y)
    rows.append({"model": "LogisticRegression (baseline)", "cv_auc_mean": mean, "cv_auc_std": std})
    print(f"LogReg:  {mean:.3f} ± {std:.3f}")

    mean, std = _score(
        GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=SEED), x, y
    )
    rows.append({"model": "sklearn GradientBoosting", "cv_auc_mean": mean, "cv_auc_std": std})
    print(f"GBM:     {mean:.3f} ± {std:.3f}")

    xgb_factory = _xgboost_factory()
    if xgb_factory is not None:
        mean, std = _score(xgb_factory(), x, y)
        rows.append({"model": "XGBoost", "cv_auc_mean": mean, "cv_auc_std": std})
        print(f"XGBoost: {mean:.3f} ± {std:.3f}")
    else:
        rows.append(
            {
                "model": "XGBoost (skipped — xgboost not installed)",
                "cv_auc_mean": float("nan"),
                "cv_auc_std": float("nan"),
            }
        )
        print("XGBoost: skipped (not installed)")

    df = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False, na_position="last")
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    lines = [
        "# Benchmark: LogReg baseline vs sklearn GBM vs XGBoost",
        "",
        "Synthetic binary classification (500 samples × 30 features, 8 informative,",
        "4 redundant, seed=42). 5-fold CV ROC-AUC, stratified implicitly by cross_val_score.",
        "",
        "| Model | CV AUC (mean ± std) |",
        "| --- | ---: |",
    ]
    for _, row in df.iterrows():
        if isinstance(row["cv_auc_mean"], float) and not np.isnan(row["cv_auc_mean"]):
            lines.append(f"| {row['model']} | {row['cv_auc_mean']:.3f} ± {row['cv_auc_std']:.3f} |")
        else:
            lines.append(f"| {row['model']} | — |")
    lines += [
        "",
        "## Interpretation",
        "",
        "LogisticRegression provides the 'can a linear model even do this' baseline;",
        "both tree ensembles are expected to beat it on a task with informative +",
        "redundant feature interactions. This benchmark is the industry-standard",
        "sanity-check when swapping ML models — it demonstrates that the harness",
        "can report leaderboard-style numbers without re-running the full",
        "pipeline.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
