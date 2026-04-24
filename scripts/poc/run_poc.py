#!/usr/bin/env python3
"""
Reproducible Proof-of-Concept for the ML drug-response harness.

Runs the full synthetic-GDSC -> preprocess -> train -> evaluate path
and writes ``results/poc/`` artifacts plus a headline manifest. This
is the reproducible-on-a-fresh-clone POC that the README and portfolio
card should track. It is DIFFERENT from the historical hardcoded
Olaparib / GDSC v17 headline numbers — those were produced on real
data that is not committed to this repo and therefore cannot be
reproduced by anyone else. This POC uses the committed synthetic
fixture (``scripts/generate_synthetic_gdsc.py``), which means every
reviewer gets the same numbers on a fresh clone.

Artifacts written to ``results/poc/``:
  - poc_summary.txt : plain-text headline summary
  - cv_auc.csv      : 5-fold CV ROC-AUC mean +/- std per classifier
  - per_fold_auc.csv: per-fold breakdown
  - manifest.json   : regenerated from cv_auc.csv via build_manifest.py

Usage
-----
    python scripts/poc/run_poc.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

REPO = Path(__file__).resolve().parents[2]
SYN = REPO / "data" / "synthetic"
POC = REPO / "results" / "poc"

# Make the repo root importable so `from src.data import ...` works whether
# the POC is invoked from the repo root or from anywhere else.
sys.path.insert(0, str(REPO))


def _ensure_fixture() -> None:
    if (SYN / "features.csv").is_file() and (SYN / "labels.csv").is_file():
        return
    print("[poc] generating synthetic-GDSC fixture", flush=True)
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "generate_synthetic_gdsc.py")],
        check=True, cwd=REPO,
    )


def main() -> int:
    from src.data import load_genomic_data, DataPreprocessor
    from src import models as mdl

    POC.mkdir(parents=True, exist_ok=True)
    _ensure_fixture()

    X_train, X_test, y_train, y_test, feature_names = load_genomic_data(
        SYN / "features.csv", SYN / "labels.csv",
        test_size=0.25, random_state=0,
    )
    pp = DataPreprocessor()
    X_train_scaled = pp.fit_transform(X_train)
    X_test_scaled = pp.transform(X_test)

    classifiers = {
        "LogisticRegression": (
            lambda: mdl.LogisticRegression(
                max_iter=1000, solver="lbfgs", random_state=42,
            )
        ),
        "RandomForest": (
            lambda: mdl.RandomForestClassifier(
                n_estimators=200, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1,
            )
        ),
        "GradientBoosting": (
            lambda: mdl.GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5,
                min_samples_split=5, min_samples_leaf=2, subsample=0.8,
                random_state=42,
            )
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    per_fold_rows: list[dict] = []
    summary_rows: list[dict] = []

    for name, factory in classifiers.items():
        fold_aucs = cross_val_score(
            factory(), X_train_scaled, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1,
        )
        mean = float(np.mean(fold_aucs))
        std = float(np.std(fold_aucs, ddof=1))
        summary_rows.append({
            "classifier": name,
            "cv_auc_mean": round(mean, 4),
            "cv_auc_std": round(std, 4),
            "n_features": X_train_scaled.shape[1],
            "n_train": X_train_scaled.shape[0],
        })
        for i, auc in enumerate(fold_aucs):
            per_fold_rows.append({
                "classifier": name, "fold": i + 1,
                "cv_auc": round(float(auc), 4),
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "cv_auc_mean", ascending=False
    )
    summary_df.to_csv(POC / "cv_auc.csv", index=False)
    pd.DataFrame(per_fold_rows).to_csv(POC / "per_fold_auc.csv", index=False)

    best = summary_df.iloc[0]
    lines = [
        "POC: ML drug-response harness on synthetic-GDSC fixture",
        "=" * 60,
        "",
        "Dataset: synthetic fixture from scripts/generate_synthetic_gdsc.py",
        "  200 cell lines x 50 features (20 mutation + 20 CNV + 10 expression)",
        "  Three planted driver features (MUT_GENE_0, CNV_GENE_1, EXPR_GENE_2)",
        "  Balanced binary sensitivity label",
        "",
        "Workflow: load -> StandardScaler -> StratifiedKFold(5) CV on train split",
        "",
        "5-fold CV ROC-AUC (on 150-cell-line train fold, 50 held out for test):",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"  {row['classifier']:<18}  {row['cv_auc_mean']:.4f}  "
            f"+/- {row['cv_auc_std']:.4f}"
        )
    lines.append("")
    lines.append(
        f"Headline: {best['classifier']} CV AUC "
        f"{best['cv_auc_mean']:.3f} +/- {best['cv_auc_std']:.3f}"
    )
    lines.append("")
    lines.append("Reproduce:")
    lines.append("  python scripts/poc/run_poc.py")
    lines.append("")
    (POC / "poc_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    # Regenerate manifest.json from cv_auc.csv.
    print("\n[poc] rebuilding manifest.json")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("build_manifest.py"))],
        check=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
