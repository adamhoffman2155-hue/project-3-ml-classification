#!/usr/bin/env python3
"""
Regenerate ``results/poc/manifest.json`` from the artefacts that
``run_poc.py`` writes (``cv_auc.csv``). Matches the manifest contract
documented at
``bioinformatics-portfolio/shared/poc-manifest.schema.json``.

Usage
-----
    python scripts/poc/build_manifest.py

Exits non-zero if the required source files are missing so a stale
manifest can't silently ship.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
POC = REPO / "results" / "poc"
OUT = POC / "manifest.json"


def main() -> int:
    cv_csv = POC / "cv_auc.csv"
    if not cv_csv.is_file():
        print(f"ERROR: missing {cv_csv}", file=sys.stderr)
        return 1

    cv = pd.read_csv(cv_csv)
    best = cv.sort_values("cv_auc_mean", ascending=False).iloc[0]
    n_features = int(best["n_features"])
    n_train = int(best["n_train"])

    manifest = {
        "$schema": (
            "https://github.com/adamhoffman2155-hue/bioinformatics-portfolio/"
            "blob/main/shared/poc-manifest.schema.json"
        ),
        "project": "project-3-ml-classification",
        "poc_title": (
            "ML drug-response harness on synthetic-GDSC fixture"
        ),
        "poc_version": "v1",
        "dataset": {
            "name": "Synthetic GDSC fixture",
            "source": "scripts/generate_synthetic_gdsc.py",
            "substitute_for": (
                "real GDSC v17 (not committed; out-of-repo)"
            ),
            "n_cell_lines": 200,
            "n_features": n_features,
            "n_planted_driver_features": 3,
        },
        "script": "scripts/poc/run_poc.py",
        "generated_at": date.today().isoformat(),
        "headline_metric": {
            "name": f"{best['classifier']} 5-fold CV ROC-AUC",
            "value": round(float(best["cv_auc_mean"]), 3),
            "std": round(float(best["cv_auc_std"]), 3),
            "n_train": n_train,
            "note": (
                "StratifiedKFold(5, shuffle=True, random_state=42) on the "
                "75% train fold; signal planted in 3 of 50 features"
            ),
        },
        "secondary_metrics": [
            {
                "name": "All classifiers 5-fold CV ROC-AUC",
                "rows": cv[
                    ["classifier", "cv_auc_mean", "cv_auc_std"]
                ].to_dict(orient="records"),
            }
        ],
        "headline_text": (
            f"{best['classifier']} CV ROC-AUC "
            f"{float(best['cv_auc_mean']):.3f} ± "
            f"{float(best['cv_auc_std']):.3f} on the committed synthetic "
            f"fixture ({n_train} train × {n_features} features); "
            "reproducible on any fresh clone."
        ),
        "artifacts": [
            "results/poc/poc_summary.txt",
            "results/poc/cv_auc.csv",
            "results/poc/per_fold_auc.csv",
        ],
    }

    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {OUT}")
    print(f"  headline: {manifest['headline_text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
