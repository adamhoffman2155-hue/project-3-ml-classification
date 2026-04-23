#!/usr/bin/env python3
"""
Generate a synthetic GDSC-shaped dataset so the pipeline runs end-to-end on a
fresh clone without requiring external data.

Outputs
-------
data/synthetic/features.csv : 200 cell lines x 50 features
    * 20 mutation columns (binary)
    * 20 copy-number columns (float, centred)
    * 10 expression columns (float)
Index column: cell_line_id
data/synthetic/labels.csv : 200 rows, binary sensitivity label
    Signal: y = 1 when (MUT_DRIVER_0 + CNV_DRIVER_1 + 0.7*EXPR_DRIVER_2 + noise) > threshold,
    so RandomForest / GradientBoosting should recover a cross-val AUC > 0.7.

Usage
-----
    python scripts/generate_synthetic_gdsc.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

N_CELL_LINES = 200
N_MUT = 20
N_CNV = 20
N_EXPR = 10
RANDOM_STATE = 42

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"


def generate():
    rng = np.random.default_rng(RANDOM_STATE)

    # Mutation columns: binary
    mut = rng.binomial(n=1, p=0.12, size=(N_CELL_LINES, N_MUT))
    # CNV columns: roughly-normal log2 ratio
    cnv = rng.normal(loc=0.0, scale=0.5, size=(N_CELL_LINES, N_CNV))
    # Expression columns: log-normal-ish
    expr = rng.normal(loc=0.0, scale=1.0, size=(N_CELL_LINES, N_EXPR))

    X = np.hstack([mut, cnv, expr])
    columns = (
        [f"MUT_GENE_{i}" for i in range(N_MUT)]
        + [f"CNV_GENE_{i}" for i in range(N_CNV)]
        + [f"EXPR_GENE_{i}" for i in range(N_EXPR)]
    )
    index = [f"CL_{i:04d}" for i in range(N_CELL_LINES)]
    X_df = pd.DataFrame(X, index=index, columns=columns)
    X_df.index.name = "cell_line_id"

    # Signal: three "driver" features contribute.
    signal = (
        1.4 * X_df["MUT_GENE_0"]          # strong mutation driver
        + 0.9 * X_df["CNV_GENE_1"]         # moderate CNV driver
        + 0.7 * X_df["EXPR_GENE_2"]        # weaker expression driver
        + rng.normal(0, 0.8, N_CELL_LINES)
    )
    y = (signal > signal.median()).astype(int)
    y_df = pd.DataFrame({"sensitivity": y}, index=index)
    y_df.index.name = "cell_line_id"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_df.to_csv(OUT_DIR / "features.csv")
    y_df.to_csv(OUT_DIR / "labels.csv")

    print(f"Wrote {OUT_DIR/'features.csv'}  ({X_df.shape[0]} x {X_df.shape[1]})")
    print(f"Wrote {OUT_DIR/'labels.csv'}    ({y_df.shape[0]} rows, "
          f"class balance: {y_df['sensitivity'].value_counts().to_dict()})")


if __name__ == "__main__":
    generate()
