"""
Proof of Concept: Predict Olaparib (PARP inhibitor) sensitivity from
cancer cell-line mutation profiles.

Data sources (selected due to network sandbox blocking direct Sanger host;
see `docs/data_provenance` in commit message / README for details):

  * GDSC mutation matrix (binary, 878 cell lines x 534 cancer-related genes):
      https://raw.githubusercontent.com/hanjunwei-lab/DeepCCDS/main/data/GDSC_mutation_input.csv
  * GDSC IC50 train/valid/test splits (combined -> full set of drug-cell
    IC50 values, including Olaparib):
      https://raw.githubusercontent.com/hanjunwei-lab/DeepCCDS/main/data/GDSC_train_IC50_by_borh_cv00.csv
      https://raw.githubusercontent.com/hanjunwei-lab/DeepCCDS/main/data/GDSC_valid_IC50_by_borh_cv00.csv
      https://raw.githubusercontent.com/hanjunwei-lab/DeepCCDS/main/data/GDSC_test_IC50_by_borh_cv00.csv

These are redistributions of GDSC / Sanger-Cancer-Rx-Gene data
(public, CC-BY for GDSC). The original release-specific files at
`https://cog.sanger.ac.uk/cancerrxgene/...` were unreachable from this
sandboxed environment (HTTP 403: host_not_allowed), so we used a published
ML-reproducibility mirror on GitHub which is GDSC-derived.

Scientific question:
  Can mutation-only features predict Olaparib sensitivity across cancer
  cell lines?

Pipeline:
  1. Load mutation matrix (cell line x gene, binary).
  2. Load IC50 data, filter to DRUG_NAME == 'Olaparib'.
  3. Build feature matrix X restricted to ~50 commonly cited cancer /
     DDR / HRD genes (superset: TP53, KRAS, BRCA1, BRCA2, ATM, ATR,
     CHEK2, PALB2, RAD51B, BARD1, BRIP1, FANCA, ..., plus common
     drivers).
  4. Binarize IC50: bottom 25% -> sensitive (1), top 25% -> resistant (0),
     drop middle 50%.
  5. Stratified 5-fold CV for Logistic Regression, Random Forest
     (n_estimators=200), and Gradient Boosting; report mean +/- std AUC.
  6. Refit the best model on all data, compute SHAP values.
  7. Save CV metrics, top-10 SHAP features, SHAP bar plot, and a
     summary text file.

Reproduce:
  python scripts/poc/run_poc.py

Output (results/poc/):
  olaparib_cv_metrics.csv
  olaparib_shap_top.csv
  shap_summary.png
  poc_summary.txt
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "poc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MUT_CSV = DATA_DIR / "GDSC_mutation_input.csv"
IC50_CSVS = [
    DATA_DIR / "GDSC_train_IC50.csv",
    DATA_DIR / "GDSC_valid_IC50.csv",
    DATA_DIR / "GDSC_test_IC50.csv",
]

# Curated gene panel: DNA-damage-response / HRD + top pan-cancer drivers.
GENE_PANEL = [
    # HRD / PARPi-relevant
    "BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2", "PALB2",
    "RAD51", "RAD51B", "RAD51C", "RAD51D", "BARD1", "BRIP1",
    "FANCA", "FANCC", "FANCD2", "FANCE", "FANCF", "FANCG", "FANCI",
    "FANCL", "FANCM", "MRE11A", "NBN", "XRCC2", "XRCC3",
    # Mismatch repair
    "MLH1", "MSH2", "MSH6", "PMS2",
    # Pan-cancer drivers
    "TP53", "KRAS", "NRAS", "HRAS", "PIK3CA", "PTEN", "APC", "CTNNB1",
    "EGFR", "BRAF", "IDH1", "IDH2", "NF1", "RB1", "VHL", "SMAD4",
    "STK11", "CDKN2A", "MYC",
]

RANDOM_SEED = 42
N_SPLITS = 5


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MUT_CSV.exists():
        raise FileNotFoundError(
            f"Mutation file not found: {MUT_CSV}. "
            "Download it first (see docstring at top of this script)."
        )
    mut = pd.read_csv(MUT_CSV)
    mut = mut.drop(columns=[c for c in ("cell_idx",) if c in mut.columns])

    ic50_parts = []
    for p in IC50_CSVS:
        if not p.exists():
            raise FileNotFoundError(f"IC50 split not found: {p}")
        ic50_parts.append(pd.read_csv(p))
    ic50 = pd.concat(ic50_parts, ignore_index=True)
    return mut, ic50


def build_feature_matrix(mut: pd.DataFrame, ic50: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.Series, list[str]
]:
    olap = ic50[ic50["drug_name"].str.lower() == "olaparib"].copy()
    # Average IC50 if a cell line appears multiple times for Olaparib.
    olap_ic50 = olap.groupby("cell_name", as_index=True)["IC50"].mean()

    # Select panel genes present in mutation matrix
    genes_present = [g for g in GENE_PANEL if g in mut.columns]
    genes_missing = [g for g in GENE_PANEL if g not in mut.columns]

    feat = mut[["cell_line"] + genes_present].set_index("cell_line")
    feat = feat.loc[feat.index.intersection(olap_ic50.index)]
    olap_ic50 = olap_ic50.loc[feat.index]

    # Binarize: bottom 25% sensitive (1), top 25% resistant (0), drop middle 50%
    q25 = olap_ic50.quantile(0.25)
    q75 = olap_ic50.quantile(0.75)
    sens_mask = olap_ic50 <= q25
    resist_mask = olap_ic50 >= q75

    keep = sens_mask | resist_mask
    y = pd.Series(
        np.where(sens_mask[keep], 1, 0),
        index=olap_ic50.index[keep],
        name="label",
    )
    X = feat.loc[y.index]
    return X, y, genes_missing


def cross_validate_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            solver="liblinear",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_SEED,
        ),
    }

    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED
    )
    records = []
    for name, model in models.items():
        aucs = []
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]
            if name == "LogisticRegression":
                scaler = StandardScaler(with_mean=False)
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                model.fit(X_tr_s, y_tr)
                proba = model.predict_proba(X_te_s)[:, 1]
            else:
                model.fit(X_tr, y_tr)
                proba = model.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, proba)
            aucs.append(auc)
        records.append(
            {
                "model": name,
                "mean_auc": float(np.mean(aucs)),
                "std_auc": float(np.std(aucs)),
                "fold_aucs": ";".join(f"{a:.4f}" for a in aucs),
            }
        )
    return pd.DataFrame(records)


def shap_top_features(
    best_model_name: str, X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, np.ndarray]:
    if best_model_name == "LogisticRegression":
        scaler = StandardScaler(with_mean=False)
        X_s = scaler.fit_transform(X)
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            solver="liblinear",
        )
        model.fit(X_s, y)
        explainer = shap.LinearExplainer(model, X_s)
        shap_values = explainer.shap_values(X_s)
    elif best_model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # sklearn RF returns (n_samples, n_features, n_classes) or list
        if isinstance(sv, list):
            shap_values = sv[1]
        else:
            shap_values = np.asarray(sv)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
    else:  # GradientBoosting
        model = GradientBoostingClassifier(random_state=RANDOM_SEED)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        shap_values = np.asarray(sv)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    top = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return top.head(10), shap_values


def save_shap_bar(top: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#4878D0")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("Top 10 mutation features\n(Olaparib sensitivity, GDSC)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    print(">> Loading data ...")
    mut, ic50 = load_data()
    print(
        f"   mutation matrix: {mut.shape[0]} cell lines x "
        f"{mut.shape[1] - 1} genes"
    )
    print(f"   IC50 rows: {len(ic50)}")

    print(">> Building feature matrix for Olaparib ...")
    X, y, missing = build_feature_matrix(mut, ic50)
    n_total = len(y)
    n_sens = int((y == 1).sum())
    n_res = int((y == 0).sum())
    print(f"   cell lines used: {n_total} (sensitive={n_sens}, resistant={n_res})")
    print(f"   panel genes used: {X.shape[1]}")
    if missing:
        print(f"   panel genes missing from source data: {missing}")

    print(">> Running 5-fold stratified CV for 3 models ...")
    cv = cross_validate_models(X, y)
    print(cv.to_string(index=False))

    cv.to_csv(RESULTS_DIR / "olaparib_cv_metrics.csv", index=False)

    best_row = cv.sort_values("mean_auc", ascending=False).iloc[0]
    best_name = best_row["model"]
    print(f">> Best model: {best_name} (mean AUC={best_row['mean_auc']:.4f})")

    print(">> Computing SHAP values for best model ...")
    top, _ = shap_top_features(best_name, X, y)
    top.to_csv(RESULTS_DIR / "olaparib_shap_top.csv", index=False)
    save_shap_bar(top, RESULTS_DIR / "shap_summary.png")
    print(top.to_string(index=False))

    # Summary file
    lines = [
        "GDSC Olaparib sensitivity - Proof of Concept",
        "=" * 60,
        "",
        "Scientific question:",
        "  Can mutation features predict Olaparib sensitivity across",
        "  cancer cell lines?",
        "",
        "Data:",
        "  mutation matrix: GDSC binary mutation matrix",
        f"  cell lines with Olaparib IC50 + mutation profile: {n_total}",
        f"  class balance: sensitive={n_sens} / resistant={n_res}",
        f"  panel genes used: {X.shape[1]} (DDR + HRD + pan-cancer drivers)",
        f"  panel genes missing: {missing or 'none'}",
        "",
        "Labelling:",
        "  sensitive = IC50 in bottom 25%",
        "  resistant = IC50 in top 25%",
        "  middle 50% dropped to sharpen the signal",
        "",
        "Cross-validation AUCs (5-fold stratified):",
    ]
    for _, r in cv.iterrows():
        lines.append(
            f"  {r['model']:<18s} {r['mean_auc']:.4f} +/- {r['std_auc']:.4f}"
        )
    lines += [
        "",
        f"Best model: {best_name}",
        "",
        "Top 10 SHAP features (|SHAP| mean):",
    ]
    for _, r in top.iterrows():
        lines.append(f"  {r['feature']:<10s} {r['mean_abs_shap']:.4f}")

    best_auc = float(best_row["mean_auc"])
    lines += ["", "Honest assessment:"]
    if best_auc > 0.65:
        lines.append(
            f"  Best CV AUC = {best_auc:.3f} > 0.65. Meaningful signal."
        )
    elif best_auc >= 0.60:
        lines.append(
            f"  Best CV AUC = {best_auc:.3f} (0.60-0.65). Weak-to-moderate "
            "signal; borderline usable."
        )
    else:
        lines.append(
            f"  Best CV AUC = {best_auc:.3f} < 0.60. WEAK signal."
        )
        lines.append(
            "  Consistent with the literature: individual mutation markers "
            "(even BRCA1/2) are weak Olaparib predictors in cancer cell "
            "lines. HRD scores and transcriptomic features typically "
            "outperform mutation-only features."
        )
    lines += [
        "",
        "Caveats:",
        "  * Small sample (~400 cell lines after top/bottom 25% filtering).",
        "  * Class definition by quantile is arbitrary vs. clinical cut-offs.",
        "  * Mutation-only features ignore CNA, methylation, and expression.",
        "  * Pan-cancer: not stratified by tissue.",
        "  * Original Sanger hosts (cog.sanger.ac.uk) were unreachable from "
        "this sandbox; GDSC-derived mutation/IC50 files from the DeepCCDS "
        "mirror on GitHub were used instead (see script docstring).",
    ]

    with open(RESULTS_DIR / "poc_summary.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(">> Done. Artifacts written to", RESULTS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
