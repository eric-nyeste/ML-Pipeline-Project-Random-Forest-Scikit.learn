#!/usr/bin/env python3
"""CSC 659/859 HW2: Random Forest classification of single-cell gene expression.

Consolidated from the Python appendices (pp. 29-46, 49-51) of the author's
51-page report. The original experimental settings and methods are retained.
Integration changes: portable paths, shared imports/functions, a CLI, and
noninteractive chart saving. This is NOT a byte-for-byte reproduction of the
separate scripts, and the original experiment has not been rerun here.

Dependencies: numpy, pandas, scikit-learn, matplotlib.
Example:
    python HW2_ML_Pipeline_Consolidated.py --data-dir ./data
    python HW2_ML_Pipeline_Consolidated.py --data-dir ./data --use-existing-split

Expected input: data/Original training DB e1 cluster.csv (871 rows, Label column)
or, with --use-existing-split: data/training_db.csv and data/verification_db.csv.
Outputs are written to --output-dir (defaults to --data-dir).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

SEED = 42
LABEL = "Label"
TREE_COUNTS = (100, 500, 1000)
FEATURE_SETTINGS = ("sqrt", 0.5)
REDUCED_FEATURE_COUNTS = (4, 6, 10, 608)
REFERENCE_GENES = ("TESPA1", "LINC00507", "SLC17A7", "KCNIP1")
RUNTIME_GENES = ("SLC17A7", "KCNIP1", "TESPA1", "LINC00152", "SFTA1P", "LINC00507")


def forest(n_estimators=100, max_features="sqrt", oob_score=False):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_features=max_features,
        bootstrap=True, oob_score=oob_score, random_state=SEED, n_jobs=-1,
    )


def split_original(data_dir):
    """Appendix 3.1: withhold one randomly sampled row from each class."""
    source = data_dir / "Original training DB e1 cluster.csv"
    df = pd.read_csv(source)
    sample_1 = df[df[LABEL] == 1].sample(n=1, random_state=SEED)
    sample_0 = df[df[LABEL] == 0].sample(n=1, random_state=SEED)
    verification = pd.concat([sample_1, sample_0])
    training = df.drop(index=verification.index)
    assert len(training) + len(verification) == len(df)
    assert training.index.intersection(verification.index).empty
    training.to_csv(data_dir / "training_db.csv", index=False)
    verification.to_csv(data_dir / "verification_db.csv", index=False)
    print("Original:", df.shape, "Training:", training.shape,
          "Verification:", verification.shape)
    print("Training class counts:", training[LABEL].value_counts().sort_index().to_dict())
    print("Verification class counts:", verification[LABEL].value_counts().sort_index().to_dict())
    print("Original Excel row numbers:", (verification.index + 2).tolist())
    return training, verification


def load_split(data_dir):
    return (pd.read_csv(data_dir / "training_db.csv"),
            pd.read_csv(data_dir / "verification_db.csv"))


def validate(training, verification):
    """Appendix 5.1: inspect features, labels, and missing values."""
    for name, df in (("training", training), ("verification", verification)):
        if LABEL not in df:
            raise ValueError(f"{name} data missing '{LABEL}' column")
        if set(df[LABEL].unique()) != {0, 1}:
            raise ValueError(f"{name} labels must contain both 0 and 1")
        if df.drop(columns=[LABEL]).isna().any().any():
            raise ValueError(f"{name} features contain missing values")
    X, y = training.drop(columns=[LABEL]), training[LABEL]
    print("Feature matrix:", X.shape, "Labels:", y.shape,
          "Classes:", sorted(y.unique()), "Missing values:", int(X.isna().sum().sum()))
    return X, y


def metrics(y, predicted):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(
        accuracy=accuracy_score(y, predicted),
        e1_precision=precision_score(y, predicted, pos_label=1, zero_division=0),
        e1_recall=recall_score(y, predicted, pos_label=1, zero_division=0),
        e1_F1=f1_score(y, predicted, pos_label=1, zero_division=0),
        non_e1_precision=precision_score(y, predicted, pos_label=0, zero_division=0),
        non_e1_recall=recall_score(y, predicted, pos_label=0, zero_division=0),
        non_e1_F1=f1_score(y, predicted, pos_label=0, zero_division=0),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
    )


def oob_experiment(X, y, out):
    """Appendices 6.1-6.2: single-model report and six OOB configurations."""
    rows = []
    for ntree in TREE_COUNTS:
        for mtry in FEATURE_SETTINGS:
            print(f"OOB: {ntree} trees; max_features={mtry}")
            rf = forest(ntree, mtry, oob_score=True)
            rf.fit(X, y)
            predictions = rf.classes_[rf.oob_decision_function_.argmax(axis=1)]
            result = metrics(y, predictions)
            rows.append({"ntree": ntree, "mtry": str(mtry),
                         "OOB_accuracy": result.pop("accuracy"), **result})
            if ntree == 100 and mtry == "sqrt":
                print("OOB score:", rf.oob_score_)
                print("OOB confusion matrix:\n", confusion_matrix(y, predictions, labels=[0, 1]))
                print(classification_report(y, predictions, digits=5))
    df = pd.DataFrame(rows)
    df.to_csv(out / "HW2_OOB_results.csv", index=False)
    print("\nSix-model OOB comparison:\n", df.to_string(index=False, float_format="%.5f"))
    return df


def stratified_folds(y):
    """Appendices 6.3 and 7.2: manually construct three stratified folds."""
    rng = np.random.default_rng(SEED)
    indices_0 = np.where(y.to_numpy() == 0)[0]
    indices_1 = np.where(y.to_numpy() == 1)[0]
    rng.shuffle(indices_0)
    rng.shuffle(indices_1)
    parts_0 = np.array_split(indices_0, 3)
    parts_1 = np.array_split(indices_1, 3)
    folds = []
    for i in range(3):
        fold = np.concatenate([parts_0[i], parts_1[i]])
        rng.shuffle(fold)
        folds.append(fold)
    all_indices = np.concatenate(folds)
    assert len(all_indices) == len(y)
    assert len(np.unique(all_indices)) == len(y)
    for i, fold in enumerate(folds, 1):
        print(f"Fold {i}: {len(fold)} samples; class counts",
              y.iloc[fold].value_counts().to_dict())
    return folds


def cross_validation(X, y, folds, out):
    """Appendix 6.3: manually evaluated 3-fold RF, with pooled predictions."""
    rows = []
    predictions = np.full(len(y), -1, dtype=int)
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(3) if j != i])
        rf = forest(100, "sqrt", oob_score=True)
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = rf.predict(X.iloc[test_idx])
        predictions[test_idx] = pred
        rows.append(dict(fold=i + 1, train_samples=len(train_idx),
                         test_samples=len(test_idx), **metrics(y.iloc[test_idx], pred)))
    assert np.all(predictions != -1)
    df = pd.DataFrame(rows)
    metric_cols = ["accuracy", "e1_precision", "e1_recall", "e1_F1",
                   "non_e1_precision", "non_e1_recall", "non_e1_F1"]
    summary = pd.DataFrame({"mean": df[metric_cols].mean(), "std": df[metric_cols].std()})
    pooled = confusion_matrix(y, predictions, labels=[0, 1])
    df.to_csv(out / "HW2_CV_fold_results.csv", index=False)
    summary.to_csv(out / "HW2_CV_summary.csv")
    pd.DataFrame(pooled, index=["Actual non-e1", "Actual e1"],
                 columns=["Predicted non-e1", "Predicted e1"]).to_csv(
                     out / "HW2_CV_confusion_matrix.csv")
    print("\nCV folds:\n", df.to_string(index=False, float_format="%.5f"))
    print("\nCV summary:\n", summary.to_string(float_format="%.5f"))
    print("\nPooled CV confusion matrix:\n", pooled)
    print("Pooled accuracy:", accuracy_score(y, predictions),
          "e1 F1:", f1_score(y, predictions, pos_label=1),
          "non-e1 F1:", f1_score(y, predictions, pos_label=0))
    return df


def rank_features(X, y, out):
    """Appendix 7.1: Gini importance from full-training-set selected RF."""
    rf = forest(100, "sqrt", oob_score=True)
    rf.fit(X, y)
    importance = pd.DataFrame({"Gene": X.columns,
                               "Gini_Importance": rf.feature_importances_})
    importance = importance.sort_values("Gini_Importance", ascending=False).reset_index(drop=True)
    importance["Rank"] = importance.index + 1
    top = importance.head(10)
    print("\nTop ten genes:\n", top.to_string(index=False, float_format="%.5f"))
    for gene in REFERENCE_GENES:
        found = importance[importance.Gene == gene]
        print(f"Reference gene {gene}:\n", found.to_string(index=False) if not found.empty else "not found")
    plt.figure(figsize=(10, 6))
    plt.barh(top["Gene"][::-1], top["Gini_Importance"][::-1])
    plt.xlabel("Gini Feature Importance")
    plt.ylabel("Gene")
    plt.title("Top 10 Random Forest Features: e1 Classification")
    plt.tight_layout()
    plt.savefig(out / "HW2_RF_Top10_Features.png", dpi=300)
    plt.close()
    importance.to_csv(out / "HW2_RF_Feature_Ranking.csv", index=False)
    return importance


def reduced_features(X, y, folds, out):
    """Appendix 7.2: rank within each training fold to avoid feature leakage."""
    if X.shape[1] != 608:
        raise ValueError("Original reduced-feature experiment expects 608 gene features")
    rows, selected_gene_records = [], []
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(3) if j != i])
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        ranking_rf = forest(100, "sqrt")
        ranking_rf.fit(X_train, y_train)
        ranking = pd.Series(ranking_rf.feature_importances_,
                            index=X_train.columns).sort_values(ascending=False)
        for count in REDUCED_FEATURE_COUNTS:
            genes = X.columns.tolist() if count == X.shape[1] else ranking.head(count).index.tolist()
            if count != X.shape[1]:
                selected_gene_records.extend(dict(fold=i + 1, feature_count=count,
                                                  rank=rank, gene=gene)
                                             for rank, gene in enumerate(genes, 1))
            rf = forest(100, "sqrt")
            rf.fit(X_train[genes], y_train)
            pred = rf.predict(X_test[genes])
            m = metrics(y_test, pred)
            rows.append(dict(features=count, fold=i + 1, train_samples=len(train_idx),
                             test_samples=len(test_idx), accuracy=m["accuracy"],
                             e1_precision=m["e1_precision"], e1_recall=m["e1_recall"],
                             e1_F1=m["e1_F1"], non_e1_F1=m["non_e1_F1"],
                             TN=m["TN"], FP=m["FP"], FN=m["FN"], TP=m["TP"]))
    df = pd.DataFrame(rows).sort_values(["features", "fold"])
    cols = ["accuracy", "e1_precision", "e1_recall", "e1_F1", "non_e1_F1"]
    summary = df.groupby("features")[cols].agg(["mean", "std"])
    pooled_rows = []
    for count in REDUCED_FEATURE_COUNTS:
        subset = df[df.features == count]
        tn, fp, fn, tp = [int(subset[c].sum()) for c in ("TN", "FP", "FN", "TP")]
        total = tn + fp + fn + tp
        pooled_rows.append(dict(features=count, TN=tn, FP=fp, FN=fn, TP=tp,
                                total=total, pooled_accuracy=(tp + tn) / total,
                                pooled_e1_F1=2 * tp / (2 * tp + fp + fn)))
    pooled = pd.DataFrame(pooled_rows)
    df.to_csv(out / "HW2_Reduced_Feature_Folds.csv", index=False)
    summary.to_csv(out / "HW2_Reduced_Feature_Summary.csv")
    pooled.to_csv(out / "HW2_Reduced_Feature_Pooled.csv", index=False)
    pd.DataFrame(selected_gene_records).to_csv(out / "HW2_Reduced_Feature_Genes.csv", index=False)
    print("\nReduced-feature fold results:\n", df.to_string(index=False, float_format="%.5f"))
    print("\nReduced-feature summary:\n", summary.to_string(float_format="%.5f"))
    print("\nReduced-feature pooled results:\n", pooled.to_string(index=False, float_format="%.5f"))
    return df


def runtime_verify(training, verification, out):
    """Appendix 8.1: six preselected genes, 100-tree RF, two held-out cases."""
    for gene in RUNTIME_GENES:
        if gene not in training.columns or gene not in verification.columns:
            raise ValueError(f"Missing runtime gene: {gene}")
    rf = forest(100, "sqrt", oob_score=True)
    rf.fit(training[list(RUNTIME_GENES)], training[LABEL])
    predictions = rf.predict(verification[list(RUNTIME_GENES)])
    probabilities = rf.predict_proba(verification[list(RUNTIME_GENES)])
    idx0 = list(rf.classes_).index(0)
    idx1 = list(rf.classes_).index(1)
    result = pd.DataFrame({"Ground_Truth": verification[LABEL].to_numpy(),
                           "Predicted_Class": predictions,
                           "P_non_e1": probabilities[:, idx0],
                           "P_e1": probabilities[:, idx1],
                           "Correct": predictions == verification[LABEL].to_numpy()})
    result.to_csv(out / "HW2_Runtime_Verification.csv", index=False)
    print("\nFinal runtime verification:\n", result.to_string(index=False, float_format="%.5f"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Folder containing input CSV files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Folder for result files")
    parser.add_argument("--use-existing-split", action="store_true",
                        help="Use existing training_db.csv and verification_db.csv; do not repartition original")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (args.output_dir or data_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training, verification = (load_split(data_dir) if args.use_existing_split else split_original(data_dir))
    X, y = validate(training, verification)
    oob_experiment(X, y, output_dir)
    folds = stratified_folds(y)
    cross_validation(X, y, folds, output_dir)
    rank_features(X, y, output_dir)
    reduced_features(X, y, folds, output_dir)
    runtime_verify(training, verification, output_dir)
    print("\nPipeline complete. Results saved to:", output_dir)


if __name__ == "__main__":
    main()
