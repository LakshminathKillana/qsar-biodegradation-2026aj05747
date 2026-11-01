from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "qsar_biodegradation.csv"
MODEL_DIR = ROOT_DIR / "model"
RESULTS_DIR = ROOT_DIR / "results"
TEST_DATA_PATH = ROOT_DIR / "test_data.csv"
CONFIG_PATH = ROOT_DIR / "project_config.json"

TARGET_COLUMN = "target"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
    config = json.load(handle)

RANDOM_SEED = int(config["random_seed"])
TEST_SIZE = float(config["test_size"])


def calculate_metrics(y_true, y_pred, y_score):
    """Return all six evaluation metrics required by the assignment."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Run: python prepare_dataset.py"
    )

data = pd.read_csv(DATA_PATH)

if TARGET_COLUMN not in data.columns:
    raise ValueError(f"Missing target column: {TARGET_COLUMN}")

X = data.drop(columns=[TARGET_COLUMN]).copy()
y = data[TARGET_COLUMN].astype(int).copy()

if X.shape[0] < 500 or X.shape[1] < 12:
    raise ValueError(
        f"Dataset does not meet assignment minimums: "
        f"{X.shape[0]} rows, {X.shape[1]} features."
    )

if X.isna().any().any() or y.isna().any():
    raise ValueError("Missing values detected.")

if set(y.unique()) != {0, 1}:
    raise ValueError(f"Target must be binary 0/1, found: {sorted(y.unique())}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y,
)

# Save the exact held-out data used for model evaluation.
test_data = X_test.copy()
test_data[TARGET_COLUMN] = y_test.to_numpy()
test_data.to_csv(TEST_DATA_PATH, index=False)


# Every model uses the same training/test split.
# Scaling is kept inside the saved pipelines so Streamlit can evaluate raw CSV input.
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ]),
    "Decision Tree": Pipeline([
        ("classifier", DecisionTreeClassifier(
            max_depth=7,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ]),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=11,
            weights="distance",
            p=2,
        )),
    ]),
    "Gaussian Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB()),
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            C=2.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ]),
}

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

rows = []

print("=" * 88)
print("QSAR BIODEGRADATION - FIVE MODEL CLASSIFICATION BENCHMARK")
print("=" * 88)
print(f"Dataset rows       : {len(data)}")
print(f"Input features     : {X.shape[1]}")
print(f"Training rows      : {len(X_train)}")
print(f"Test rows          : {len(X_test)}")
print(f"Random seed        : {RANDOM_SEED}")

for model_name, model in models.items():
    print("\n" + "-" * 88)
    print(f"Training: {model_name}")
    print("-" * 88)

    started = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities,
    )

    rows.append({
        "ML Model Name": model_name,
        **metrics,
        "Training Time Seconds": elapsed,
    })

    joblib.dump(
        model,
        MODEL_DIR / model_files[model_name],
        compress=3,
    )

    safe_name = (
        model_name.lower()
        .replace("-", "")
        .replace(" ", "_")
    )

    cm = confusion_matrix(y_test, predictions)
    pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    ).to_csv(
        RESULTS_DIR / f"{safe_name}_confusion_matrix.csv"
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(
        RESULTS_DIR / f"{safe_name}_classification_report.csv"
    )

    for metric_name, value in metrics.items():
        print(f"{metric_name:10s}: {value:.4f}")
    print(f"{'Train Time':10s}: {elapsed:.3f} sec")


comparison = pd.DataFrame(rows)

comparison = comparison[
    [
        "ML Model Name",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "Training Time Seconds",
    ]
]

comparison.to_csv(
    RESULTS_DIR / "model_metrics.csv",
    index=False,
)

# Transparent winner rule:
# MCC is primary because it incorporates all four confusion-matrix cells.
# AUC and F1 are used as tie-breakers.
ranked = comparison.sort_values(
    by=["MCC", "AUC", "F1"],
    ascending=False,
).reset_index(drop=True)

winner = ranked.iloc[0]["ML Model Name"]

metadata = {
    "dataset_name": "UCI QSAR Biodegradation",
    "dataset_id": 254,
    "dataset_doi": "10.24432/C5H60M",
    "target_column": TARGET_COLUMN,
    "positive_class": "Ready biodegradable (RB) = 1",
    "negative_class": "Not ready biodegradable (NRB) = 0",
    "feature_names": list(X.columns),
    "feature_count": int(X.shape[1]),
    "instance_count": int(X.shape[0]),
    "random_seed": RANDOM_SEED,
    "test_size": TEST_SIZE,
    "model_files": model_files,
    "winner_rule": "Highest MCC; AUC then F1 used as tie-breakers",
    "winner": winner,
}

with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)

print("\n" + "=" * 110)
print("FINAL MODEL COMPARISON")
print("=" * 110)
print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nOverall winner by the stated ranking rule:", winner)
print("\nSaved:")
print(" - test_data.csv")
print(" - five trained .joblib model pipelines")
print(" - model/metadata.json")
print(" - results/model_metrics.csv")
print(" - confusion matrices and classification reports")
print("\nTraining and evaluation completed successfully.")
