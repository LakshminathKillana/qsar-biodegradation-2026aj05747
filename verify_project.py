from pathlib import Path
import ast
import json

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent

required_root_files = [
    "streamlit_app.py",
    "train_and_evaluate.py",
    "prepare_dataset.py",
    "generate_readme.py",
    "requirements.txt",
    "README.md",
    "test_data.csv",
]

required_model_files = [
    "logistic_regression.joblib",
    "decision_tree.joblib",
    "knn.joblib",
    "gaussian_naive_bayes.joblib",
    "support_vector_machine.joblib",
    "metadata.json",
]

required_metrics = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]

print("=" * 78)
print("STRICT PROJECT VERIFICATION")
print("=" * 78)

for file_name in required_root_files:
    path = ROOT / file_name
    assert path.exists(), f"Missing required file: {file_name}"
    print(f"PASS root file: {file_name}")

for file_name in required_model_files:
    path = ROOT / "model" / file_name
    assert path.exists(), f"Missing model artifact: {file_name}"
    print(f"PASS model file: {file_name}")

metadata_path = ROOT / "model" / "metadata.json"
with open(metadata_path, "r", encoding="utf-8") as handle:
    metadata = json.load(handle)

assert metadata["instance_count"] >= 500
assert metadata["feature_count"] >= 12
print(
    f"PASS dataset minimums: "
    f"{metadata['instance_count']} instances, "
    f"{metadata['feature_count']} features"
)

metrics_path = ROOT / "results" / "model_metrics.csv"
assert metrics_path.exists(), "Missing results/model_metrics.csv"

metrics = pd.read_csv(metrics_path)
assert len(metrics) == 5, f"Expected 5 model rows, found {len(metrics)}"

for column in required_metrics:
    assert column in metrics.columns, f"Missing metric column: {column}"
    assert metrics[column].notna().all(), f"NaN found in metric: {column}"

print("PASS five models and all six required metrics")

test_data = pd.read_csv(ROOT / "test_data.csv")
expected_columns = set(metadata["feature_names"] + [metadata["target_column"]])
assert expected_columns.issubset(test_data.columns)
assert test_data[metadata["target_column"]].nunique() == 2
print("PASS test_data.csv contains all features, target, and both classes")

for model_name, file_name in metadata["model_files"].items():
    model = joblib.load(ROOT / "model" / file_name)
    X = test_data[metadata["feature_names"]]
    preds = model.predict(X)
    assert len(preds) == len(test_data)
    print(f"PASS model load/predict: {model_name}")

# Static check: Streamlit must not retrain on uploaded data.
streamlit_source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
tree = ast.parse(streamlit_source)

fit_calls = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr in {"fit", "fit_transform"}:
            fit_calls.append((node.func.attr, getattr(node, "lineno", None)))

assert not fit_calls, f"Training call found in Streamlit app: {fit_calls}"
print("PASS Streamlit app contains no fit()/fit_transform() training calls")

readme = (ROOT / "README.md").read_text(encoding="utf-8")
required_readme_sections = [
    "Problem Statement",
    "Dataset Description",
    "GitHub Repository Link",
    "Models Used and Evaluation Metrics",
    "Comparison Table",
    "Model Performance Observations",
    "Overall, Winner for the dataset?",
]

for section in required_readme_sections:
    assert section in readme, f"README section missing: {section}"

print("PASS required README sections")
print("\nALL STRICT PROJECT CHECKS PASSED")
