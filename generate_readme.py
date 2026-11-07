from pathlib import Path
import json

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
METRICS_PATH = ROOT_DIR / "results" / "model_metrics.csv"
METADATA_PATH = ROOT_DIR / "model" / "metadata.json"
CONFIG_PATH = ROOT_DIR / "project_config.json"
README_PATH = ROOT_DIR / "README.md"

if not METRICS_PATH.exists():
    raise FileNotFoundError("Run train_and_evaluate.py first.")

with open(METADATA_PATH, "r", encoding="utf-8") as handle:
    metadata = json.load(handle)

with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
    config = json.load(handle)

metrics = pd.read_csv(METRICS_PATH)

display_columns = [
    "ML Model Name",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]

metrics_display = metrics[display_columns].copy()

for column in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
    metrics_display[column] = metrics_display[column].map(lambda x: f"{x:.4f}")

def markdown_table(df):
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(map(str, row.tolist())) + " |")
    return "\n".join(lines)

def observation_for(row, table):
    name = row["ML Model Name"]
    best_mcc = table["MCC"].max()
    best_auc = table["AUC"].max()
    best_recall = table["Recall"].max()
    best_precision = table["Precision"].max()

    notes = [
        f"{name} achieved Accuracy={row['Accuracy']:.4f}, "
        f"AUC={row['AUC']:.4f}, Precision={row['Precision']:.4f}, "
        f"Recall={row['Recall']:.4f}, F1={row['F1']:.4f}, "
        f"and MCC={row['MCC']:.4f}."
    ]

    if row["MCC"] == best_mcc:
        notes.append("It produced the strongest overall MCC in this experiment.")
    if row["AUC"] == best_auc:
        notes.append("It also achieved the highest AUC.")
    if row["Recall"] == best_recall:
        notes.append("Its recall was the highest among the five models.")
    if row["Precision"] == best_precision:
        notes.append("Its precision was the highest among the five models.")

    return " ".join(notes)

observations = {}
for _, row in metrics.iterrows():
    observations[row["ML Model Name"]] = observation_for(row, metrics)

def model_name_format(name):
    if name == "K-Nearest Neighbors":
        return "KNN"
    if name == "Gaussian Naive Bayes":
        return "Naive Bayes"
    return name

metrics_display_table = metrics_display.copy()
metrics_display_table["ML Model Name"] = metrics_display_table["ML Model Name"].map(model_name_format)

observations_table = {}
for name, obs in observations.items():
    observations_table[model_name_format(name)] = obs

winner = metadata["winner"]

readme = f"""# BioDegrade Lab — QSAR Classification Benchmark

## a. Problem Statement

The objective of this project is to compare five supervised machine-learning
classifiers for predicting whether a chemical is ready biodegradable from
molecular-descriptor data.

The project trains all five required classifiers on the same training split,
evaluates them on one held-out test split, saves the trained models, and exposes
the saved models through an interactive Streamlit application. The Streamlit
application evaluates uploaded test data without retraining the models.

## b. Dataset Description

| Property | Details |
| --- | --- |
| **Dataset** | UCI QSAR Biodegradation |
| **UCI Dataset ID** | 254 |
| **DOI** | 10.24432/C5H60M |
| **Task** | Binary classification |
| **Instances** | {metadata['instance_count']} (exceeds requirement of >= 500) |
| **Input features** | {metadata['feature_count']} (exceeds requirement of >= 12) |
| **Feature type** | Numeric molecular descriptors |
| **Target** | Ready biodegradable (RB=1) vs Not ready biodegradable (NRB=0) |
| **Missing values** | None in the UCI dataset |
| **Train/test split** | {(1-metadata['test_size'])*100:.0f}% / {metadata['test_size']*100:.0f}% |
| **Random seed** | {metadata['random_seed']} |
| **UCI Source URL** | https://archive.ics.uci.edu/dataset/254/qsar+biodegradation |

## c. GitHub Repository Link

- **GitHub Repository:** {config['github_url']}
- **Live Streamlit App Link:** {config['streamlit_url']}

## d. Models Used

<!-- Models Used and Evaluation Metrics -->
The five required classification models implemented on the same dataset are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (KNN)
4. Naive Bayes Classifier (Gaussian Naive Bayes)
5. Support Vector Machine

All models were evaluated using the following six required evaluation metrics:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC Score)

### Comparison of Evaluation Metrics
<!-- Comparison Table -->

{markdown_table(metrics_display_table)}

### Observations on Model Performance
<!-- Model Performance Observations -->

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | {observations_table['Logistic Regression']} |
| Decision Tree | {observations_table['Decision Tree']} |
| KNN | {observations_table['KNN']} |
| Naive Bayes | {observations_table['Naive Bayes']} |
| Support Vector Machine | {observations_table['Support Vector Machine']} |
| **Overall, Winner for the dataset?** | **{winner}**. The winner was selected using the highest MCC, with AUC and F1 as tie-breakers. |
| **Overall, Winner for your dataset?** | **{winner}**. The winner was selected using the highest MCC, with AUC and F1 as tie-breakers. |

## Streamlit Application Features

The production-ready Streamlit application provides an interactive benchmark dashboard:

- **Executive Header & State Indicators:** High-contrast header with dynamic status pills showing models loaded, active evaluation dataset, and selected classifier.
- **Light High-Readability Sidebar:** Clean light blue-grey (`#e8edf5`) sidebar layout with an indigo border accent (`#2563eb`), offering clear contrast, dark text labels, and immediate top-level model selection without vertical scrolling.
- **Dataset Management & Overview:** Sidebar sample test CSV download button, custom CSV uploader, and dataset summary metric cards (total samples, feature count, class balance breakdown).
- **Single-Model Evaluation Suite:**
  - 6 key performance cards (Accuracy, AUC, Precision, Recall, F1, MCC) equipped with top-accent color coding and delta benchmarking against the dataset's top performer.
  - Balanced side-by-side Confusion Matrix (counts & percentage breakdowns) and individual ROC Curve with AUC metric and random baseline.
  - Full-width, horizontal-scroll-free Classification Report table with descriptive class names (`Class 0 — NRB`, `Class 1 — RB`, macro/weighted averages).
  - One-click export for model predictions (`.csv`).
- **All-Models Comparative Analysis:**
  - Clean corporate Metrics Table with highlighted top-performers, active model selection border, and CSV export.
  - Multi-model grouped bar chart comparison across all 6 metrics with high-res PNG export.
  - Combined multi-model ROC Curves overlay with high-res PNG export.
  - Multi-axis Radar/Spider chart comparison with PNG export.
- **Automated Winner Callout & Metadata:** Prominent badge highlight recognizing the best-performing model based on MCC → AUC → F1 hierarchy, paired with assignment footer metadata.

The uploaded test data contains all model input features and the target column.
The application loads the saved `.joblib` pipelines and evaluates them without
retraining.

## Project Structure

```text
qsar_biodegradation_ml_2026AJ05747/
├── prepare_dataset.py
├── train_and_evaluate.py
├── streamlit_app.py
├── generate_readme.py
├── verify_project.py
├── requirements.txt
├── project_config.json
├── README.md
├── test_data.csv
├── data/
│   └── qsar_biodegradation.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── gaussian_naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   └── metadata.json
└── results/
    ├── model_metrics.csv
    ├── *_confusion_matrix.csv
    └── *_classification_report.csv
```

## Running the Project

```bash
python -m pip install -r requirements.txt
python prepare_dataset.py
python train_and_evaluate.py
python generate_readme.py
python verify_project.py
streamlit run streamlit_app.py
```

## Reproducibility

The same held-out test set is used for all five classifiers. The file
`test_data.csv` contains the exact test observations used for model evaluation.
The trained models are saved before deployment, and the Streamlit app performs
evaluation only; it does not call model training on uploaded data.
"""

README_PATH.write_text(readme, encoding="utf-8")
print(f"Generated README: {README_PATH}")
print(f"Winner: {winner}")
