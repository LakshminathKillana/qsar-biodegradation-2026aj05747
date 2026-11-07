# BioDegrade Lab — QSAR Classification Benchmark

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
| **Instances** | 1055 (exceeds requirement of >= 500) |
| **Input features** | 41 (exceeds requirement of >= 12) |
| **Feature type** | Numeric molecular descriptors |
| **Target** | Ready biodegradable (RB=1) vs Not ready biodegradable (NRB=0) |
| **Missing values** | None in the UCI dataset |
| **Train/test split** | 80% / 20% |
| **Random seed** | 5747 |
| **UCI Source URL** | https://archive.ics.uci.edu/dataset/254/qsar+biodegradation |

## c. GitHub Repository Link

- **GitHub Repository:** https://github.com/LakshminathKillana/qsar-biodegradation-2026aj05747
- **Live Streamlit App Link:** https://qsar-biodegradation-2026aj05747.streamlit.app/

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

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8910 | 0.9392 | 0.8000 | 0.9014 | 0.8477 | 0.7666 |
| Decision Tree | 0.8531 | 0.9076 | 0.7564 | 0.8310 | 0.7919 | 0.6806 |
| KNN | 0.8768 | 0.9183 | 0.7848 | 0.8732 | 0.8267 | 0.7340 |
| Naive Bayes | 0.6588 | 0.8076 | 0.4964 | 0.9577 | 0.6538 | 0.4603 |
| Support Vector Machine | 0.9005 | 0.9229 | 0.8289 | 0.8873 | 0.8571 | 0.7820 |

### Observations on Model Performance
<!-- Model Performance Observations -->

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Logistic Regression achieved Accuracy=0.8910, AUC=0.9392, Precision=0.8000, Recall=0.9014, F1=0.8477, and MCC=0.7666. It also achieved the highest AUC. |
| Decision Tree | Decision Tree achieved Accuracy=0.8531, AUC=0.9076, Precision=0.7564, Recall=0.8310, F1=0.7919, and MCC=0.6806. |
| KNN | K-Nearest Neighbors achieved Accuracy=0.8768, AUC=0.9183, Precision=0.7848, Recall=0.8732, F1=0.8267, and MCC=0.7340. |
| Naive Bayes | Gaussian Naive Bayes achieved Accuracy=0.6588, AUC=0.8076, Precision=0.4964, Recall=0.9577, F1=0.6538, and MCC=0.4603. Its recall was the highest among the five models. |
| Support Vector Machine | Support Vector Machine achieved Accuracy=0.9005, AUC=0.9229, Precision=0.8289, Recall=0.8873, F1=0.8571, and MCC=0.7820. It produced the strongest overall MCC in this experiment. Its precision was the highest among the five models. |
| **Overall, Winner for the dataset?** | **Support Vector Machine**. The winner was selected using the highest MCC, with AUC and F1 as tie-breakers. |
| **Overall, Winner for your dataset?** | **Support Vector Machine**. The winner was selected using the highest MCC, with AUC and F1 as tie-breakers. |

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
