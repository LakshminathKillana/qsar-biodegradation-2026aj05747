from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

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


st.set_page_config(
    page_title="BioDegrade Lab",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.45rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #eef7f2, #f6f7fb);
        border: 1px solid #d9e7df;
        margin-bottom: 0.8rem;
    }

    .hero h1 {
        margin-bottom: 0.35rem;
        font-size: 2.05rem;
    }

    .hero p {
        margin-bottom: 0;
        color: #475569;
        font-size: 1rem;
        line-height: 1.55;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 0.75rem 0.85rem;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    .small-note {
        color: #5a6570;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .section-note {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: -0.35rem;
        margin-bottom: 0.8rem;
    }

    .footer-note {
        color: #64748b;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
METADATA_PATH = MODEL_DIR / "metadata.json"
TEST_DATA_PATH = ROOT_DIR / "test_data.csv"

if not METADATA_PATH.exists():
    st.error("Model metadata is missing. Run train_and_evaluate.py first.")
    st.stop()

with open(METADATA_PATH, "r", encoding="utf-8") as handle:
    metadata = json.load(handle)

TARGET_COLUMN = metadata["target_column"]
FEATURE_NAMES = metadata["feature_names"]
MODEL_FILES = metadata["model_files"]


@st.cache_resource
def load_models():
    loaded = {}
    for model_name, file_name in MODEL_FILES.items():
        model_path = MODEL_DIR / file_name
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {file_name}")
        loaded[model_name] = joblib.load(model_path)
    return loaded


def calculate_metrics(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    metrics = calculate_metrics(y, y_pred, y_score)
    return y_pred, metrics


models = load_models()

st.markdown(
    """
    <div class="hero">
      <h1>🧪 BioDegrade Lab</h1>
      <p>
        Interactive evaluation of five machine-learning classifiers for
        predicting ready biodegradability from QSAR molecular descriptors.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "UCI QSAR Biodegradation • 41 molecular descriptors • "
    "5 trained classifiers • saved-model evaluation only"
)

with st.expander("About this evaluation"):
    st.write(
        "The app evaluates previously trained classifiers on uploaded "
        "held-out test data. It does not retrain any model after upload."
    )
    st.write(
        "Target: 1 = ready biodegradable (RB), "
        "0 = not ready biodegradable (NRB)."
    )

if TEST_DATA_PATH.exists():
    st.download_button(
        "Download the bundled test_data.csv",
        data=TEST_DATA_PATH.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )

st.subheader("Test Data")

st.markdown(
    '<div class="section-note">'
    'Upload the held-out CSV used for evaluation. '
    'The file must contain all 41 input features and the target column.'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload test CSV",
    type=["csv"],
    help="Use the generated test_data.csv file for this evaluation.",
)

if uploaded_file is None:
    st.info("Upload test_data.csv to start the evaluation.")
    st.stop()

try:
    uploaded = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

required_columns = set(FEATURE_NAMES + [TARGET_COLUMN])
missing_columns = sorted(required_columns - set(uploaded.columns))

if missing_columns:
    st.error(
        "Uploaded CSV is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

extra_columns = sorted(set(uploaded.columns) - required_columns)
if extra_columns:
    st.warning(
        "Extra columns will be ignored: " + ", ".join(extra_columns)
    )

X_uploaded = uploaded[FEATURE_NAMES].copy()
y_uploaded = uploaded[TARGET_COLUMN].astype(int).copy()

if set(y_uploaded.unique()) - {0, 1}:
    st.error("The target column must contain only 0 and 1.")
    st.stop()

if y_uploaded.nunique() < 2:
    st.error("Both target classes are required to calculate AUC.")
    st.stop()

st.subheader("Uploaded Test Data Summary")
summary_columns = st.columns(4)
summary_columns[0].metric("Test Rows", len(uploaded))
summary_columns[1].metric("Input Features", len(FEATURE_NAMES))
summary_columns[2].metric("NRB Samples", int((y_uploaded == 0).sum()))
summary_columns[3].metric("RB Samples", int((y_uploaded == 1).sum()))

st.caption(
    "Target classes: 0 = Not Ready Biodegradable (NRB), "
    "1 = Ready Biodegradable (RB)."
)

with st.expander("Preview uploaded test data", expanded=False):
    st.dataframe(
        uploaded.head(8),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.subheader("Model Evaluation")

st.markdown(
    '<div class="section-note">'
    'Select any trained classifier to view its evaluation results '
    'on the uploaded test data.'
    '</div>',
    unsafe_allow_html=True,
)

selected_model_name = st.selectbox(
    "Choose a classification model",
    options=list(models.keys()),
)

selected_model = models[selected_model_name]
selected_predictions, selected_metrics = evaluate(
    selected_model,
    X_uploaded,
    y_uploaded,
)

st.markdown(f"### Results — {selected_model_name}")

metric_order = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
metric_cols = st.columns(6)

for col, metric_name in zip(metric_cols, metric_order):
    col.metric(metric_name, f"{selected_metrics[metric_name]:.4f}")

left, right = st.columns([1, 1.2])

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_uploaded, selected_predictions)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    image = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Predicted NRB", "Predicted RB"])
    ax.set_yticks([0, 1], labels=["Actual NRB", "Actual RB"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"{selected_model_name} — Confusion Matrix")

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.caption(
        "NRB = Not Ready Biodegradable • RB = Ready Biodegradable"
    )

with right:
    st.subheader("Classification Report")
    report = classification_report(
        y_uploaded,
        selected_predictions,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.round(4)
    st.dataframe(report_df, use_container_width=True)

st.divider()
st.header("All Five Models Comparison")

st.markdown(
    '<div class="section-note">'
    'All five saved classifiers are evaluated on the same uploaded '
    'test data for a direct comparison.'
    '</div>',
    unsafe_allow_html=True,
)

comparison_rows = []

for model_name, model in models.items():
    _, metrics = evaluate(model, X_uploaded, y_uploaded)
    comparison_rows.append(
        {
            "ML Model Name": model_name,
            **metrics,
        }
    )

comparison = pd.DataFrame(comparison_rows)

comparison = comparison[
    ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
]

for column in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
    comparison[column] = comparison[column].round(4)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True,
)

winner = (
    comparison
    .sort_values(["MCC", "AUC", "F1"], ascending=False)
    .iloc[0]["ML Model Name"]
)

st.success(
    f"🏆 Best overall model on this uploaded test set: {winner}. "
    "Ranking uses MCC as the primary criterion, "
    "with AUC and F1 as tie-breakers."
)

st.markdown(
    '<p class="small-note">'
    'All predictions above come from previously trained .joblib models. '
    'The uploaded CSV is used only for evaluation; no model training '
    'or fitting is performed inside this application.'
    '</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="footer-note">'
    'BioDegrade Lab • Machine Learning Assignment 2 • '
    'UCI QSAR Biodegradation Dataset'
    '</div>',
    unsafe_allow_html=True,
)
