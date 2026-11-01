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
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #eef7f2, #f6f7fb);
        border: 1px solid #d9e7df;
        margin-bottom: 1rem;
    }
    .small-note {
        color: #5a6570;
        font-size: 0.92rem;
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
        Interactive comparison of five classification models for
        predicting ready biodegradability from molecular descriptors.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "UCI QSAR Biodegradation • 41 molecular descriptors • "
    "saved-model evaluation only"
)

with st.expander("About this experiment"):
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

uploaded_file = st.file_uploader(
    "Upload test CSV",
    type=["csv"],
    help="The CSV must contain all 41 input features and the target column.",
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

st.subheader("Uploaded Test Data")
summary_columns = st.columns(4)
summary_columns[0].metric("Rows", len(uploaded))
summary_columns[1].metric("Input Features", len(FEATURE_NAMES))
summary_columns[2].metric("Class 0", int((y_uploaded == 0).sum()))
summary_columns[3].metric("Class 1", int((y_uploaded == 1).sum()))

st.dataframe(uploaded.head(8), use_container_width=True)

st.divider()

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

st.header(f"Evaluation — {selected_model_name}")

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
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(selected_model_name)

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
st.header("All Five Models on the Uploaded Test Data")

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
    f"Highest-ranked model on this uploaded test set: {winner} "
    "(MCC primary; AUC and F1 tie-breakers)."
)

st.markdown(
    '<p class="small-note">'
    'All predictions above come from saved .joblib models. '
    'No training is performed on the uploaded CSV.'
    '</p>',
    unsafe_allow_html=True,
)
