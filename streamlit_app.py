"""
BioDegrade Lab — QSAR Classification Benchmark
Evaluation Dashboard  ·  Student ID: 2026AJ05747
"""

from pathlib import Path
import io
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioDegrade Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global matplotlib style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.grid":          True,
    "grid.color":         "#f1f5f9",
    "grid.linewidth":     0.8,
    "axes.axisbelow":     True,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.edgecolor":     "#cbd5e1",
    "xtick.color":        "#6b7280",
    "ytick.color":        "#6b7280",
    "axes.labelcolor":    "#6b7280",
    "axes.labelsize":     9,
    "axes.titlesize":     11,
    "axes.titleweight":   "600",
    "axes.titlecolor":    "#111827",
    "axes.titlelocation": "left",
    "font.family":        "sans-serif",
    "font.size":          9.5,
    "legend.fontsize":    8.5,
    "legend.framealpha":  0.95,
    "legend.edgecolor":   "#e2e8f0",
    "legend.fancybox":    False,
    "figure.dpi":         120,
})

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Page background — slate-100, not white ── */
[data-testid="stAppViewContainer"] {
    background-color: #f0f4f8;
}
[data-testid="stMain"] {
    background-color: #f0f4f8;
}
.block-container {
    max-width: 1280px;
    padding-top: 0;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 3rem;
    background-color: #f0f4f8;
}

/* ── App header bar ── */
.app-header {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #dde3ec;
    border-left: 4px solid #2563eb;
    padding: 1.1rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.app-header-left {}
.app-title {
    font-size: 1.5rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.025em;
    display: block;
    line-height: 1.2;
}
.app-subtitle {
    font-size: 0.82rem;
    color: #64748b;
    font-weight: 400;
    margin-top: 0.2rem;
    display: block;
    letter-spacing: 0.01em;
}
.app-header-right {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-shrink: 0;
}
.ctx-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.74rem;
    font-weight: 500;
    color: #475569;
    background: #f1f5f9;
    border: 1px solid #dde3ec;
    border-radius: 100px;
    padding: 0.28rem 0.75rem;
    white-space: nowrap;
}
.ctx-pill .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #22c55e;
    flex-shrink: 0;
}

/* ── Section heading ── */
.sec-heading {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.85rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid #dde3ec;
}

/* ── Stat cards (dataset overview) ── */
.stat-card {
    padding: 1rem 1rem 0.9rem 1rem;
    background: #ffffff;
    border: 1px solid #dde3ec;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.stat-num {
    font-size: 1.9rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1;
    letter-spacing: -0.025em;
}
.stat-lbl {
    font-size: 0.68rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.3rem;
}

/* ── Metric cards ── */
.m-card {
    background: #ffffff;
    border: 1px solid #dde3ec;
    border-top: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.9rem 0.75rem 0.8rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.m-name {
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.45rem;
}
.m-val {
    font-size: 1.55rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1;
    letter-spacing: -0.02em;
}
.m-desc {
    font-size: 0.63rem;
    color: #b8c5d6;
    margin-top: 0.35rem;
}

/* ── Winner ── */
.winner-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin-top: 1.1rem;
}
.winner-eyebrow {
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #16a34a;
    margin-bottom: 0.25rem;
}
.winner-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #14532d;
}
.winner-note {
    font-size: 0.75rem;
    margin-top: 0.25rem;
    color: #15803d;
}

/* ── Footer ── */
.app-footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #dde3ec;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.74rem;
    color: #94a3b8;
}
.app-footer a { color: #94a3b8; text-decoration: none; }
.app-footer a:hover { color: #64748b; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #dde3ec;
    padding-bottom: 0;
    margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.6rem 1.1rem;
    color: #6b7280;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #2563eb !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #2563eb !important;
    background: transparent !important;
}

/* ── Sidebar — light steel-blue, professional and readable ── */
section[data-testid="stSidebar"] {
    background: #e8edf5 !important;
    border-right: 3px solid #2563eb !important;
}
section[data-testid="stSidebar"] > div:first-child {
    background: #e8edf5 !important;
}
/* All text inside sidebar → dark for readability */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #1e293b !important;
}
/* Divider line */
section[data-testid="stSidebar"] hr {
    border-color: #c8d4e8 !important;
}
/* Selectbox — light, high-contrast text */
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #ffffff !important;
    border-color: #c8d4e8 !important;
    color: #0f172a !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
}
section[data-testid="stSidebar"] .stSelectbox svg {
    fill: #475569 !important;
}
/* File uploader */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #ffffff !important;
    border-color: #c8d4e8 !important;
    border-radius: 8px;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] p {
    color: #64748b !important;
}
/* Download button in sidebar */
section[data-testid="stSidebar"] .stDownloadButton button {
    background: #ffffff !important;
    border-color: #c8d4e8 !important;
    color: #374151 !important;
    font-size: 0.8rem;
}
section[data-testid="stSidebar"] .stDownloadButton button:hover {
    background: #f1f5f9 !important;
    border-color: #94a3b8 !important;
}

/* ── Misc ── */
div[data-testid="stMetric"] { display: none; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

div[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #dde3ec;
    border-radius: 8px;
}
div[data-testid="stExpander"] summary {
    font-size: 0.84rem;
    color: #374151;
}

.stDownloadButton button {
    font-size: 0.8rem;
    padding: 0.35rem 0.9rem;
    border-radius: 6px;
    border: 1px solid #dde3ec !important;
    color: #374151 !important;
    background: #ffffff !important;
    font-weight: 500;
}
.stDownloadButton button:hover {
    border-color: #94a3b8 !important;
    background: #f8fafc !important;
}

/* Info/warning box */
div[data-testid="stAlert"] {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─── Paths & config ───────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent
MODEL_DIR      = ROOT_DIR / "model"
METADATA_PATH  = MODEL_DIR / "metadata.json"
TEST_DATA_PATH = ROOT_DIR / "test_data.csv"
CONFIG_PATH    = ROOT_DIR / "project_config.json"

CFG = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        CFG = json.load(fh)

if not METADATA_PATH.exists():
    st.error("Model metadata not found — run `train_and_evaluate.py` first.")
    st.stop()

with open(METADATA_PATH, "r", encoding="utf-8") as fh:
    meta = json.load(fh)

TARGET_COL    = meta["target_column"]
FEATURE_NAMES = meta["feature_names"]
MODEL_FILES   = meta["model_files"]

PALETTE = {
    "Logistic Regression":    "#6366f1",
    "Decision Tree":          "#f59e0b",
    "K-Nearest Neighbors":    "#10b981",
    "Gaussian Naive Bayes":   "#f43f5e",
    "Support Vector Machine": "#0ea5e9",
}
METRIC_ACCENTS = {
    "Accuracy":  "#6366f1",
    "AUC":       "#0ea5e9",
    "Precision": "#10b981",
    "Recall":    "#f59e0b",
    "F1":        "#f43f5e",
    "MCC":       "#8b5cf6",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    out = {}
    for name, fname in MODEL_FILES.items():
        p = MODEL_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing: {fname}")
        out[name] = joblib.load(p)
    return out


def metrics_for(y_true, y_pred, y_score):
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred),                    4),
        "AUC":       round(roc_auc_score(y_true, y_score),                    4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0),  4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0),     4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0),         4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred),                 4),
    }


@st.cache_data
def evaluate_all(_models, cache_key, X_df, y_arr):
    """
    X_df  : pd.DataFrame with named feature columns (avoids sklearn feature-name warnings)
    y_arr : 1-D numpy int array of true labels
    """
    rows, scores, preds = [], {}, {}
    for name, model in _models.items():
        yp = model.predict(X_df)
        ys = model.predict_proba(X_df)[:, 1]
        rows.append({"Model": name, **metrics_for(y_arr, yp, ys)})
        scores[name] = ys
        preds[name]  = yp
    return rows, scores, preds


def to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ─── Charts ──────────────────────────────────────────────────────────────────
def _ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def chart_cm(cm):
    fig, ax = _ax(5.2, 4.2)
    img = ax.imshow(cm, cmap="Blues", vmin=0, interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["NRB", "RB"], fontsize=9.5, fontweight="600")
    ax.set_yticklabels(["NRB", "RB"], fontsize=9.5, fontweight="600")
    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("True label",      fontsize=9)
    ax.set_title("Confusion matrix", pad=10)
    # grid off on imshow
    ax.grid(False)
    ax.spines[:].set_visible(False)

    total  = cm.sum()
    cutoff = cm.max() * 0.55
    for r in range(2):
        for c in range(2):
            v = cm[r, c]
            ax.text(c, r - 0.1, str(v),
                    ha="center", va="center", fontsize=16, fontweight="700",
                    color="white" if v > cutoff else "#0f172a")
            ax.text(c, r + 0.2, f"{v / total * 100:.1f}%",
                    ha="center", va="center", fontsize=9,
                    color="white" if v > cutoff else "#6b7280")

    fig.colorbar(img, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout(pad=1.0)
    return fig


def chart_roc_one(y_true, y_score, model_name):
    color = PALETTE.get(model_name, "#2563eb")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = _ax(4.2, 3.8)
    ax.fill_between(fpr, tpr, alpha=0.08, color=color)
    ax.plot(fpr, tpr, color=color, lw=2.0, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#cbd5e1", lw=1.0, label="Random")
    ax.set_xlabel("False positive rate", fontsize=9)
    ax.set_ylabel("True positive rate",  fontsize=9)
    ax.set_title("ROC curve", pad=10)
    ax.legend(loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.04)
    fig.tight_layout(pad=1.0)
    return fig


def chart_roc_all(y_true, all_scores):
    fig, ax = _ax(6.0, 4.6)
    ax.plot([0, 1], [0, 1], "--", color="#cbd5e1", lw=1.0, label="Random (0.50)")
    for name, ys in all_scores.items():
        fpr, tpr, _ = roc_curve(y_true, ys)
        auc = roc_auc_score(y_true, ys)
        ax.plot(fpr, tpr, lw=1.8, color=PALETTE.get(name, "#6b7280"),
                label=f"{name}  (AUC {auc:.4f})")
    ax.set_xlabel("False positive rate", fontsize=9)
    ax.set_ylabel("True positive rate",  fontsize=9)
    ax.set_title("ROC curves — all models", pad=10)
    ax.legend(loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.04)
    fig.tight_layout(pad=1.0)
    return fig


def chart_bars(df):
    cols    = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    models  = df["Model"].tolist()
    n       = len(models)
    x       = np.arange(len(cols))
    w       = 0.14
    offsets = np.linspace(-(n - 1) / 2 * w, (n - 1) / 2 * w, n)

    fig, ax = _ax(10, 4.4)
    for name, off in zip(models, offsets):
        vals  = [df.loc[df["Model"] == name, c].values[0] for c in cols]
        color = PALETTE.get(name, "#6b7280")
        ax.bar(x + off, vals, w, label=name, color=color, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(cols, fontsize=9.5, fontweight="600")
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Metric comparison — all models", pad=10)
    ax.legend(ncol=3)
    fig.tight_layout(pad=1.0)
    return fig


def chart_radar(df):
    cols   = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    n      = len(cols)
    angles = (np.linspace(0, 2 * np.pi, n, endpoint=False)).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.0, 5.0), subplot_kw={"polar": True})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for _, row in df.iterrows():
        name  = row["Model"]
        vals  = [row[c] for c in cols] + [row[cols[0]]]
        color = PALETTE.get(name, "#6b7280")
        ax.plot(angles, vals, lw=1.8, color=color, label=name)
        ax.fill(angles, vals, color=color, alpha=0.07)

    ax.set_thetagrids(np.degrees(angles[:-1]), cols,
                      fontsize=9, fontweight="600")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7.5, color="#94a3b8")
    ax.grid(color="#e2e8f0", alpha=0.9)
    ax.spines["polar"].set_color("#e2e8f0")
    ax.set_title("Radar chart", pad=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    return fig


def styled_df(df):
    num_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    return (
        df.style
          .format({c: "{:.4f}" for c in num_cols})
          .background_gradient(subset=num_cols, cmap="Blues", vmin=0.5, vmax=1.0)
          .set_properties(**{"text-align": "center", "font-size": "13px"})
          .set_table_styles([
              {"selector": "th", "props": [
                  ("background-color", "#f8fafc"),
                  ("color", "#374151"),
                  ("font-weight", "600"),
                  ("font-size", "11.5px"),
                  ("text-align", "center"),
                  ("border-bottom", "2px solid #e2e8f0"),
                  ("padding", "0.55rem 0.9rem"),
              ]},
              {"selector": "td",   "props": [("padding", "0.5rem 0.9rem")]},
              {"selector": "tr:hover td", "props": [("background-color", "#f8fafc")]},
          ])
    )


# ─── Load models ──────────────────────────────────────────────────────────────
models = load_models()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Branding — light sidebar, dark text
    st.markdown(
        "<div style='padding:1rem 0 0.85rem 0;border-bottom:1px solid #c8d4e8;margin-bottom:0.75rem;'>"
        "<span style='font-size:1rem;font-weight:800;color:#0f172a;letter-spacing:-0.01em;'>🧪 BioDegrade Lab</span><br/>"
        "<span style='font-size:0.74rem;color:#2563eb;font-weight:500;'>QSAR Classifier Evaluation</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── 1. Classifier selector  (most-used control → visible immediately) ──────
    st.markdown(
        "<p style='font-size:0.68rem;font-weight:700;color:#2563eb;"
        "text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.5rem;margin-top:0.25rem;'>"
        "Classifier</p>",
        unsafe_allow_html=True,
    )
    sel_name = st.selectbox(
        "Select model",
        options=list(models.keys()),
        label_visibility="collapsed",
    )

    # Colour swatch for selected model — on white pill, dark text
    _swatch = PALETTE.get(sel_name, "#6b7280")
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:0.5rem;"
        f"margin-top:0.35rem;margin-bottom:0.75rem;"
        f"padding:0.45rem 0.65rem;background:#ffffff;border-radius:6px;"
        f"border:1px solid #c8d4e8;box-shadow:0 1px 2px rgba(0,0,0,0.05);'>"
        f"<span style='width:9px;height:9px;border-radius:50%;"
        f"background:{_swatch};flex-shrink:0;display:inline-block;'></span>"
        f"<span style='font-size:0.76rem;color:#1e293b;font-weight:600;'>{sel_name}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#c8d4e8;margin:0.25rem 0 0.75rem 0;'/>", unsafe_allow_html=True)

    # ── 2. Test data ───────────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:0.68rem;font-weight:700;color:#2563eb;"
        "text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.5rem;'>"
        "Test Data</p>",
        unsafe_allow_html=True,
    )
    if TEST_DATA_PATH.exists():
        st.download_button(
            "↓  Download test_data.csv",
            data=TEST_DATA_PATH.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload test CSV",
        type=["csv"],
        help="Must contain all 41 feature columns and the target column.",
    )
    st.markdown("<hr style='border-color:#c8d4e8;margin:0.75rem 0;'/>", unsafe_allow_html=True)

    # ── 3. Project metadata (compact, no scrolling needed) ─────────────────────
    st.markdown(
        f"<div style='font-size:0.74rem;line-height:1.9;'>"
        f"<div style='display:grid;grid-template-columns:auto 1fr;gap:0 0.65rem;'>"
        f"<span style='color:#2563eb;font-weight:600;'>Dataset</span>"
        f"<span style='color:#374151;'>UCI QSAR #254</span>"
        f"<span style='color:#2563eb;font-weight:600;'>Target</span>"
        f"<span style='color:#374151;'>0 = NRB &nbsp;·&nbsp; 1 = RB</span>"
        f"<span style='color:#2563eb;font-weight:600;'>Features</span>"
        f"<span style='color:#374151;'>41 descriptors</span>"
        f"<span style='color:#2563eb;font-weight:600;'>Seed</span>"
        f"<span style='color:#374151;'>{CFG.get('random_seed', 5747)}</span>"
        f"<span style='color:#2563eb;font-weight:600;'>Student</span>"
        f"<span style='color:#374151;'>{CFG.get('student_id', '2026AJ05747')}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

# ─── App header bar ───────────────────────────────────────────────────────────
n_models = len(models)
file_ctx  = "No file uploaded" if uploaded_file is None else uploaded_file.name
file_dot  = "#94a3b8" if uploaded_file is None else "#22c55e"

st.markdown(
    f"""
    <div class="app-header">
        <div class="app-header-left">
            <span class="app-title">🧪 BioDegrade Lab</span>
            <span class="app-subtitle">QSAR Biodegradation &nbsp;·&nbsp; Classifier Evaluation Dashboard</span>
        </div>
        <div class="app-header-right">
            <span class="ctx-pill">
                <span class="dot" style="background:#22c55e;"></span>
                {n_models} models loaded
            </span>
            <span class="ctx-pill">
                <span class="dot" style="background:{file_dot};"></span>
                {file_ctx}
            </span>
            <span class="ctx-pill">&#9654;&nbsp; {sel_name}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Upload gate ──────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info(
        "Upload **test_data.csv** in the sidebar to begin evaluation.  "
        "The bundled test file is available to download from the sidebar."
    )

    st.markdown('<p class="sec-heading" style="margin-top:1.5rem;">Available classifiers</p>', unsafe_allow_html=True)
    cols = st.columns(5)
    descriptions = {
        "Logistic Regression":    "Linear baseline · highest AUC",
        "Decision Tree":          "Non-linear · fully interpretable",
        "K-Nearest Neighbors":    "Instance-based · lazy learner",
        "Gaussian Naive Bayes":   "Probabilistic · highest recall",
        "Support Vector Machine": "Kernel-based · strongest MCC",
    }
    for col, (name, desc) in zip(cols, descriptions.items()):
        color = PALETTE.get(name, "#6b7280")
        with col:
            st.markdown(
                f"<div style='border-top:3px solid {color};background:#ffffff;"
                f"border:1px solid #dde3ec;border-top:3px solid {color};"
                f"border-radius:8px;padding:0.9rem 0.9rem;"
                f"box-shadow:0 1px 3px rgba(0,0,0,0.05);'>"
                f"<div style='font-size:0.82rem;font-weight:600;color:#0f172a;"
                f"margin-bottom:0.3rem;'>{name}</div>"
                f"<div style='font-size:0.74rem;color:#64748b;line-height:1.5;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.stop()

# ─── Parse & validate ─────────────────────────────────────────────────────────
try:
    df_up = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

required = set(FEATURE_NAMES + [TARGET_COL])
missing  = sorted(required - set(df_up.columns))
if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

extra = sorted(set(df_up.columns) - required)
if extra:
    st.warning("Ignoring unrecognised columns: " + ", ".join(extra))

X = df_up[FEATURE_NAMES].copy()
y = df_up[TARGET_COL].astype(int).copy()

if set(y.unique()) - {0, 1}:
    st.error("Target column must contain only 0 and 1.")
    st.stop()
if y.nunique() < 2:
    st.error("Both classes must be present to compute AUC.")
    st.stop()

n_nrb   = int((y == 0).sum())
n_rb    = int((y == 1).sum())
n_total = len(y)

# ─── Evaluate (cached by file hash) ──────────────────────────────────────────
_key = str(hash(X.values.tobytes()))
with st.spinner("Evaluating models…"):
    comp_rows, all_scores, all_preds = evaluate_all(models, _key, X, y.values)

sel_preds   = all_preds[sel_name]
sel_scores  = all_scores[sel_name]
sel_met_row = next(r for r in comp_rows if r["Model"] == sel_name)
sel_met     = {k: v for k, v in sel_met_row.items() if k != "Model"}

comp_df = pd.DataFrame(comp_rows)[
    ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
]
winner = comp_df.sort_values(["MCC", "AUC", "F1"], ascending=False).iloc[0]["Model"]

# Best value per metric (for delta calculation in metric cards)
_metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
best_vals = {m: comp_df[m].max() for m in _metric_cols}

# ─── Dataset overview ─────────────────────────────────────────────────────────
st.markdown('<p class="sec-heading">Test set</p>', unsafe_allow_html=True)

ov1, ov2, ov3, ov4, ov5 = st.columns(5)
for col, num, lbl, color in [
    (ov1, n_total,            "Total samples",   "#0f172a"),
    (ov2, len(FEATURE_NAMES), "Input features",  "#0f172a"),
    (ov3, len(models),        "Classifiers",     "#0f172a"),
    (ov4, n_nrb,              "NRB (class 0)",   "#dc2626"),
    (ov5, n_rb,               "RB  (class 1)",   "#059669"),
]:
    with col:
        st.markdown(
            f"<div class='stat-card'>"
            f"<div class='stat-num' style='color:{color};'>{num}</div>"
            f"<div class='stat-lbl'>{lbl}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

with st.expander("Preview data", expanded=False):
    st.dataframe(df_up.head(10), use_container_width=True, hide_index=True)

# ─── Single model evaluation ──────────────────────────────────────────────────
st.markdown("<div style='margin-top:1.75rem'></div>", unsafe_allow_html=True)
st.markdown(
    f"<p class='sec-heading'>Model evaluation</p>",
    unsafe_allow_html=True,
)

# Model name sub-header with colour swatch and winner badge
_sel_color = PALETTE.get(sel_name, "#6b7280")
_is_winner = sel_name == winner
_badge = (
    "<span style='font-size:0.7rem;font-weight:600;color:#15803d;"
    "background:#dcfce7;border:1px solid #bbf7d0;border-radius:100px;"
    "padding:0.15rem 0.65rem;margin-left:0.6rem;vertical-align:middle;'>🏆 Best</span>"
    if _is_winner else ""
)
st.markdown(
    f"<div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:1rem;'>"
    f"<span style='width:12px;height:12px;border-radius:50%;background:{_sel_color};"
    f"flex-shrink:0;display:inline-block;'></span>"
    f"<span style='font-size:1rem;font-weight:700;color:#0f172a;'>{sel_name}</span>"
    f"{_badge}"
    f"</div>",
    unsafe_allow_html=True,
)

# Metric cards with delta vs best model
metric_defs = [
    ("Accuracy",  "Overall correct"),
    ("AUC",       "Area under ROC"),
    ("Precision", "Positive pred. value"),
    ("Recall",    "True positive rate"),
    ("F1",        "Harmonic mean P/R"),
    ("MCC",       "Matthews Correlation"),
]
m_cols = st.columns(6)
for col, (name, desc) in zip(m_cols, metric_defs):
    with col:
        accent = METRIC_ACCENTS.get(name, "#6366f1")
        val    = sel_met[name]
        best   = best_vals[name]
        delta  = val - best
        # Delta display: green "▲ 0.0000" if best, grey "▼ 0.0023" otherwise
        if abs(delta) < 1e-9:
            delta_html = (
                "<div style='font-size:0.63rem;font-weight:600;color:#22c55e;"
                "margin-top:0.3rem;'>★ Best</div>"
            )
        else:
            delta_html = (
                f"<div style='font-size:0.63rem;color:#94a3b8;margin-top:0.3rem;'>"
                f"▼ {abs(delta):.4f} vs best</div>"
            )
        st.markdown(
            f"<div class='m-card' style='--accent:{accent};'>"
            f"<div class='m-name'>{name}</div>"
            f"<div class='m-val'>{val:.4f}</div>"
            f"{delta_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Row: Confusion Matrix | ROC Curve (equal halves, no cramping) ─────────────
col_cm, col_roc = st.columns(2, gap="large")

with col_cm:
    st.markdown(
        "<p style='font-size:0.8rem;font-weight:600;color:#374151;"
        "margin-bottom:0.5rem;'>Confusion Matrix</p>",
        unsafe_allow_html=True,
    )
    fig_cm = chart_cm(confusion_matrix(y, sel_preds))
    st.pyplot(fig_cm, clear_figure=True, use_container_width=True)
    st.caption("Rows = actual class · Columns = predicted class · Values = count (% of total)")

with col_roc:
    st.markdown(
        "<p style='font-size:0.8rem;font-weight:600;color:#374151;"
        "margin-bottom:0.5rem;'>ROC Curve</p>",
        unsafe_allow_html=True,
    )
    fig_roc1 = chart_roc_one(y, sel_scores, sel_name)
    st.pyplot(fig_roc1, clear_figure=True, use_container_width=True)
    st.caption(f"AUC = {sel_met['AUC']:.4f} · Dashed line = random classifier baseline")

# ── Full-width classification report (no horizontal scroll) ───────────────────
st.markdown(
    "<p style='font-size:0.8rem;font-weight:600;color:#374151;"
    "margin-top:1rem;margin-bottom:0.5rem;'>Classification Report</p>",
    unsafe_allow_html=True,
)
rpt      = classification_report(y, sel_preds, output_dict=True, zero_division=0)
rpt_df   = pd.DataFrame(rpt).T.round(4)

# Reset index so class labels appear as a proper column, not a narrow index
rpt_display = rpt_df.reset_index().rename(columns={"index": "Class"})
rpt_display["Class"] = rpt_display["Class"].replace(
    {"0": "Class 0 — NRB", "1": "Class 1 — RB",
     "accuracy": "Accuracy", "macro avg": "Macro Avg",
     "weighted avg": "Weighted Avg"}
)

st.dataframe(
    rpt_display.style
        .format({c: "{:.4f}" for c in ["precision", "recall", "f1-score"]})
        .format({"support": "{:.0f}"})
        .set_properties(**{"text-align": "center", "font-size": "13px"})
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#f8fafc"), ("color", "#374151"),
                ("font-weight", "600"), ("font-size", "12px"),
                ("text-align", "center"), ("border-bottom", "2px solid #e2e8f0"),
                ("padding", "0.5rem 1rem"),
            ]},
            {"selector": "td", "props": [
                ("padding", "0.5rem 1rem"),
            ]},
            {"selector": "td:first-child", "props": [
                ("text-align", "left"), ("font-weight", "500"), ("color", "#374151"),
            ]},
        ]),
    use_container_width=True,
    hide_index=True,
    height=220,
)
st.caption("NRB = Not Ready Biodegradable · RB = Ready Biodegradable · Support = number of samples")

st.download_button(
    f"↓  Export {sel_name} predictions (.csv)",
    data=pd.DataFrame({
        "y_true":   y.values,
        "y_pred":   sel_preds,
        "prob_RB":  sel_scores.round(6),
    }).to_csv(index=False).encode(),
    file_name=f"{sel_name.lower().replace(' ','_')}_predictions.csv",
    mime="text/csv",
)

# ─── All models comparison ────────────────────────────────────────────────────
st.markdown("<div style='margin-top:1.75rem'></div>", unsafe_allow_html=True)
st.markdown('<p class="sec-heading">All models — comparison</p>', unsafe_allow_html=True)

tab_tbl, tab_bar, tab_roc, tab_radar = st.tabs([
    "Metrics table", "Bar chart", "ROC curves", "Radar chart",
])

with tab_tbl:
    st.dataframe(
        styled_df(comp_df),
        use_container_width=True,
        hide_index=True,
        height=236,
    )
    st.download_button(
        "↓  Export comparison table (.csv)",
        data=comp_df.to_csv(index=False).encode(),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

with tab_bar:
    fig_bar = chart_bars(comp_df)
    st.pyplot(fig_bar, clear_figure=True, use_container_width=True)
    st.download_button(
        "↓  Export chart (.png)",
        data=to_png(fig_bar),
        file_name="metric_comparison.png",
        mime="image/png",
    )

with tab_roc:
    fig_roc_a = chart_roc_all(y, all_scores)
    st.pyplot(fig_roc_a, clear_figure=True, use_container_width=True)
    st.download_button(
        "↓  Export chart (.png)",
        data=to_png(fig_roc_a),
        file_name="roc_curves.png",
        mime="image/png",
    )

with tab_radar:
    r1, r2, r3 = st.columns([0.5, 1, 0.5])
    with r2:
        fig_radar = chart_radar(comp_df)
        st.pyplot(fig_radar, clear_figure=True, use_container_width=True)
    st.download_button(
        "↓  Export chart (.png)",
        data=to_png(fig_radar),
        file_name="radar_chart.png",
        mime="image/png",
    )

# Winner
st.markdown(
    f"""
    <div class="winner-box">
        <div class="winner-eyebrow">Best overall model &nbsp;·&nbsp; ranked by MCC → AUC → F1</div>
        <div class="winner-name">🏆 &nbsp;{winner}</div>
        <div class="winner-note">
            Evaluated on the uploaded test set using pre-trained .joblib pipelines.
            No retraining was performed inside this application.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Footer ───────────────────────────────────────────────────────────────────
sid    = CFG.get("student_id",  "2026AJ05747")
seed   = CFG.get("random_seed", 5747)
tsplit = int(CFG.get("test_size", 0.2) * 100)
gh     = CFG.get("github_url",    "#")
app    = CFG.get("streamlit_url", "#")

st.markdown(
    f"""
    <div class="app-footer">
        <span>
            BioDegrade Lab &nbsp;·&nbsp; ML Assignment 2 &nbsp;·&nbsp;
            UCI QSAR Biodegradation (ID 254) &nbsp;·&nbsp;
            Student {sid} &nbsp;·&nbsp; Seed {seed} &nbsp;·&nbsp; Test split {tsplit}%
        </span>
        <span>
            <a href="{gh}"  target="_blank">GitHub</a> &nbsp;·&nbsp;
            <a href="{app}" target="_blank">Live app</a>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
