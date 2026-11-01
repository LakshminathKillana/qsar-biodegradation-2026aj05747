from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import io

import pandas as pd


# =========================================================
# Project paths and official UCI dataset source
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = DATA_DIR / "qsar_biodegradation.csv"

# Official UCI legacy file URL for dataset 254.
# The current UCI page lists this file as biodeg.csv.
UCI_DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00254/biodeg.csv"
)


# =========================================================
# Download helper
# =========================================================

def download_text(url: str) -> str:
    """
    Download the official UCI CSV using Python's standard library.
    A browser-like User-Agent is supplied because some university/network
    environments reject requests that do not provide one.
    """
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            )
        },
    )

    try:
        with urlopen(request, timeout=60) as response:
            raw_bytes = response.read()

    except (HTTPError, URLError) as error:
        raise RuntimeError(
            "\nUnable to download the QSAR Biodegradation dataset "
            "from the official UCI source.\n"
            f"URL: {url}\n"
            f"Reason: {error}\n"
            "\nOpen the UCI dataset page in a browser, download "
            "'biodeg.csv' manually, and place it inside the data folder."
        ) from error

    return raw_bytes.decode("utf-8")


# =========================================================
# Load official UCI data
# =========================================================

print("=" * 72)
print("QSAR BIODEGRADATION DATASET PREPARATION")
print("=" * 72)

print("\nDownloading official UCI biodeg.csv ...")

csv_text = download_text(UCI_DATA_URL)

# UCI's biodeg.csv:
# - has no header row
# - uses semicolon separators
# - contains 41 numeric descriptors + 1 class column
raw_df = pd.read_csv(
    io.StringIO(csv_text),
    sep=";",
    header=None,
)

print("Download completed successfully.")


# =========================================================
# Structural validation
# =========================================================

EXPECTED_ROWS = 1055
EXPECTED_FEATURES = 41
EXPECTED_COLUMNS = EXPECTED_FEATURES + 1

if raw_df.shape[1] != EXPECTED_COLUMNS:
    raise ValueError(
        f"Expected {EXPECTED_COLUMNS} columns "
        f"(41 features + class), but found {raw_df.shape[1]}."
    )

if raw_df.shape[0] != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} rows, but found {raw_df.shape[0]}."
    )


# =========================================================
# Assign clear local column names
# =========================================================

feature_columns = [
    f"feature_{index:02d}"
    for index in range(1, EXPECTED_FEATURES + 1)
]

raw_df.columns = feature_columns + ["class_label"]


# =========================================================
# Clean and encode target
# =========================================================

class_labels = (
    raw_df["class_label"]
    .astype(str)
    .str.strip()
    .str.upper()
)

unexpected_labels = sorted(
    set(class_labels.unique()) - {"RB", "NRB"}
)

if unexpected_labels:
    raise ValueError(
        f"Unexpected class labels found: {unexpected_labels}"
    )

# Positive class:
# RB  = Ready Biodegradable      -> 1
# NRB = Not Ready Biodegradable  -> 0
prepared_df = raw_df.drop(columns=["class_label"]).copy()

prepared_df["target"] = (
    class_labels
    .map({"NRB": 0, "RB": 1})
    .astype(int)
)


# =========================================================
# Data-quality validation
# =========================================================

feature_data = prepared_df[feature_columns]

# Make sure all 41 descriptors are numeric.
for column in feature_columns:
    prepared_df[column] = pd.to_numeric(
        prepared_df[column],
        errors="raise",
    )

missing_values = int(prepared_df.isna().sum().sum())

if missing_values != 0:
    raise ValueError(
        f"Unexpected missing values found: {missing_values}"
    )

if set(prepared_df["target"].unique()) != {0, 1}:
    raise ValueError(
        "The encoded target is not binary 0/1."
    )

assert prepared_df.shape[0] >= 500, (
    "Assignment requires at least 500 instances."
)

assert len(feature_columns) >= 12, (
    "Assignment requires at least 12 input features."
)


# =========================================================
# Save prepared dataset
# =========================================================

prepared_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# =========================================================
# Summary
# =========================================================

print(f"\nInstances       : {prepared_df.shape[0]}")
print(f"Input features  : {len(feature_columns)}")
print(f"Total columns   : {prepared_df.shape[1]}")
print(f"Missing values  : {missing_values}")

print("\nTarget distribution:")
print(
    prepared_df["target"]
    .value_counts()
    .sort_index()
    .rename(index={
        0: "0 = NRB (Not Ready Biodegradable)",
        1: "1 = RB (Ready Biodegradable)",
    })
)

print(f"\nSaved dataset to:\n{OUTPUT_PATH}")

print("\nDataset requirement checks:")
print(f"Instances >= 500 : PASS ({prepared_df.shape[0]})")
print(f"Features  >= 12  : PASS ({len(feature_columns)})")

print("\nDATASET PREPARATION COMPLETED SUCCESSFULLY")
