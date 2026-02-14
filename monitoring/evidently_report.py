"""
Evidently AI — Data Drift & Classification Monitoring Report.

Splits the dataset into reference (first half) and current (second half)
to simulate temporal drift, adds model predictions, and generates
an interactive HTML report.
"""
import os
import pandas as pd
import joblib

from evidently import Report, DataDefinition, Dataset, MulticlassClassification
from evidently.presets import DataDriftPreset, ClassificationPreset

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = "data/cleaned_dataset_advanced.csv"
MODEL_PATH = "models/classifier.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"
REPORT_DIR = "monitoring/reports"
REPORT_PATH = os.path.join(REPORT_DIR, "drift_report.html")
TARGET_COL = "type"
TEXT_COL = "full_text"
PREDICTION_COL = "prediction"


def main():
    # ── Load data & model ──────────────────────────────────────────────────
    print("Loading dataset and trained model...")
    df = pd.read_csv(DATA_PATH)
    df[TEXT_COL] = df[TEXT_COL].fillna("")
    df = df.dropna(subset=[TARGET_COL])

    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # ── Add predictions ────────────────────────────────────────────────────
    print("Generating model predictions for entire dataset...")
    X_tfidf = vectorizer.transform(df[TEXT_COL])
    df[PREDICTION_COL] = clf.predict(X_tfidf)

    # ── Prepare analysis DataFrame ─────────────────────────────────────────
    # Keep meaningful categorical features + target + prediction
    categorical_cols = ["priority", "language", "queue"]
    keep_cols = categorical_cols + [TARGET_COL, PREDICTION_COL]
    analysis_df = df[keep_cols].copy()

    # Add a numerical feature: text length (for drift detection)
    analysis_df["text_length"] = df[TEXT_COL].str.len()

    # ── Split into reference (old) vs current (new) ────────────────────────
    midpoint = len(analysis_df) // 2
    ref_df = analysis_df.iloc[:midpoint].reset_index(drop=True)
    cur_df = analysis_df.iloc[midpoint:].reset_index(drop=True)
    print(f"  → Reference: {len(ref_df)} rows | Current: {len(cur_df)} rows")

    # ── Define data mapping for Evidently v0.7+ ────────────────────────────
    data_def = DataDefinition(
        classification=[
            MulticlassClassification(
                target=TARGET_COL,
                prediction_labels=PREDICTION_COL,
            )
        ]
    )

    ref_ds = Dataset.from_pandas(ref_df, data_definition=data_def)
    cur_ds = Dataset.from_pandas(cur_df, data_definition=data_def)

    # ── Build Evidently Report ─────────────────────────────────────────────
    print("Building Evidently report (DataDrift + Classification)...")
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset(),
    ])

    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(REPORT_DIR, exist_ok=True)
    snapshot.save_html(REPORT_PATH)
    print(f"\n✅ Evidently report saved → {REPORT_PATH}")
    print(f"   Open in browser: file://{os.path.abspath(REPORT_PATH)}")


if __name__ == "__main__":
    main()
