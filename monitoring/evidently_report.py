import os
import pandas as pd
import joblib
import chromadb
from chromadb.utils import embedding_functions

from evidently import Report, DataDefinition, Dataset, MulticlassClassification
from evidently.presets import DataDriftPreset, ClassificationPreset

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = "data/cleaned_dataset_advanced.csv"
BASELINE_PATH = "data/baseline.parquet"
MODEL_PATH = "models/classifier.joblib"
DB_PATH = "chroma_db"
COLLECTION_NAME = "support_tickets"
REPORT_DIR = "monitoring/reports"
REPORT_PATH = os.path.join(REPORT_DIR, "drift_report.html")
TARGET_COL = "type"
PREDICTION_COL = "prediction"


def main():
    # ── Load baseline (Reference) ──────────────────────────────────────────
    if not os.path.exists(BASELINE_PATH):
        print(f"Error: Baseline file {BASELINE_PATH} not found. Please run train_classifier.py first.")
        return
    
    print(f"Loading baseline from {BASELINE_PATH}...")
    ref_df = pd.read_parquet(BASELINE_PATH)

    # ── Prepare current data (Current) ─────────────────────────────────────
    print(f"Loading current data from {DATA_PATH} and ChromaDB...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET_COL])

    # Fetch embeddings for current data
    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
    
    all_data = collection.get(include=['embeddings', 'metadatas'])
    emb_map = {all_data['ids'][i]: all_data['embeddings'][i] for i in range(len(all_data['ids']))}
    
    embeddings = []
    valid_indices = []
    for idx in df.index:
        id_str = str(idx)
        if id_str in emb_map:
            embeddings.append(emb_map[id_str])
            valid_indices.append(idx)
    
    cur_df = pd.DataFrame(embeddings)
    cur_df[TARGET_COL] = df.loc[valid_indices, TARGET_COL].values
    
    # Add metadata
    for col in ['priority', 'language', 'queue']:
        if col in df.columns:
            cur_df[col] = df.loc[valid_indices, col].values

    # ── Add model predictions for current data ─────────────────────────────
    print("Generating model predictions for current data...")
    clf = joblib.load(MODEL_PATH)
    cur_df[PREDICTION_COL] = clf.predict(cur_df.drop(columns=[TARGET_COL, 'priority', 'language', 'queue'], errors='ignore'))

    print(f"  → Reference: {len(ref_df)} rows | Current: {len(cur_df)} rows")

    # ── Define data mapping ────────────────────────────────────────────────
    # Features in current_df are the embedding dimensions (integers/named columns)
    # We identify categorical columns for drift analysis
    categorical_cols = ["priority", "language", "queue"]
    
    data_def = DataDefinition(
        categorical_columns=[col for col in categorical_cols if col in cur_df.columns],
        classification=[
            MulticlassClassification(
                target=TARGET_COL,
                prediction_labels=PREDICTION_COL
            )
        ]
    )

    # Wrap DataFrames in Dataset objects
    ref_ds = Dataset.from_pandas(ref_df, data_definition=data_def)
    cur_ds = Dataset.from_pandas(cur_df, data_definition=data_def)

    # ── Build Evidently Report ─────────────────────────────────────────────
    print("Building Evidently report (DataDrift + Classification)...")
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset(),
    ])

    report.run(reference_data=ref_ds, current_data=cur_ds)
    
    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(REPORT_DIR, exist_ok=True)
    report.save_html(REPORT_PATH)
    print(f"\n✅ Evidently report saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
