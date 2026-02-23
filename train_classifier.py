import os
import pandas as pd
import joblib
import chromadb
from chromadb.utils import embedding_functions
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = "data/cleaned_dataset_advanced.csv"
DB_PATH = "chroma_db"
COLLECTION_NAME = "support_tickets"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
BASELINE_PATH = "data/baseline.parquet"
TARGET_COL = "type"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    # ── Load data labels ───────────────────────────────────────────────────
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET_COL])
    print(f"  → {len(df)} rows, {df[TARGET_COL].nunique()} classes")

    # ── Fetch embeddings from ChromaDB ─────────────────────────────────────
    print(f"Fetching embeddings from ChromaDB (collection: {COLLECTION_NAME})...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # We need the same embedding function to ensure consistency if we were querying, 
    # but here we just want to retrieve the existing embeddings.
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
    
    # Retrieve all items (embeddings + metadatas)
    # Using the IDs from the dataframe (which were used as keys in generate_embeddings.py)
    all_data = collection.get(include=['embeddings', 'metadatas'])
    
    if not all_data['embeddings']:
        print("Error: No embeddings found in ChromaDB. Please run generate_embeddings.py first.")
        return

    # Create a mapping from ID to embedding to ensure order matches our DataFrame
    emb_map = {all_data['ids'][i]: all_data['embeddings'][i] for i in range(len(all_data['ids']))}
    
    # Align embeddings with the current dataframe rows
    embeddings = []
    valid_indices = []
    for idx in df.index:
        id_str = str(idx)
        if id_str in emb_map:
            embeddings.append(emb_map[id_str])
            valid_indices.append(idx)
    
    df = df.loc[valid_indices]
    X = pd.DataFrame(embeddings)
    y = df[TARGET_COL]

    # ── Train / Test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  → Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Train Logistic Regression ──────────────────────────────────────────
    print("Training Logistic Regression on embeddings...")
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Using ChromaDB Embeddings)")
    print("=" * 60)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Macro : {f1:.4f}")
    print("=" * 60)

    # ── Save artefacts ─────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    
    # Save training data as baseline for Evidently AI
    # We combine features and target for the baseline
    baseline_df = X_train.copy()
    baseline_df[TARGET_COL] = y_train.values
    baseline_df['prediction'] = clf.predict(X_train)
    
    # Add some metadata for drift analysis (e.g. language, priority if available in original df)
    for col in ['priority', 'language']:
        if col in df.columns:
            baseline_df[col] = df.loc[X_train.index, col].values

    baseline_df.to_parquet(BASELINE_PATH, index=False)
    
    print(f"\nModel saved    → {MODEL_PATH}")
    print(f"Baseline saved → {BASELINE_PATH}")


if __name__ == "__main__":
    main()
