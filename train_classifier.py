"""
Train the final classification model (Logistic Regression + TF-IDF).
Saves the trained model and vectorizer to the models/ directory.
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = "data/cleaned_dataset_advanced.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
TARGET_COL = "type"
TEXT_COL = "full_text"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df[TEXT_COL] = df[TEXT_COL].fillna("")
    df = df.dropna(subset=[TARGET_COL])
    print(f"  → {len(df)} rows, {df[TARGET_COL].nunique()} classes")

    # ── Train / Test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COL], df[TARGET_COL],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )
    print(f"  → Train: {len(X_train)} | Test: {len(X_test)}")

    # ── TF-IDF Vectorization ──────────────────────────────────────────────
    print("Fitting TF-IDF vectorizer (1–3 n-grams, max 10 000 features)...")
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 3),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # ── Train Logistic Regression ──────────────────────────────────────────
    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train_tfidf, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Macro : {f1:.4f}")
    print("=" * 60)

    # ── Save artefacts ─────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nModel saved      → {MODEL_PATH}")
    print(f"Vectorizer saved → {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
