# NLP Model Selection Report

## 1. Introduction
This report details the comparison of three NLP models for classifying support tickets into their respective **types** (Incident, Request, Problem, Change), aligning with the project brief's requirement to "predict the ticket type".

## 2. Methodology
- **Target Variable**: `type` (4 classes).
- **Preprocessing**: Advanced NLP pipeline including lowercasing, punctuation removal, stopword removal, and lemmatization (using spaCy `en_core_web_sm`).
- **Models Evaluated**:
    - **Logistic Regression**: TF-IDF vectorization (1-3 n-grams) + Logistic Regression.
    - **spaCy TextCategorizer**: CNN-based text classifier (`en_core_web_md` equivalent architecture).
    - **DistilBERT**: Transformer-based classifier (`distilbert-base-uncased`).

## 3. Benchmark Results
The models were evaluated on a held-out test set (20% split).

| Model | Accuracy | F1 Macro | Training Time (s) |
|---|---|---|---|
| **Logistic Regression** | **0.7885** | **0.7663** | 9.00 |
| DistilBERT | 0.7390 | 0.6353 | 150.89 |
| spaCy | 0.7068 | 0.7151 | 188.50 |

*Note: DistilBERT and spaCy were trained on a subsample (5000 rows) for 5 epochs due to CPU constraints. Logistic Regression was trained on the full dataset.*

## 4. Analysis
- **Logistic Regression** significantly outperformed deep learning models in this CPU-constrained environment, achieving **~79% accuracy**. It effectively utilizes the cleaned text features (lemmas).
- **DistilBERT** showed promise (74% accuracy) but suffered from limited training data and epochs. It requires GPU resources to properly fine-tune on the full dataset.
- **spaCy** was the slowest to train with the lowest accuracy in this configuration.

## 5. Recommendation
**Selected Model: Logistic Regression**

We recommend proceeding with the Logistic Regression model for the industrialization phase. 
- **Alignment with Brief**: Fulfills "Step 3: Training with scikit-learn".
- **Performance**: High accuracy and F1 score for the target variable `type`.
- **Efficiency**: Extremely fast inference, suitable for the resource-constrained environment (CPU).

**Future Improvements (per Brief)**:
- **Embeddings**: While we use TF-IDF for the classifier, we will still generate Hugging Face embeddings for ChromaDB (Step 2) as a separate retrieval/search feature.
