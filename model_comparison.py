import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import spacy
from spacy.util import minibatch, compounding
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Constants
DATA_PATH = 'data/cleaned_dataset_advanced.csv'
TARGET_COL = 'type' # Changed from 'queue' to 'type'
TEXT_COL = 'full_text'
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_and_preprocess():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Fill NaN just in case
    df[TEXT_COL] = df[TEXT_COL].fillna("")
    
    # Filter out rows with missing target
    df = df.dropna(subset=[TARGET_COL])
    
    return df

def run_logistic_regression(train_df, test_df):
    print("\n--- Running Logistic Regression (Baseline) ---")
    start_time = time.time()
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(train_df[TEXT_COL])
    X_test = vectorizer.transform(test_df[TEXT_COL])
    
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, train_df[TARGET_COL])
    
    preds = clf.predict(X_test)
    duration = time.time() - start_time
    
    report = classification_report(test_df[TARGET_COL], preds, output_dict=True)
    print(f"Logistic Regression Accuracy: {accuracy_score(test_df[TARGET_COL], preds):.4f}")
    return {
        'model': 'LogisticRegression',
        'accuracy': accuracy_score(test_df[TARGET_COL], preds),
        'f1_macro': f1_score(test_df[TARGET_COL], preds, average='macro'),
        'duration': duration
    }

def run_spacy(train_df, test_df):
    print("\n--- Running spaCy TextCategorizer ---")
    start_time = time.time()
    
    # Load existing model
    nlp = spacy.load("en_core_web_md")
    
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat", last=True)
    else:
        textcat = nlp.get_pipe("textcat")
    
    # Add labels
    for label in train_df[TARGET_COL].unique():
        textcat.add_label(label)
    
    # Subsetting for benchmark speed if needed
    train_sub = train_df.sample(min(5000, len(train_df)), random_state=RANDOM_STATE)
    
    # Prepare training data
    from spacy.training import Example
    
    # Training (simplified for benchmark)
    optimizer = nlp.begin_training()
    for i in range(5):  # 5 iterations
        losses = {}
        batches = minibatch(train_sub.index, size=compounding(4.0, 32.0, 1.001))
        for batch_indices in batches:
            examples = []
            for idx in batch_indices:
                text = train_df.loc[idx, TEXT_COL]
                label = train_df.loc[idx, TARGET_COL]
                cats = {l: 1.0 if l == label else 0.0 for l in train_df[TARGET_COL].unique()}
                examples.append(Example.from_dict(nlp.make_doc(text), {"cats": cats}))
            nlp.update(examples, sgd=optimizer, losses=losses)
        print(f"Iteration {i} Losses: {losses}")
    
    # Evaluation
    preds = []
    for text in test_df[TEXT_COL]:
        doc = nlp(text)
        preds.append(max(doc.cats, key=doc.cats.get))
    
    duration = time.time() - start_time
    print(f"spaCy Accuracy: {accuracy_score(test_df[TARGET_COL], preds):.4f}")
    return {
        'model': 'spaCy',
        'accuracy': accuracy_score(test_df[TARGET_COL], preds),
        'f1_macro': f1_score(test_df[TARGET_COL], preds, average='macro'),
        'duration': duration
    }

def run_distilbert(train_df, test_df):
    print("\n--- Running DistilBERT (Transformers) ---")
    # Subsetting for benchmark speed if needed, but 4k rows is usually fine
    train_sub = train_df.sample(min(5000, len(train_df)), random_state=RANDOM_STATE) 
    test_sub = test_df.sample(min(1000, len(test_df)), random_state=RANDOM_STATE)
    
    start_time = time.time()
    
    # Encode labels
    labels = sorted(train_df[TARGET_COL].unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    
    # Prepare datasets
    train_ds = Dataset.from_dict({'text': train_sub[TEXT_COL].tolist(), 'label': [label2id[l] for l in train_sub[TARGET_COL]]})
    test_ds = Dataset.from_dict({'text': test_sub[TEXT_COL].tolist(), 'label': [label2id[l] for l in test_sub[TARGET_COL]]})
    
    train_ds = train_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)
    
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="no",
        fp16=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds
    )
    
    trainer.train()
    
    # Evaluate
    outputs = trainer.predict(test_ds)
    preds = np.argmax(outputs.predictions, axis=1)
    true_labels = outputs.label_ids
    
    duration = time.time() - start_time
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='macro')
    
    print(f"DistilBERT (Subset) Accuracy: {acc:.4f}")
    return {
        'model': 'DistilBERT',
        'accuracy': acc,
        'f1_macro': f1,
        'duration': duration
    }

def main():
    df = load_and_preprocess()
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COL])
    
    results = []
    results.append(run_logistic_regression(train_df, test_df))
    results.append(run_spacy(train_df, test_df))
    results.append(run_distilbert(train_df, test_df))
    
    print("\n=== Final Results ===")
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv('model_comparison_results.csv', index=False)

if __name__ == "__main__":
    main()
