import pandas as pd
import re
import spacy
from tqdm import tqdm

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("Downloading 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text_advanced(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase & remove HTML (basic)
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    
    # SpaCy processing: tokenization, stopword removal, punctuation removal, lemmatization
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num]
    
    return " ".join(tokens)

def preprocess_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Normalizing and tokenizing text (this might take a while)...")
    # Using progress_apply if possible, or just standard apply with tqdm
    tqdm.pandas()
    
    df['clean_subject'] = df['subject'].fillna("").progress_apply(clean_text_advanced)
    df['clean_body'] = df['body'].fillna("").progress_apply(clean_text_advanced)
    
    # Feature engineering: Combined text
    df['full_text'] = df['clean_subject'] + " " + df['clean_body']
    
    print(f"Saving advanced cleaned data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    input_file = 'data/dataset.csv'
    output_file = 'data/cleaned_dataset_advanced.csv'
    preprocess_data(input_file, output_file)
