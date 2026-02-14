import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import os


def generate_and_store_embeddings(input_path, db_path, collection_name):
    # Create DB directory if it doesn't exist
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # Handle missing values
    df['full_text'] = df['full_text'].fillna("")

    # Filter out empty texts if any
    df = df[df['full_text'].str.strip() != ""]

    print(f"Initializing ChromaDB client at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)

    # Initialize embedding function (HuggingFace)
    print("Initializing embedding function (all-MiniLM-L6-v2)...")
    model_name = "all-MiniLM-L6-v2"
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name)

    # Create or get collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn
    )

    print(
        f"Generating embeddings and adding {
            len(df)} documents to ChromaDB...")

    # ChromaDB supports batching. Let's batch to avoid memory issues and
    # improve speed
    batch_size = 500
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i + batch_size]

        ids = [str(idx) for idx in batch_df.index]
        documents = batch_df['full_text'].tolist()

        # Prepare metadata - ensure no NaN values as ChromaDB doesn't like them
        metadatas = batch_df[['subject', 'type', 'priority', 'language']].fillna(
            "unknown").to_dict('records')

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    print(
        f"Successfully added all documents to collection '{collection_name}'.")


if __name__ == "__main__":
    input_file = '/home/dicksons/projects/nlp/data/cleaned_dataset_advanced.csv'
    db_directory = '/home/dicksons/projects/nlp/chroma_db'
    coll_name = 'support_tickets'

    generate_and_store_embeddings(input_file, db_directory, coll_name)
