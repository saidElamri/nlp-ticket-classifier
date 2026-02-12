import chromadb
from chromadb.utils import embedding_functions

def verify_chroma(db_path, collection_name):
    print(f"Connecting to ChromaDB at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    
    # Use the same embedding function
    model_name = "all-MiniLM-L6-v2"
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    
    collection = client.get_collection(name=collection_name, embedding_function=emb_fn)
    
    count = collection.count()
    print(f"Collection '{collection_name}' contains {count} documents.")
    
    # Test queries
    queries = [
        "Network connection issues on Ubuntu",
        "Data analysis platform crash due to memory",
        "How to optimize my investment portfolio?"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            print(f"Result {i+1} (Distance: {distance:.4f}):")
            print(f"  Subject: {metadata.get('subject')}")
            print(f"  Type: {metadata.get('type')}")
            print(f"  Snippet: {doc[:100]}...")

if __name__ == "__main__":
    db_directory = '/home/dicksons/projects/nlp/chroma_db'
    coll_name = 'support_tickets'
    verify_chroma(db_directory, coll_name)
