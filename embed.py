import json
import sys
import os
import chromadb
from sentence_transformers import SentenceTransformer

ARTICLES_PATH = "articles.json"

CHROMA_DB_PATH = "./chroma_db"

COLLECTION_NAME = "nepal_constitution"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

BATCH_SIZE = 32

def load_embedding_model(model_name):
    model = SentenceTransformer(model_name)
    return model

def setup_chromadb(db_path, collection_name):
    client = chromadb.PersistentClient(path=db_path)

    collection = client.get_or_create_collection(
        name = collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    return collection

def embed_and_store(articles, model, collection):
    existing_count = collection.count()
    if existing_count >= len(articles):
        print("ChromaDB already exist.")
        return
    
    total_batches = (len(articles) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        start  = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(articles))
        batch  = articles[start:end]

        texts_to_embed = [f"passage: {article['article_text']}" for article in batch]

        embeddings = model.encode(
            texts_to_embed,
            show_progress_bar = False,
            normalize_embeddings = True
        )

        collection.add(
            documents = [a["article_text"] for a in batch],

            embeddings = embeddings.tolist(),

            metadatas = [
                {
                    "article_number": a["article_number"],
                    "source": a["source"],
                    "language": a["language"],
                    "chunk_id": a ["chunk_id"]
                }
                for a in batch
            ],

            ids = [a["chunk_id"] for a in batch]
        )

        print(f"Batch {batch_num + 1}/{total_batches}-"
              f"articles {start + 1}-{end} embedded")

def main():
    if not os.path.exists(ARTICLES_PATH):
        print(f"Error: {ARTICLES_PATH} not found.")
        sys.exit(1)

    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        aritcles = json.load(f)
    print(f"Loaded {len(aritcles)} articles.\n")

    model = load_embedding_model(EMBEDDING_MODEL)

    collection = setup_chromadb(CHROMA_DB_PATH, COLLECTION_NAME)

    embed_and_store(aritcles, model, collection)

    print("=" * 55)
    print(f"  DONE. Database saved to {CHROMA_DB_PATH}/")
    print(f"  Next step: run streamlit run step3_app.py")
    print("=" * 55)
    
if __name__ == "__main__":
    main()