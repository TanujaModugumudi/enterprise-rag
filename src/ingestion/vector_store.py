import chromadb
from chromadb.config import Settings
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import CHROMA_DIR, COLLECTION_NAME


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Creates (or connects to) a persistent Chroma client.
    'Persistent' means data is saved to disk at CHROMA_DIR.
    Every time you run the app, it loads from the same folder —
    you don't re-embed documents on every run.
    """
    # Create the directory if it doesn't exist yet
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    return client


def get_or_create_collection(client: chromadb.PersistentClient):
    """
    Gets an existing Chroma collection or creates a new one.
    A collection is like a table in a regular database —
    it groups related vectors together.
    We use one collection for all domains (finance, legal, hr).
    Domain filtering happens via metadata at query time.
    """
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity
    )
    return collection


def store_chunks(chunks: list[dict]) -> None:
    """
    Stores all embedded chunks into Chroma.
    Chroma expects four parallel lists:
      - ids        : unique string ID for each chunk
      - embeddings : the 384-dim vector for each chunk
      - documents  : the raw text of each chunk
      - metadatas  : dict of source, domain, page_number

    We insert in batches of 500 to avoid memory issues.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Check if already populated — don't re-insert if ingestion
    # was already run. This makes the script safe to re-run.
    existing_count = collection.count()
    if existing_count > 0:
        print(f"Collection already contains {existing_count} chunks.")
        print("Skipping insertion. Delete chroma_db/ to re-ingest.")
        return

    # Prepare the four parallel lists Chroma expects
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "domain": chunk["domain"],
            "page_number": chunk["page_number"],
        })

    # Insert in batches of 500
    batch_size = 500
    total = len(ids)

    print(f"Storing {total} chunks in Chroma...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

        print(f"  Stored {end}/{total} chunks")

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has "
          f"{collection.count()} chunks.")


def verify_store() -> None:
    """
    Quick sanity check — queries the collection with a test question
    to confirm retrieval is working after ingestion.
    """
    from embedder import get_embedding_model

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    print(f"\nCollection size: {collection.count()} chunks")

    # Embed a test query and retrieve top 3 results
    model = get_embedding_model()
    test_query = "gratuity payment after resignation"
    query_vector = model.encode(test_query).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\nTest query: '{test_query}'")
    print("-" * 50)

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        # Chroma with cosine space returns distance (0=identical, 2=opposite)
        # Convert to similarity score: similarity = 1 - (distance/2)
        similarity = round(1 - (dist / 2), 3)
        print(f"\nResult {i+1}:")
        print(f"  Source    : {meta['source']}")
        print(f"  Domain    : {meta['domain']}")
        print(f"  Page      : {meta['page_number']}")
        print(f"  Similarity: {similarity}")
        print(f"  Text      : {doc[:150]}...")


if __name__ == "__main__":
    # This block only runs when you execute this file directly
    print("vector_store.py loaded. Use ingest.py to run the full pipeline.")