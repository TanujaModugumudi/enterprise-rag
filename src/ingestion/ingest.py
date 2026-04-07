"""
ingest.py — runs the complete ingestion pipeline end to end.

Run this once to populate your Chroma vector store.
After this runs successfully, you never need to run it again
unless you add new documents or delete chroma_db/.

"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


from src.ingestion.loader import load_domain
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import embed_chunks
from src.ingestion.vector_store import store_chunks, verify_store


def run_ingestion():
    print("=" * 60)
    print("ENTERPRISE RAG — INGESTION PIPELINE")
    print("=" * 60)

    # Early exit — if Chroma already has chunks, skip everything
    # and just run verify. No point re-embedding 10k chunks.
    from src.ingestion.vector_store import get_chroma_client, get_or_create_collection
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    if collection.count() > 0:
        print(f"\nChroma already contains {collection.count()} chunks.")
        print("Skipping ingestion. Running verify only...\n")
        from src.ingestion.vector_store import verify_store
        verify_store()
        print("\n" + "=" * 60)
        print("Verify complete.")
        print("=" * 60)
        return

    # Step 1: Load all PDFs across all 3 domains
    print("\n[1/4] Loading documents...")
    pages = []
    pages.extend(load_domain("data/raw/finance", "finance"))
    pages.extend(load_domain("data/raw/legal",   "legal"))
    pages.extend(load_domain("data/raw/hr",      "hr"))
    print(f"\nTotal pages loaded: {len(pages)}")

    # Step 2: Chunk pages into smaller pieces
    print("\n[2/4] Chunking pages...")
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks.")

    # Step 3: Embed chunks into vectors
    print("\n[3/4] Embedding chunks...")
    # This is the slow step — ~38 mins on CPU for 10k chunks
    # Your laptop fan will spin. Do not close the terminal.
    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks.")

    # Step 4: Store in Chroma
    print("\n[4/4] Storing in Chroma vector store...")
    store_chunks(chunks)

    # Verify retrieval works
    print("\n[VERIFY] Running test query...")
    verify_store()

    print("\n" + "=" * 60)
    print("Ingestion complete. Your vector store is ready.")
    print(f"Chroma DB saved to: chroma_db/")
    print("=" * 60)