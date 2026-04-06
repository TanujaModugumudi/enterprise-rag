from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import EMBEDDING_MODEL


# Module-level model instance — loaded once, reused across calls.
# Loading a model takes ~3 seconds. If we loaded it inside a function,
# every function call would pay that cost. Loading it once at import
# time means we pay the cost once per program run.
_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Returns the embedding model, loading it on first call only.
    This pattern is called lazy initialization — we don't load
    the model until something actually needs it.
    """
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        print("(First load downloads ~90MB — subsequent runs use cache)")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("Model loaded.")
    return _model


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Takes a list of chunk dicts and adds an 'embedding' key to each.
    The embedding is a list of 384 floats representing the chunk's
    semantic meaning in vector space.

    We embed in batches of 64 for memory efficiency — embedding all
    10,000+ chunks at once would require too much RAM on your machine.
    """
    model = get_embedding_model()

    # Extract just the text strings for batch embedding
    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks in batches of 64...")
    # show_progress_bar gives us a live progress indicator
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,   # returns numpy arrays, compatible with Chroma
    )

    # Attach each embedding back to its chunk dict
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()  # convert numpy → plain list

    print(f"Done. Each embedding has {len(chunks[0]['embedding'])} dimensions.")
    return chunks


if __name__ == "__main__":
    from loader import load_all_documents
    from chunker import chunk_pages

    pages = load_all_documents()
    chunks = chunk_pages(pages)

    # Test on just the first 50 chunks — don't embed all 10k during testing
    sample_chunks = chunks[:50]
    embedded = embed_chunks(sample_chunks)

    print(f"\n--- SAMPLE EMBEDDING ---")
    print(f"chunk_id : {embedded[0]['chunk_id']}")
    print(f"text     : {embedded[0]['text'][:100]}...")
    print(f"embedding: [{embedded[0]['embedding'][0]:.6f}, "
          f"{embedded[0]['embedding'][1]:.6f}, ... "
          f"({len(embedded[0]['embedding'])} dims)]")