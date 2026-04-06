from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def create_splitter() -> RecursiveCharacterTextSplitter:
    """
    RecursiveCharacterTextSplitter splits text by trying these
    separators in order: paragraphs → sentences → words → characters.
    It only moves to the next separator if the chunk is still too big.
    This preserves natural language boundaries as much as possible.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Takes the list of page dicts from the loader and splits each page
    into smaller chunks. Every chunk inherits the metadata of its
    parent page (source, domain, page_number) and gets a unique chunk_id.

    Returns a flat list of chunk dicts.
    """
    splitter = create_splitter()
    all_chunks = []

    for page in pages:
        # Split the page text into chunks
        texts = splitter.split_text(page["text"])

        for i, text in enumerate(texts):
            # Skip chunks that are too short to be meaningful
            if len(text.strip()) < 30:
                continue

            chunk = {
                "text": text,
                "source": page["source"],
                "domain": page["domain"],
                "page_number": page["page_number"],
                # unique id: filename_stem + page + chunk index
                "chunk_id": f"{Path(page['source']).stem}_p{page['page_number']}_c{i}",
            }
            all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":
    # Test chunker with the loader
    from loader import load_all_documents

    pages = load_all_documents()
    chunks = chunk_pages(pages)

    print(f"\nTotal pages : {len(pages)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunks per page: {len(chunks)/len(pages):.1f}")

    # Show a sample chunk
    sample = chunks[10]
    print(f"\n--- SAMPLE CHUNK ---")
    print(f"chunk_id   : {sample['chunk_id']}")
    print(f"source     : {sample['source']}")
    print(f"domain     : {sample['domain']}")
    print(f"page_number: {sample['page_number']}")
    print(f"text length: {len(sample['text'])} chars")
    print(f"text preview: {sample['text'][:200]}")