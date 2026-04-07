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


def chunk_pages(pages) -> list[dict]:
    """
    Takes the list of Document objects from the loader and splits
    each page into smaller chunks. Every chunk inherits the metadata
    of its parent page (source, domain, page_number) and gets a
    unique chunk_id.

    Returns a flat list of chunk dicts.
    """
    splitter = create_splitter()
    all_chunks = []

    for page in pages:
        # Document objects use .page_content and .metadata
        # not dictionary keys like page["text"]
        text        = page.page_content
        source      = page.metadata["source"]
        domain      = page.metadata["domain"]
        page_number = page.metadata["page_number"]

        # Split the page text into chunks
        texts = splitter.split_text(text)

        for i, chunk_text in enumerate(texts):
            # Skip chunks that are too short to be meaningful
            if len(chunk_text.strip()) < 30:
                continue

            chunk = {
                "text":        chunk_text,
                "source":      source,
                "domain":      domain,
                "page_number": page_number,
                # unique id: filename_stem + page + chunk index
                "chunk_id": f"{Path(source).stem}_p{page_number}_c{i}",
            }
            all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":
    from loader import load_domain

    pages = []
    pages.extend(load_domain("data/raw/finance", "finance"))
    pages.extend(load_domain("data/raw/legal",   "legal"))
    pages.extend(load_domain("data/raw/hr",      "hr"))

    chunks = chunk_pages(pages)

    print(f"\nTotal pages : {len(pages)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunks per page: {len(chunks)/len(pages):.1f}")

    sample = chunks[10]
    print(f"\n--- SAMPLE CHUNK ---")
    print(f"chunk_id   : {sample['chunk_id']}")
    print(f"source     : {sample['source']}")
    print(f"domain     : {sample['domain']}")
    print(f"page_number: {sample['page_number']}")
    print(f"text length: {len(sample['text'])} chars")
    print(f"text preview: {sample['text'][:200]}")