import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import DATA_RAW_DIR, DOMAINS


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Opens a PDF and extracts text page by page.
    Returns a list of dicts — one per page — each containing:
      - text: the raw extracted text for that page
      - page_number: the page index (1-based)
      - source: the filename
      - domain: which folder it came from (finance/legal/hr)
    """
    domain = pdf_path.parent.name  # folder name = domain
    pages = []

    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Skip pages with almost no text (covers, blank pages, image-only pages)
        if len(text.strip()) < 50:
            continue

        pages.append({
            "text": text,
            "page_number": page_num + 1,  # human-readable: start from 1
            "source": pdf_path.name,
            "domain": domain,
        })

    doc.close()
    return pages


def load_all_documents() -> list[dict]:
    """
    Walks through all 3 domain folders and loads every PDF.
    Returns a flat list of page-level dicts across all documents.
    """
    all_pages = []

    for domain in DOMAINS:
        domain_path = DATA_RAW_DIR / domain
        pdf_files = list(domain_path.glob("*.pdf"))

        print(f"\n[{domain.upper()}] Found {len(pdf_files)} PDFs")

        for pdf_path in tqdm(pdf_files, desc=f"  Loading {domain}"):
            pages = extract_text_from_pdf(pdf_path)
            all_pages.extend(pages)
            print(f"    {pdf_path.name}: {len(pages)} pages extracted")

    print(f"\nTotal pages extracted across all documents: {len(all_pages)}")
    return all_pages


if __name__ == "__main__":
    pages = load_all_documents()
    # Print a sample from the first page of the first document
    if pages:
        sample = pages[0]
        print("\n--- SAMPLE OUTPUT ---")
        print(f"Source  : {sample['source']}")
        print(f"Domain  : {sample['domain']}")
        print(f"Page    : {sample['page_number']}")
        print(f"Text    : {sample['text'][:300]}...")