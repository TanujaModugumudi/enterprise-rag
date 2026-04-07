# src/ingestion/preprocessor.py

import os
from pathlib import Path
from src.ingestion.loader import load_pdf      # reuse your existing loader

# Where raw PDFs live, where cleaned text will go
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def preprocess_all_domains():
    """
    Walks data/raw/<domain>/<file>.pdf for every domain folder.
    Saves cleaned text to data/processed/<domain>/<file>.txt
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for domain_folder in RAW_DIR.iterdir():
        # Skip anything that isn't a folder (e.g. a stray .DS_Store file)
        if not domain_folder.is_dir():
            continue

        domain = domain_folder.name          # "finance", "legal", "hr"
        out_folder = PROCESSED_DIR / domain
        out_folder.mkdir(parents=True, exist_ok=True)

        pdf_files = list(domain_folder.glob("*.pdf"))
        print(f"\n[{domain.upper()}] Found {len(pdf_files)} PDFs")

        for pdf_path in pdf_files:
            print(f"  Processing: {pdf_path.name}")

            # load_pdf() already calls clean_text() on each page now
            docs = load_pdf(str(pdf_path), domain=domain)

            if not docs:
                print(f"    WARNING: 0 pages extracted — skipping")
                continue

            # Join all page texts with a separator so you can see page breaks
            full_text = "\n\n--- PAGE BREAK ---\n\n".join(
                doc.page_content for doc in docs
            )

            # Save as .txt with same stem as the PDF
            out_path = out_folder / (pdf_path.stem + ".txt")
            out_path.write_text(full_text, encoding="utf-8")

            print(f"    Saved: {out_path.name}  ({len(docs)} pages, {len(full_text):,} chars)")


if __name__ == "__main__":
    preprocess_all_domains()