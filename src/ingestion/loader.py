# src/ingestion/loader.py

import fitz  # PyMuPDF
import re
from pathlib import Path
from langchain.schema import Document


def clean_text(text: str) -> str:
    """
    Removes common PDF extraction noise from a page's text.
    Called on each page's text before it is stored as a Document.
    """

    # 1. Remove page number patterns like "Page 42", "Page 42 of 210"
    text = re.sub(r'Page\s+\d+(\s+of\s+\d+)?', '', text, flags=re.IGNORECASE)

    # 2. Remove "42 | 210" style page references
    text = re.sub(r'\b\d{1,3}\s*\|\s*\d{1,3}\b', '', text)

    # 3. Remove figure/table captions like "Figure 3.2:", "Table 4:", "Fig. 2 —"
    text = re.sub(r'(Figure|Fig\.?|Table|Chart|Exhibit)\s*[\d\.]+[\s:\-–]*', '', text, flags=re.IGNORECASE)
    
    # 4. Remove running headers — company name + report/annual + year
    text = re.sub(r'(Infosys|TCS|Wipro).{0,40}(Report|Annual|Limited)\s*\d{4}[-–]?\d{0,2}', '', text, flags=re.IGNORECASE)

    # 5. Remove generic "Integrated Annual Report YYYY-YY" headers without company name
    text = re.sub(r'Integrated Annual Report\s*\d{4}[-–]?\d{0,2}', '', text, flags=re.IGNORECASE)

    # 6. Remove Wipro page navigation strip labels
    text = re.sub(r'^\s*(Overview|Governance|Financials|Stakeholder Value|Annexures)\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)

    # 7. Remove "Environment & Climate" navigation label
    text = re.sub(r'^\s*Environment\s*&\s*Climate\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)

    # 8. Remove standalone FY labels — "FY", "FY22", "FY 2024", "2021*", "2024*"
    text = re.sub(r'^\s*(FY\s*\d{0,4}\*?|\d{4}\*?)\s*$', '', text, flags=re.MULTILINE)

    # 9. Remove stray footnote markers — lone asterisks, "5*" on their own line
    text = re.sub(r'^\s*[\d]*\s*\*+\s*$', '', text, flags=re.MULTILINE)

    # 10. Remove standalone checkmarks and tick symbols
    text = re.sub(r'^\s*[✓✔√☑]\s*$', '', text, flags=re.MULTILINE)
    
    # 10. Remove lines that contain only a number (standalone page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # 11. Collapse 3 or more blank lines into a single blank line
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 12. Collapse multiple spaces or tabs within a line into one space
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # 13. Strip leading and trailing whitespace from the whole page
    text = text.strip()

    return text


def load_pdf(file_path: str, domain: str = "general") -> list[Document]:
    """
    Opens a PDF, extracts text page by page, cleans each page,
    and returns a list of LangChain Document objects with metadata.

    Args:
        file_path: full path to the PDF file
        domain:    category label — "finance", "legal", or "hr"

    Returns:
        list of Document objects, one per usable page
    """

    documents = []
    path = Path(file_path)

    try:
        pdf = fitz.open(file_path)
    except Exception as e:
        print(f"  ERROR opening {path.name}: {e}")
        return []

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text()        # extract raw text from this page
        text = clean_text(text)       # <-- NEW: clean before storing

        # Skip pages that are too short to be useful (images, blank pages, etc.)
        if len(text) < 50:
            continue

        doc = Document(
            page_content=text,
            metadata={
                "source":      path.name,
                "domain":      domain,
                "page_number": page_num + 1,   # human-readable (1-indexed)
            }
        )
        documents.append(doc)

    pdf.close()

    print(f"  Loaded: {path.name}  →  {len(documents)} pages extracted")
    return documents


def load_domain(domain_path: str, domain: str) -> list[Document]:
    """
    Loads all PDFs inside a domain folder.

    Args:
        domain_path: path to the folder, e.g. "data/raw/finance"
        domain:      label to attach to each document's metadata

    Returns:
        combined list of Documents from all PDFs in that folder
    """

    all_docs = []
    folder = Path(domain_path)

    pdf_files = list(folder.glob("*.pdf"))
    print(f"\n[{domain.upper()}] Found {len(pdf_files)} PDFs")

    for pdf_file in pdf_files:
        docs = load_pdf(str(pdf_file), domain=domain)
        all_docs.extend(docs)

    print(f"[{domain.upper()}] Total pages loaded: {len(all_docs)}")
    return all_docs