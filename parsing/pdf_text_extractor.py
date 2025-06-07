# cv_extractor_project/parsing/pdf_text_extractor.py

import fitz  # PyMuPDF
from .text_cleaner import clean_extracted_text

def extract_text_from_pdf_sorted(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    raw_text = ""

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))

        raw_text += f"\n==== Page {page_num + 1} ====\n"
        for block in blocks:
            text = block[4].strip()
            if text:
                raw_text += text + "\n"

    doc.close()

    # Pipe through the enhanced cleaner
    clean_text = clean_extracted_text(raw_text)
    return clean_text
