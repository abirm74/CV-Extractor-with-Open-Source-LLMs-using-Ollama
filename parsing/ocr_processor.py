# cv_extractor_project/parsing/ocr_processor.py

import fitz
import pytesseract
from PIL import Image
import io
from .text_cleaner import clean_extracted_text

def extract_text_from_scanned_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    raw_text = ""

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        text = pytesseract.image_to_string(image, lang="eng")
        raw_text += f"\n==== Page {page_num + 1} ====\n{text}\n"

    doc.close()

    # Pipe through the enhanced cleaner
    clean_text = clean_extracted_text(raw_text)
    return clean_text
