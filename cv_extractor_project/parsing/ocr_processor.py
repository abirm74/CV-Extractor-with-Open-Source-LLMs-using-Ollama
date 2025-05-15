import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

def extract_text_from_scanned_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)  # High-res rendering
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        text = pytesseract.image_to_string(image)
        full_text += f"\n==== Page {page_num + 1} ====\n{text}\n"

    doc.close()
    return full_text
