import fitz

def extract_text_from_pdf_sorted(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
        full_text += f"\n==== Page {page_num + 1} ====\n"
        for b in blocks:
            text = b[4].strip()
            if text:
                full_text += text + "\n"

    doc.close()
    return full_text
