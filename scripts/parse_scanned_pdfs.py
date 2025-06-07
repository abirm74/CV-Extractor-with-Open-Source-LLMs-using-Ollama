# cv_extractor_project/scripts/parse_scanned_pdfs.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parsing.ocr_processor import extract_text_from_scanned_pdf

def parse_folder(input_folder: str, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith(".pdf"):
            input_path = os.path.join(input_folder, file)
            output_name = file.replace(".pdf", "_clean_ocr_text.txt")
            output_path = os.path.join(output_folder, output_name)

            print(f"🔍 OCR & cleaning: {file}")
            text = extract_text_from_scanned_pdf(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"✅ Saved cleaned OCR text to: {output_path}")

if __name__ == "__main__":
    parse_folder("data/scanned_cvs", "outputs/scanned")
