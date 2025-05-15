import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parsing.ocr_processor import extract_text_from_scanned_pdf

def parse_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith(".pdf"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file.replace(".pdf", "_ocr_text.txt"))

            print(f"🔍 OCR processing: {file}")
            text = extract_text_from_scanned_pdf(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"✅ Saved to: {output_path}")

if __name__ == "__main__":
    parse_folder("data/scanned_cvs", "outputs/scanned")
