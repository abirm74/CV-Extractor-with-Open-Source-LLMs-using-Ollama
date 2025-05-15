import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add project root to Python path
from parsing.pdf_text_extractor import extract_text_from_pdf_sorted
def parse_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith(".pdf"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file.replace(".pdf", "_raw_text.txt"))

            print(f"📄 Extracting: {file}")
            text = extract_text_from_pdf_sorted(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Saved to: {output_path}")

if __name__ == "__main__":
    parse_folder("data/text_cvs", "outputs/text")
