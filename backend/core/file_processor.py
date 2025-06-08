# /backend/core/file_processor.py


from pathlib import Path
import fitz


# Import existing parsing modules
from parsing.pdf_text_extractor import extract_text_from_pdf_sorted
from parsing.ocr_processor import extract_text_from_scanned_pdf

class FileProcessor:

    def __init__(self):
        self.supported_extensions = {'.pdf'}

    def is_supported_file(self, filename: str) -> bool:
        """Check if file extension is supported"""
        return Path(filename).suffix.lower() in self.supported_extensions

    def detect_pdf_type(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text().strip()
            if text:
                return 'text'  # Found selectable text
        return 'scanned'
    def process_file(self, file_path: str, file_type: str = None) -> str:
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            # Determine PDF type if not specified
            if file_type is None:
                file_type = self.detect_pdf_type(file_path)

            if file_type == 'text':
                return extract_text_from_pdf_sorted(file_path)
            else:
                return extract_text_from_scanned_pdf(file_path)

        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
