from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from pathlib import Path

from ..core.file_processor import FileProcessor
from ..core.llm_processor import LLMProcessor

router = APIRouter()
file_processor = FileProcessor()
llm_processor = LLMProcessor()


async def handle_resume_extraction(file: UploadFile, model: str = None) -> dict:
    if not file_processor.is_supported_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {file_processor.supported_extensions}"
        )

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
    try:
        content = await file.read()
        tmp_file.write(content)
        tmp_file.close()

        # Extract text
        extracted_text = file_processor.process_file(tmp_file.name)

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file")

        # Use LLM to get structured data
        structured_data = llm_processor.extract_resume_data(extracted_text, model=model)

        # Return ONLY the structured JSON data
        return structured_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        os.unlink(tmp_file.name)


@router.post("/upload")
async def process_resume(file: UploadFile = File(...)):
    """Process resume and return only JSON data"""
    result = await handle_resume_extraction(file)
    return {"resume_data": result}  # Clean, simple response