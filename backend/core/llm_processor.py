# /backend/core/llm_processor.py

import subprocess
import json
import time
from typing import Dict, Any, Optional


class LLMProcessor:
    def __init__(self, default_model: str = "llama3:instruct", timeout: int = 120):
        self.default_model = default_model
        self.timeout = timeout

    def call_ollama(self, prompt: str, model: str, retries: int = 3) -> str:
        """Run the prompt through Ollama model and return JSON if successful."""
        for attempt in range(retries):
            result = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            output = result.stdout.decode("utf-8").strip()

            try:
                json_start = output.index('{')
                json_end = output.rindex('}') + 1
                candidate = output[json_start:json_end]
                json.loads(candidate)  # Validate JSON
                return candidate
            except Exception:
                print(f"⚠️ Attempt {attempt + 1} failed for {model}")
                time.sleep(1)

        return ""

    def build_prompt(self, resume_text: str) -> str:
        """Build the prompt for resume extraction."""
        return f"""
Extract the following fields from the resume below and return them as a flat valid JSON object. Only include the fields defined in the schema.

✅ Rules:
- Do NOT infer or reformat. Preserve the original text exactly (including special characters and formatting).
- Return empty string "" or empty list [] for optional/missing fields.
- Preserve all text exactly as-is from the original txt file.
- For experience.full_description, split on each newline or bullet ("-", "*", "•", etc.) so that each line/sentence becomes one array element.
- For experience.full_description, Return the whole sentence.
- Return only valid JSON, without markdown or comments.

🎯 JSON Schema:
{{
  "name": "...",
  "phone": "...",
  "mail": "...",
  "location": "...",
  "linkedin": "...",
  "education": [
    {{
      "degree": "...",
      "university": "...",
      "location": "...",
      "duration": "..."
    }}
  ],
  "experience": [
    {{
      "title": "<Job title — as written.>",
      "company": "<Company name — do not infer.>",
      "duration": "<Job duration — as listed, preserve format.>",
      "location": "<Location — directly extracted.>",
      "full_description": [
        "<First bullet or sentence — Full job description — MUST preserve full bullet points and line breaks exactly as in resume. Do not paraphrase, compress, or remove formatting.>",
        "<Second bullet or sentence — Full job description — MUST preserve full bullet points and line breaks exactly as in resume. Do not paraphrase, compress, or remove formatting.>",
        ...
      ]
    }}
  ],
  "skills": [
    "<Individual skill 1 — include short skills like 'SQL', 'R', etc.>",
    "<Individual skill 2>",
    ...
  ]
}}

--------------------
RESUME TEXT:
--------------------
\"\"\" 
{resume_text.strip()}
\"\"\"
"""

    def extract_resume_data(self, resume_text: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Extract structured data from resume text using LLM."""
        model = model or self.default_model
        prompt = self.build_prompt(resume_text)
        response = self.call_ollama(prompt, model)

        if not response:
            raise RuntimeError(f"Failed to get valid response from LLM model: {model}")

        return json.loads(response)