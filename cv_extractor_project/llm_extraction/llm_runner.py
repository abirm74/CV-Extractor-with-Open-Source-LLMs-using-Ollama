import os
import subprocess
import json
import time

# Base directory relative to this script (llm_extraction/)
base_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
subdirs = ['text', 'scanned']
#models = ['mistral', 'phi', 'llama2']
#models = ['phi3:3.8b','llama2:7b']
#models = ['mistral','llama3:instruct','qwen:7b','phi3:3.8b ']
models = ['phi3:3.8b']
def call_ollama(prompt: str, model: str = "mistral",retries=3) -> str:
    """Call the Ollama LLM model with the given prompt."""
    for attempt in range(retries):
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output = result.stdout.decode("utf-8").strip()

        # Attempt to extract valid JSON
        try:
            json_start = output.index('{')
            json_end = output.rindex('}') + 1
            output = output[json_start:json_end]
            json.loads(output)  # Validate
            return output
        except Exception:
            print(f"⚠️ Attempt {attempt + 1} failed for {model}: invalid JSON")
            time.sleep(1)

    return ""

def build_prompt(resume_text: str) -> str:
    return f"""
Your task is to extract structured information from the resume below into a strict flat JSON object.
   
Y ou MUST follow the format and extraction rules exactly as described per field.

Do not include any fields not listed below. If a field is optional and not found, return an empty string "" or empty list [] as specified.

Character replacement rule: If and only if you encounter the exact character `�`, replace only that character with `[UNK]` (including brackets and uppercase letters). Do **not** alter or remove adjacent characters or valid digits. Do **not** change line breaks, bullet points, formatting, or any other text.

Return only valid JSON, without markdown or comments.

OUTPUT JSON FORMAT:
{{
  "name": "<Extracted full name — do not infer, must be directly stated.>",
  "phone": "<Extracted phone number — preserve exactly as written in the resume. If a � character appears, replace only that character with [UNK]. Do not modify or format anything else. Do not replace digits.>",
  "mail": "<Extracted email address — OPTIONAL. If not present, return empty string \"\">",
  "location": "<Extracted city and/or region — must match resume verbatim.>",
  "linkedin": "<Extracted LinkedIn URL — OPTIONAL. If not present, return empty string \"\">",
  "education": [
    {{
      "degree": "<Degree name — directly extracted, do not infer.>",
      "university": "<University or institution name — verbatim.>",
      "location": "<City/Location of the university — verbatim.>",
      "duration": "<Date range or period as stated — preserve original format.Replace all � with [UNK].>"
    }}
  ],
  "experience": [
    {{
      "title": "<Job title — as written.>",
      "company": "<Company name — do not infer.>",
      "duration": "<Job duration — as listed, preserve format.>",
      "location": "<Location — directly extracted.>",
      "full_description": "<Full job description — MUST preserve full bullet points and line breaks exactly as in resume. Do not paraphrase, compress, or remove formatting. Replace only � with [UNK]. Leave all other characters and formatting untouched.>"
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

# Process each model independently
for model in models:
    print(f"\n🔁 Running extraction using model: {model}")
    safe_model_name = model.replace(":", "_")
    model_output_dir = os.path.join(base_dir, f'json_{safe_model_name}')
    os.makedirs(model_output_dir, exist_ok=True)

    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)

        for dirpath, _, filenames in os.walk(subdir_path):
            for filename in filenames:
                if filename.lower().endswith('.txt'):
                    file_path = os.path.join(dirpath, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            resume_text = f.read().strip()

                        if not resume_text:
                            print(f"⚠️ Empty file skipped: {filename}")
                            continue

                        prompt = build_prompt(resume_text)
                        print(f"\n📄 Processing: {filename}")
                        response = call_ollama(prompt, model=model)

                        if not response:
                            print(f"❌ Skipped {filename}: No valid JSON from {model}")

                            # Create skipped_outputs folder if it doesn't exist
                            skipped_dir = os.path.join(base_dir, "skipped_outputs")
                            os.makedirs(skipped_dir, exist_ok=True)

                            # Log the filename and model to a retry log
                            with open(os.path.join(skipped_dir, "skipped_files.log"), "a", encoding="utf-8") as logf:
                                logf.write(f"{filename} | {model}\n")

                            # Save the prompt and model output (if any) to a debug file
                            raw_output_path = os.path.join(skipped_dir, f"{filename}_{safe_model_name}_raw.txt")
                            with open(raw_output_path, "w", encoding="utf-8") as rawf:
                                rawf.write("==== PROMPT ====\n")
                                rawf.write(prompt)
                                rawf.write("\n\n==== MODEL OUTPUT ====\n")
                                rawf.write(response or "[No output received]")

                            continue  # Skip to the next file

                        try:
                            parsed_json = json.loads(response)
                            #json_filename = filename.replace('.txt', f'_{model}.json')
                            json_filename = filename.replace('.txt', f'_{safe_model_name}.json')
                            output_path = os.path.join(model_output_dir, json_filename)

                            with open(output_path, 'w', encoding='utf-8') as out_f:
                                json.dump(parsed_json, out_f, indent=2)
                            print(f"✅ Saved: {json_filename}")
                        except json.JSONDecodeError:
                            print(f"❌ Final JSON decode failed for: {filename} ({model})")
                            print(response[:500])

                    except Exception as e:
                        print(f"❗ Error reading {file_path}: {e}")
