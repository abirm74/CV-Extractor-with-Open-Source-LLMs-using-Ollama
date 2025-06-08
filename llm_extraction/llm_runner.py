#/llm_extraction/llm_runner.py
import os
import subprocess
import json
import time
import glob

rerun_skipped = True  # Set to False if you want to skip rerun
max_reruns = 2
# Base directory relative to this script (llm_extraction/)
base_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
subdirs = ['text', 'scanned']
#models = ['llama3:instruct','qwen:7b','mistral']
models = ['qwen:7b']

def call_ollama(prompt: str, model: str, retries=3) -> str:
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
            print(f"⚠️ Attempt {attempt+1} failed for {model}")
            time.sleep(1)

    return ""
def build_prompt(resume_text: str) -> str:
    return f"""
Extract the following fields from the resume below and return them as a flat valid JSON object. Only include the fields defined in the schema.

✅ Rules:
- Do NOT infer or reformat. Preserve the original text exactly (including special characters and formatting).
- Return empty string "" or empty list [] for optional/missing fields.
- Preserve all text exactly as-is from the original txt file.
- For experience.full_description, split on each newline or bullet (“-”, “*”, “•”, etc.) so that each line/sentence becomes one array element.
- For experience.full_description, Return the whole sentence.
- Return only valid JSON, without markdown or comments..

🎯 JSON Schema:
{{
  "name": "...",
  "phone": "...",
  "mail": "...",                      // OPTIONAL
  "location": "...",
  "linkedin": "...",                  // OPTIONAL
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


if not rerun_skipped:
    for model in models:
        print(f"\n🔁 Model: {model}")
        safe_model_name = model.replace(":", "_")
        model_output_dir = os.path.join(base_dir, f'json_{safe_model_name}')
        os.makedirs(model_output_dir, exist_ok=True)

        for subdir in subdirs:
            subdir_path = os.path.join(base_dir, subdir)

            for dirpath, _, filenames in os.walk(subdir_path):
                for filename in filenames:
                    if not filename.lower().endswith('.txt'):
                        continue

                    file_path = os.path.join(dirpath, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            resume_text = f.read()

                        if not resume_text.strip():
                            print(f"⚠️ Empty file skipped: {filename}")
                            continue

                        prompt = build_prompt(resume_text)
                        print(f"📄 Processing: {filename}")
                        response = call_ollama(prompt, model=model)

                        if not response:
                            print(f"❌ Failed: {filename}")
                            skipped_dir = os.path.join(base_dir, "skipped_outputs", safe_model_name)
                            os.makedirs(skipped_dir, exist_ok=True)

                            with open(os.path.join(skipped_dir, "skipped_files.log"), "a", encoding="utf-8") as logf:
                                logf.write(f"{filename}\n")

                            with open(os.path.join(skipped_dir, f"{filename}_prompt.txt"), "w", encoding="utf-8") as pf:
                                pf.write(prompt)
                            with open(os.path.join(skipped_dir, f"{filename}_output.txt"), "w", encoding="utf-8") as of:
                                of.write(response or "[No output]")

                            continue

                        parsed_json = json.loads(response)
                        json_filename = filename.replace('.txt', f'_{safe_model_name}.json')
                        output_path = os.path.join(model_output_dir, json_filename)

                        with open(output_path, 'w', encoding='utf-8') as out_f:
                            json.dump(parsed_json, out_f, indent=2)
                        print(f"✅ Saved: {json_filename}")

                    except Exception as e:
                        print(f"❗ Error processing {filename}: {e}")

else:
    for model in models:
        print(f"\n🔁 Rerunning skipped for model: {model}")
        safe_model_name = model.replace(":", "_")
        skipped_dir = os.path.join(base_dir, "skipped_outputs", safe_model_name)
        skipped_log = os.path.join(skipped_dir, "skipped_files.log")

        if not os.path.exists(skipped_log):
            print(f"📭 No skipped files found for model: {model}")
            continue

        with open(skipped_log, 'r', encoding='utf-8') as f:
            skipped_files = [line.strip() for line in f.readlines() if line.strip()]

        if not skipped_files:
            print(f"📭 Skipped log is empty for model: {model}")
            continue

        model_output_dir = os.path.join(base_dir, f'json_{safe_model_name}')
        os.makedirs(model_output_dir, exist_ok=True)

        for subdir in subdirs:
            subdir_path = os.path.join(base_dir, subdir)

            for filename in skipped_files:
                file_path = os.path.join(subdir_path, filename)
                if not os.path.exists(file_path):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        resume_text = f.read()

                    if not resume_text.strip():
                        print(f"⚠️ Empty file still skipped: {filename}")
                        continue

                    prompt = build_prompt(resume_text)
                    print(f"♻️ Retrying: {filename}")
                    response = call_ollama(prompt, model=model)

                    if not response:
                        print(f"❌ Retry failed: {filename}")
                        continue  # Don't re-log, already logged earlier

                    parsed_json = json.loads(response)
                    json_filename = filename.replace('.txt', f'_{safe_model_name}.json')
                    output_path = os.path.join(model_output_dir, json_filename)

                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        json.dump(parsed_json, out_f, indent=2)
                    print(f"✅ Retry succeeded: {json_filename}")

                except Exception as e:
                    print(f"❗ Error retrying {filename}: {e}")