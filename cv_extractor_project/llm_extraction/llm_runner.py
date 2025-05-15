import os
import subprocess
import json
import time

# Base directory relative to this script (llm_extraction/)
base_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
subdirs = ['text', 'scanned']
#models = ['mistral', 'phi', 'llama2']
models = ['phi3:3.8b','llama2:7b']

def call_ollama(prompt: str, model: str = "mistral", retries=2) -> str:
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
    """Build the LLM prompt for extraction."""
    return f"""
Extract the following fields from the resume text below:

- Name
- Email
- Phone
- Location
- LinkedIn (optional)
- Education (list: Degree, University, City/Location, Duration or Start/End)
- Experience (list: Title, Company, Duration, Location, Description)
- Skills 
- Awards (optional)

Return a valid JSON object.
Do NOT include any fields that are missing from the text.
Do NOT include explanations or comments.
Only output the JSON object.

Resume:
\"\"\"
{resume_text}
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
                            continue

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
