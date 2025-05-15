import json
import requests
from pathlib import Path

# === CONFIGURATION ===
LOG_PATH = Path(__file__).parent / "metrics_log.json"
API_URL  = "http://localhost:11434/api/generate"
MODEL    = "mistral"

# === LOAD LOG ===
def load_log():
    return json.loads(LOG_PATH.read_text())

# === INTERPRETATION PROMPTS ===
def interpret_current(latest_metrics):
    prompt = (
        "You have evaluation metrics for three models, each with Precision, Recall, and F1 per field.\n"
        "Produce valid JSON only, without extra text, with keys:\n"
        "  summary: { modelName: <one-sentence summary>, … },\n"
        "  diagnostics: { modelName: [<fields with F1 <= 0.50>], … },\n"
        "  recommendations: { modelName: [<2 concrete next steps>], … }\n"
        "Use the following metrics (0.00–1.00). Values >=0.80 are strong; <=0.50 need improvement.\n\n"
        f"LATEST_METRICS:\n{json.dumps(latest_metrics, indent=2)}"
    )
    resp = requests.post(API_URL, json={"model": MODEL, "prompt": prompt, "options": {"temperature": 0.0}})
    # Try to parse JSON, otherwise return raw text
    try:
        return resp.json()
    except Exception:
        return resp.text


def interpret_comparison(previous_metrics, latest_metrics):
    prompt = (
        "Compare two sets of evaluation metrics. Return valid JSON only, no extra text, with keys:\n"
        "  deltas: { modelName: { field: <newF1 - oldF1>, … }, … },\n"
        "  improvements: { modelName: [<fields where F1 increased>], … },\n"
        "  regressions: { modelName: [<fields where F1 decreased>], … },\n"
        "  next_steps: { modelName: [<1 action per model>] }\n\n"
        f"PREVIOUS_METRICS:\n{json.dumps(previous_metrics, indent=2)}\n\n"
        f"LATEST_METRICS:\n{json.dumps(latest_metrics, indent=2)}"
    )
    resp = requests.post(API_URL, json={"model": MODEL, "prompt": prompt, "options": {"temperature": 0.0}})
    try:
        return resp.json()
    except Exception:
        return resp.text

# === ENTRY POINT ===
if __name__ == "__main__":
    log = load_log()
    latest   = log[-1]["scores"]
    previous = log[-2]["scores"] if len(log) > 1 else None

    print("🔹 SUMMARY OF LATEST RUN 🔹")
    summary = interpret_current(latest)
    print(json.dumps(summary, indent=2) if isinstance(summary, dict) else summary)

    if previous:
        print("\n🔹 COMPARISON WITH PREVIOUS RUN 🔹")
        comparison = interpret_comparison(previous, latest)
        print(json.dumps(comparison, indent=2) if isinstance(comparison, dict) else comparison)
