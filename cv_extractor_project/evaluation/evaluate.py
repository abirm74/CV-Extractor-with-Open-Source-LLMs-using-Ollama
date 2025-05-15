import os
import json
from datetime import datetime
from tabulate import tabulate

# === CONFIGURATION ===
GROUND_TRUTH_PATH = "evaluation/ground_truth.json"
MODEL_OUTPUTS_DIR = "outputs"
MODELS = ["mistral", "phi3:3.8b", "llama2:7b"]
FIELDS_TO_COMPARE = ["Name", "Email", "Phone", "LinkedIn", "Location", "Skills", "Education", "Experience"]
LOG_PATH = "evaluation/metrics_log.json"

# === LOAD GROUND TRUTH ===
def load_ground_truth(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# === PRECISION/RECALL/F1 ===
import json as _json

def compute_prf1(gt_list, pred_list):
    def serialize(item):
        return _json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item)

    gt_set = set(serialize(x) for x in gt_list)
    pred_set = set(serialize(x) for x in pred_list)

    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1

# === COMPARE A SINGLE RESUME ===
def compare_resume(gt_entry, pred_entry):
    comparison = {}
    for field in FIELDS_TO_COMPARE:
        gt = gt_entry.get(field)
        pred = pred_entry.get(field)
        if gt is None:
            continue
        if isinstance(gt, list) and isinstance(pred, list):
            p, r, f1 = compute_prf1(gt, pred)
            comparison[field] = {"precision": p, "recall": r, "f1": f1}
        else:
            match = 1.0 if gt == pred else 0.0
            comparison[field] = {"precision": match, "recall": match, "f1": match}
    return comparison

# === EVALUATE MODEL ===
def evaluate_model(model_name, ground_truth):
    safe_model = model_name.replace(':', '_')
    model_dir = os.path.join(MODEL_OUTPUTS_DIR, f"json_{safe_model}")
    field_scores = {f: {"precision": [], "recall": [], "f1": []} for f in FIELDS_TO_COMPARE}

    for entry in ground_truth:
        filename = entry["filename"].replace('.txt', f"_{safe_model}.json")
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            print(f"⚠️ Missing: {path}")
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                pred = json.load(f)
            result = compare_resume(entry, pred)
            for field, metrics in result.items():
                for m, v in metrics.items():
                    field_scores[field][m].append(v)
        except Exception as e:
            print(f"❌ Failed on {filename}: {e}")
    return field_scores

# === LOGGING ===
def append_metrics(log_path, all_scores):
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            log = json.load(f)
    else:
        log = []
    snapshot = {"timestamp": datetime.utcnow().isoformat() + 'Z', "scores": all_scores}
    log.append(snapshot)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)

# === PRINT RESULTS ===
def print_scores(model_name, field_scores):
    print(f"\n📊 Results for {model_name}")
    for field, metrics in field_scores.items():
        avg_p = sum(metrics['precision']) / len(metrics['precision']) if metrics['precision'] else 0.0
        avg_r = sum(metrics['recall'])    / len(metrics['recall'])    if metrics['recall']    else 0.0
        avg_f = sum(metrics['f1'])        / len(metrics['f1'])        if metrics['f1']        else 0.0
        print(f"  {field}: Precision={avg_p:.2f} Recall={avg_r:.2f} F1={avg_f:.2f}")

# === MARKDOWN TABLE ===
def print_markdown_table(all_scores):
    table, headers = [], ["Model","Field","Precision","Recall","F1"]
    for model, fscores in all_scores.items():
        for field, m in fscores.items():
            p = sum(m["precision"])/len(m["precision"]) if m["precision"] else 0.0
            r = sum(m["recall"])/len(m["recall"])       if m["recall"]    else 0.0
            f1= sum(m["f1"])/len(m["f1"])               if m["f1"]        else 0.0
            table.append([model, field, f"{p:.2f}", f"{r:.2f}", f"{f1:.2f}"])
    print(tabulate(table, headers=headers, tablefmt="github"))

# === MAIN ===
if __name__ == "__main__":
    gt = load_ground_truth(GROUND_TRUTH_PATH)
    all_scores = {}
    for model in MODELS:
        scores = evaluate_model(model, gt)
        all_scores[model] = scores
        print_scores(model, scores)
    append_metrics(LOG_PATH, all_scores)
    print_markdown_table(all_scores)
