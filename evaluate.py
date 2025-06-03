import os
import json
import re
from rapidfuzz import fuzz
def lowercase_keys_and_strings(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_key = k.lower()
            new_dict[new_key] = lowercase_keys_and_strings(v)
        return new_dict

    elif isinstance(obj, list):
        return [lowercase_keys_and_strings(elem) for elem in obj]

    elif isinstance(obj, str):
        return obj.strip().lower()

    else:
        return obj
def normalize_phone_number(phone_str):
    if not isinstance(phone_str, str):
        return ""
    digits_only = re.sub(r"\D", "", phone_str)
    return digits_only


def clean_experience_description(raw_desc):

    if not isinstance(raw_desc, str):
        return []

    # 1) Remove literal "\uXXXX" escapes
    no_unicode_escape = re.sub(r"\\u[0-9A-Fa-f]{4}", "", raw_desc)

    # 2) Split on actual newline (\n or \r\n) or literal "\\n"
    parts = re.split(r"(?:\r?\n|\\n)", no_unicode_escape)

    cleaned = []
    for part in parts:
        # 3) Remove leading bullets/spaces: "*", "»" (\u00bb), "–", "•", ">", "+", "=", etc.
        line = re.sub(r"^[\*\u00bb\-\–\•\>\+\=\s]+", "", part).strip()
        if line:
            cleaned.append(line)

    return cleaned
def normalize_date_field(raw_date):
    if not isinstance(raw_date, str) or not raw_date.strip():
        return ""

    cleaned = re.sub(r"[^0-9/\-]", "", raw_date)

    parts = re.split(r"[–\-]", cleaned)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) == 2:
        start, end = parts
        # If both sides match "MM/YYYY", return "MM/YYYY-MM/YYYY"
        if re.fullmatch(r"\d{2}/\d{4}", start) and re.fullmatch(r"\d{2}/\d{4}", end):
            return f"{start}-{end}"
        return cleaned

    # If only one part, check if "MM/YYYY"
    if parts and re.fullmatch(r"\d{2}/\d{4}", parts[0]):
        return parts[0]

    return cleaned
FIELD_NAME_MAP = {
    # Top-level fields (all lowercased)
    "name": "name",
    "mail": "email",      # some models use "mail" instead of "email"
    "email": "email",
    "phone": "phone",
    "location": "location",
    "linkedin": "linkedin",
    "skills": "skills",
    "education": "education",
    "experience": "experience",

    # Education subfields
    "degree": "degree",
    "university": "university",
    "city/location": "city/location",
    "city": "city/location",
    "location": "location",
    "duration": "duration",

    # Experience subfields
    "title": "title",
    "company": "company",
    "duration": "duration",
    "location": "location",
    "description": "description",            # GT key
    "full_description": "description",       # prediction key
}
def remap_field_names(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            mapped_key = FIELD_NAME_MAP.get(k, k)
            new_dict[mapped_key] = remap_field_names(v)
        return new_dict

    elif isinstance(obj, list):
        return [remap_field_names(elem) for elem in obj]

    else:
        return obj
def simple_text_match(gt_val, pred_val, field_name, fuzz_threshold=90):
    gt = (gt_val or "").strip().lower()
    pred = (pred_val or "").strip().lower()

    # Case A: both empty → match
    if not gt and not pred:
        return True

    # Case B: exact match
    if gt == pred:
        return True

    # Case C: fuzzy match for name/location
    if field_name in ["name", "location"] and gt and pred:
        score = fuzz.ratio(gt, pred)
        return score >= fuzz_threshold

    return False



def phone_match(gt_phone, pred_phone):
    return normalize_phone_number(gt_phone) == normalize_phone_number(pred_phone)

def skills_match(gt_skills, pred_skills):
    gt_set = set(gt_skills or [])
    pred_set = set(pred_skills or [])
    return gt_set == pred_set


def compute_field_metrics(counts):
    metrics = {}
    for field, c in counts.items():
        tp = c['tp']
        fp = c['fp']
        fn = c['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        metrics[field] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    return metrics

def aggregate_block_metrics(metrics, block_name):
    tp_sum = fp_sum = fn_sum = 0
    for field, vals in metrics.items():
        if field.startswith(block_name + "."):
            tp_sum += vals['tp']
            fp_sum += vals['fp']
            fn_sum += vals['fn']

    if tp_sum + fp_sum > 0:
        precision = tp_sum / (tp_sum + fp_sum)
    else:
        precision = 0.0

    if tp_sum + fn_sum > 0:
        recall = tp_sum / (tp_sum + fn_sum)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp_sum,
        'fp': fp_sum,
        'fn': fn_sum
    }
if __name__ == "__main__":

    GROUND_TRUTH_PATH = "evaluation/ground_truth.json"
    OUTPUTS_DIR       = "outputs"
    MODELS            = ["mistral", "llama3_instruct", "qwen_7b"]

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    ground_truth_map = {
        os.path.splitext(entry["filename"])[0]: entry
        for entry in ground_truth
    }
    print(f"✅ Loaded {len(ground_truth_map)} ground-truth records.\n")

    fields = [
        "name", "email", "phone", "location", "linkedin", "skills",
        "education.degree", "education.university", "education.city/location", "education.duration",
        "experience.title", "experience.company", "experience.duration", "experience.location", "experience.description"
    ]

    all_model_metrics = {}

    for model in MODELS:
        print(f"🔍 Evaluating model: {model}")
        model_output_dir = os.path.join(OUTPUTS_DIR, f"json_{model}")
        counts = {field: {'tp': 0, 'fp': 0, 'fn': 0} for field in fields}

        missing_preds = 0
        loaded_preds  = 0

        for base_filename, gt_entry in ground_truth_map.items():
            pred_filename = f"{base_filename}_{model}.json"
            pred_path     = os.path.join(model_output_dir, pred_filename)

            if not os.path.exists(pred_path):
                missing_preds += 1

                gt_lc   = lowercase_keys_and_strings(gt_entry)
                gt_norm = remap_field_names(gt_lc)

                # If a top‐level field is present in GT but missing in prediction, count FN
                for field in ["name", "email", "phone", "location", "linkedin"]:
                    gt_val = gt_norm.get(field, "")
                    gt_has = isinstance(gt_val, str) and bool(gt_val.strip())
                    if gt_has:
                        counts[field]['fn'] += 1

                # Skills
                gt_skills = gt_norm.get("skills", [])
                if isinstance(gt_skills, list) and len(gt_skills) > 0:
                    counts["skills"]['fn'] += 1

                # Education subfields
                gt_edu_list = gt_norm.get("education", []) or []
                for i in range(len(gt_edu_list)):
                    gt_item = gt_edu_list[i]
                    for sub in ["degree", "university", "city/location", "duration"]:
                        gt_sub = gt_item.get(sub, "")
                        gt_has = False
                        if sub == "duration":
                            gt_norm_date = normalize_date_field(gt_sub)
                            gt_has = bool(gt_norm_date)
                        else:
                            gt_has = isinstance(gt_sub, str) and bool(gt_sub.strip())
                        if gt_has:
                            counts[f"education.{sub}"]['fn'] += 1

                # Experience subfields
                gt_exp_list = gt_norm.get("experience", []) or []
                for i in range(len(gt_exp_list)):
                    gt_item = gt_exp_list[i]
                    for sub in ["title", "company", "duration", "location", "description"]:
                        gt_sub = gt_item.get(sub, "")
                        gt_has = False
                        if sub == "duration":
                            gt_norm_date = normalize_date_field(gt_sub)
                            gt_has = bool(gt_norm_date)
                        elif sub == "description":
                            gt_clean = clean_experience_description(gt_sub)
                            gt_has = bool(gt_clean)
                        else:
                            gt_has = isinstance(gt_sub, str) and bool(gt_sub.strip())
                        if gt_has:
                            counts[f"experience.{sub}"]['fn'] += 1

                continue  # move to next resume

            try:
                with open(pred_path, "r", encoding="utf-8") as f:
                    pred_entry = json.load(f)
                loaded_preds += 1
            except Exception as e:
                print(f"❌ Failed to load: {pred_filename} | Error: {e}")
                continue

            # Lowercase & remap GT and prediction
            gt_lc    = lowercase_keys_and_strings(gt_entry)
            gt_norm  = remap_field_names(gt_lc)

            pred_lc   = lowercase_keys_and_strings(pred_entry)
            pred_norm = remap_field_names(pred_lc)

            # Top‐level fields: name, email, phone, location, linkedin
            for field in ["name", "email", "phone", "location", "linkedin"]:
                gt_val   = gt_norm.get(field, "")
                pred_val = pred_norm.get(field, "")

                gt_has   = isinstance(gt_val, str) and bool(gt_val.strip())
                pred_has = isinstance(pred_val, str) and bool(pred_val.strip())

                if field in ["name", "location", "email", "linkedin"]:
                    matched = simple_text_match(gt_val, pred_val, field)
                elif field == "phone":
                    matched = phone_match(gt_val, pred_val)
                else:
                    matched = False

                if gt_has and pred_has:
                    if matched:
                        counts[field]['tp'] += 1
                    else:
                        counts[field]['fn'] += 1
                elif not gt_has and pred_has:
                    counts[field]['fp'] += 1
                elif gt_has and not pred_has:
                    counts[field]['fn'] += 1
                # else both empty: true negative → ignore

            # Skills
            gt_skills   = gt_norm.get("skills", [])
            pred_skills = pred_norm.get("skills", [])

            gt_has   = isinstance(gt_skills, list) and len(gt_skills) > 0
            pred_has = isinstance(pred_skills, list) and len(pred_skills) > 0
            matched = skills_match(gt_skills, pred_skills)

            if gt_has and pred_has:
                if matched:
                    counts["skills"]['tp'] += 1
                else:
                    counts["skills"]['fn'] += 1
            elif not gt_has and pred_has:
                counts["skills"]['fp'] += 1
            elif gt_has and not pred_has:
                counts["skills"]['fn'] += 1
            # else both empty: ignore

            # Education subfields
            gt_edu_list   = gt_norm.get("education", []) or []
            pred_edu_list = pred_norm.get("education", []) or []
            max_edu_len   = max(len(gt_edu_list), len(pred_edu_list))
            for i in range(max_edu_len):
                gt_item   = gt_edu_list[i]   if i < len(gt_edu_list)   else {}
                pred_item = pred_edu_list[i] if i < len(pred_edu_list) else {}
                for sub in ["degree", "university", "city/location", "duration"]:
                    gt_sub   = gt_item.get(sub, "")
                    pred_sub = pred_item.get(sub, "")

                    if sub == "duration":
                        gt_norm_date   = normalize_date_field(gt_sub)
                        pred_norm_date = normalize_date_field(pred_sub)
                        gt_has   = bool(gt_norm_date)
                        pred_has = bool(pred_norm_date)
                        matched  = (gt_norm_date == pred_norm_date)
                    else:
                        gt_has   = isinstance(gt_sub, str) and bool(gt_sub.strip())
                        pred_has = isinstance(pred_sub, str) and bool(pred_sub.strip())
                        matched  = simple_text_match(gt_sub, pred_sub, sub)

                    field_name = f"education.{sub}"
                    if gt_has and pred_has:
                        if matched:
                            counts[field_name]['tp'] += 1
                        else:
                            counts[field_name]['fn'] += 1
                    elif not gt_has and pred_has:
                        counts[field_name]['fp'] += 1
                    elif gt_has and not pred_has:
                        counts[field_name]['fn'] += 1
                    # else both empty: ignore

            # Experience subfields
            gt_exp_list   = gt_norm.get("experience", []) or []
            pred_exp_list = pred_norm.get("experience", []) or []
            max_exp_len   = max(len(gt_exp_list), len(pred_exp_list))
            for i in range(max_exp_len):
                gt_item   = gt_exp_list[i]   if i < len(gt_exp_list)   else {}
                pred_item = pred_exp_list[i] if i < len(pred_exp_list) else {}
                for sub in ["title", "company", "duration", "location", "description"]:
                    gt_sub   = gt_item.get(sub, "")
                    pred_sub = pred_item.get(sub, "")

                    if sub == "duration":
                        gt_norm_date   = normalize_date_field(gt_sub)
                        pred_norm_date = normalize_date_field(pred_sub)
                        gt_has   = bool(gt_norm_date)
                        pred_has = bool(pred_norm_date)
                        matched  = (gt_norm_date == pred_norm_date)
                    elif sub == "description":
                        gt_clean   = clean_experience_description(gt_sub)
                        pred_clean = clean_experience_description(pred_sub)
                        gt_has   = bool(gt_clean)
                        pred_has = bool(pred_clean)
                        matched  = (gt_clean == pred_clean)
                    else:  # title, company, location
                        gt_has   = isinstance(gt_sub, str) and bool(gt_sub.strip())
                        pred_has = isinstance(pred_sub, str) and bool(pred_sub.strip())
                        matched  = simple_text_match(gt_sub, pred_sub, sub)

                    field_name = f"experience.{sub}"
                    if gt_has and pred_has:
                        if matched:
                            counts[field_name]['tp'] += 1
                        else:
                            counts[field_name]['fn'] += 1
                    elif not gt_has and pred_has:
                        counts[field_name]['fp'] += 1
                    elif gt_has and not pred_has:
                        counts[field_name]['fn'] += 1
                    # else both empty: ignore

        # — Compute per‐subfield metrics for this model
        metrics = compute_field_metrics(counts)

        # — AGGREGATE “education” block ONCE and SAVE INTO metrics["education"]
        edu_block = aggregate_block_metrics(metrics, "education")
        metrics["education"] = edu_block  # now stored as a top‐level key

        # — AGGREGATE “experience” block ONCE and SAVE INTO metrics["experience"]
        exp_block = aggregate_block_metrics(metrics, "experience")
        metrics["experience"] = exp_block  # now stored as a top‐level key

        # — Store everything in all_model_metrics
        all_model_metrics[model] = metrics

        # — Print subfield results
        print(f"🔢 {model} → Predictions loaded: {loaded_preds} | Missing files: {missing_preds}")
        print("Field                          | Precision   | Recall      | F1-score    |  TP   FP   FN")
        print("-------------------------------|-------------|-------------|-------------|------------------------")
        for field in fields:
            m = metrics[field]
            print(f"{field:30}| "
                  f"{m['precision']:.2f}        | "
                  f"{m['recall']:.2f}        | "
                  f"{m['f1']:.2f}        | "
                  f"{m['tp']:>3}   {m['fp']:>3}   {m['fn']:>3}")
        print()

        # — Print the blocks (now read from metrics["education"] and metrics["experience"])
        edu_blk = metrics["education"]
        print(f"Block: education            | "
              f"{edu_blk['precision']:.2f}        | "
              f"{edu_blk['recall']:.2f}        | "
              f"{edu_blk['f1']:.2f}        | "
              f"{edu_blk['tp']:>3}   {edu_blk['fp']:>3}   {edu_blk['fn']:>3}")

        exp_blk = metrics["experience"]
        print(f"Block: experience           | "
              f"{exp_blk['precision']:.2f}        | "
              f"{exp_blk['recall']:.2f}        | "
              f"{exp_blk['f1']:.2f}        | "
              f"{exp_blk['tp']:>3}   {exp_blk['fp']:>3}   {exp_blk['fn']:>3}")

        print("\n")  # blank line before next model

    # — AFTER looping through all models, write out the combined metrics JSON
    output_path = os.path.join("evaluation", "analysis.json")
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(all_model_metrics, f_out, indent=2)
    print(f"\n✔ Saved all metrics (including block‐level) to {output_path}")
# if __name__ == "__main__":
#
#     GROUND_TRUTH_PATH = "evaluation/ground_truth.json"
#     OUTPUTS_DIR       = "outputs"
#     MODELS            = ["mistral", "llama3_instruct", "qwen_7b"]
#     with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
#         ground_truth = json.load(f)
#     ground_truth_map = {
#         os.path.splitext(entry["filename"])[0]: entry
#         for entry in ground_truth
#     }
#     print(f"✅ Loaded {len(ground_truth_map)} ground-truth records.\n")
#     fields = [
#         "name", "email", "phone", "location", "linkedin", "skills",
#         "education.degree", "education.university", "education.city/location", "education.duration",
#         "experience.title", "experience.company", "experience.duration", "experience.location", "experience.description"
#     ]
#     all_model_metrics = {}
#     for model in MODELS:
#         print(f"🔍 Evaluating model: {model}")
#         model_output_dir = os.path.join(OUTPUTS_DIR, f"json_{model}")
#         counts = {field: {'tp': 0, 'fp': 0, 'fn': 0} for field in fields}
#
#         missing_preds = 0
#         loaded_preds  = 0
#
#         for base_filename, gt_entry in ground_truth_map.items():
#             pred_filename = f"{base_filename}_{model}.json"
#             pred_path     = os.path.join(model_output_dir, pred_filename)
#
#             if not os.path.exists(pred_path):
#                 missing_preds += 1
#
#                 gt_lc   = lowercase_keys_and_strings(gt_entry)
#                 gt_norm = remap_field_names(gt_lc)
#
#                 for field in ["name", "email", "phone", "location", "linkedin"]:
#                     gt_val = gt_norm.get(field, "")
#                     gt_has = isinstance(gt_val, str) and bool(gt_val.strip())
#                     if gt_has:
#                         counts[field]['fn'] += 1
#
#                 # Skills
#                 gt_skills = gt_norm.get("skills", [])
#                 if isinstance(gt_skills, list) and len(gt_skills) > 0:
#                     counts["skills"]['fn'] += 1
#
#                 gt_edu_list = gt_norm.get("education", []) or []
#                 for i in range(len(gt_edu_list)):
#                     gt_item = gt_edu_list[i]
#                     for sub in ["degree", "university", "city/location", "duration"]:
#                         gt_sub = gt_item.get(sub, "")
#                         gt_has = False
#                         if sub == "duration":
#                             gt_norm_date = normalize_date_field(gt_sub)
#                             gt_has = bool(gt_norm_date)
#                         else:
#                             gt_has = isinstance(gt_sub, str) and bool(gt_sub.strip())
#
#                         if gt_has:
#                             counts[f"education.{sub}"]['fn'] += 1
#
#                 gt_exp_list = gt_norm.get("experience", []) or []
#                 for i in range(len(gt_exp_list)):
#                     gt_item = gt_exp_list[i]
#                     for sub in ["title", "company", "duration", "location", "description"]:
#                         gt_sub = gt_item.get(sub, "")
#                         gt_has = False
#                         if sub == "duration":
#                             gt_norm_date = normalize_date_field(gt_sub)
#                             gt_has = bool(gt_norm_date)
#                         elif sub == "description":
#                             gt_clean = clean_experience_description(gt_sub)
#                             gt_has = bool(gt_clean)
#                         else:
#                             gt_has = isinstance(gt_sub, str) and bool(gt_sub.strip())
#
#                         if gt_has:
#                             counts[f"experience.{sub}"]['fn'] += 1
#
#                 continue  # move to next resume
#             try:
#                 with open(pred_path, "r", encoding="utf-8") as f:
#                     pred_entry = json.load(f)
#                 loaded_preds += 1
#             except Exception as e:
#                 print(f"❌ Failed to load: {pred_filename} | Error: {e}")
#                 continue
#
#             # Lowercase & remap GT and prediction
#             gt_lc    = lowercase_keys_and_strings(gt_entry)
#             gt_norm  = remap_field_names(gt_lc)
#
#             pred_lc   = lowercase_keys_and_strings(pred_entry)
#             pred_norm = remap_field_names(pred_lc)
#
#             for field in ["name", "email", "phone", "location", "linkedin"]:
#                 gt_val   = gt_norm.get(field, "")
#                 pred_val = pred_norm.get(field, "")
#
#                 gt_has   = isinstance(gt_val, str) and bool(gt_val.strip())
#                 pred_has = isinstance(pred_val, str) and bool(pred_val.strip())
#
#                 if field in ["name", "location", "email", "linkedin"]:
#                     matched = simple_text_match(gt_val, pred_val, field)
#                 elif field == "phone":
#                     matched = phone_match(gt_val, pred_val)
#                 else:
#                     matched = False
#
#                 if gt_has and pred_has:
#                     if matched:
#                         counts[field]['tp'] += 1
#                     else:
#                         counts[field]['fn'] += 1
#                 elif not gt_has and pred_has:
#                     counts[field]['fp'] += 1
#                 elif gt_has and not pred_has:
#                     counts[field]['fn'] += 1
#                 # else both empty: true negative → ignore
#
#             gt_skills   = gt_norm.get("skills", [])
#             pred_skills = pred_norm.get("skills", [])
#
#             gt_has   = isinstance(gt_skills, list) and len(gt_skills) > 0
#             pred_has = isinstance(pred_skills, list) and len(pred_skills) > 0
#
#             matched = skills_match(gt_skills, pred_skills)
#
#             if gt_has and pred_has:
#                 if matched:
#                     counts["skills"]['tp'] += 1
#                 else:
#                     counts["skills"]['fn'] += 1
#             elif not gt_has and pred_has:
#                 counts["skills"]['fp'] += 1
#             elif gt_has and not pred_has:
#                 counts["skills"]['fn'] += 1
#             # else both empty: ignore
#
#             gt_edu_list   = gt_norm.get("education", []) or []
#             pred_edu_list = pred_norm.get("education", []) or []
#             max_edu_len   = max(len(gt_edu_list), len(pred_edu_list))
#
#             for i in range(max_edu_len):
#                 gt_item   = gt_edu_list[i]   if i < len(gt_edu_list)   else {}
#                 pred_item = pred_edu_list[i] if i < len(pred_edu_list) else {}
#
#                 for sub in ["degree", "university", "city/location", "duration"]:
#                     gt_sub   = gt_item.get(sub, "")
#                     pred_sub = pred_item.get(sub, "")
#
#                     if sub == "duration":
#                         gt_norm_date   = normalize_date_field(gt_sub)
#                         pred_norm_date = normalize_date_field(pred_sub)
#                         gt_has   = bool(gt_norm_date)
#                         pred_has = bool(pred_norm_date)
#                         matched  = (gt_norm_date == pred_norm_date)
#                     else:
#                         gt_has   = isinstance(gt_sub, str) and bool(gt_sub.strip())
#                         pred_has = isinstance(pred_sub, str) and bool(pred_sub.strip())
#                         matched  = simple_text_match(gt_sub, pred_sub, sub)
#
#                     field_name = f"education.{sub}"
#                     if gt_has and pred_has:
#                         if matched:
#                             counts[field_name]['tp'] += 1
#                         else:
#                             counts[field_name]['fn'] += 1
#                     elif not gt_has and pred_has:
#                         counts[field_name]['fp'] += 1
#                     elif gt_has and not pred_has:
#                         counts[field_name]['fn'] += 1
#                     # else both empty: ignore
#
#             gt_exp_list   = gt_norm.get("experience", []) or []
#             pred_exp_list = pred_norm.get("experience", []) or []
#             max_exp_len   = max(len(gt_exp_list), len(pred_exp_list))
#
#             for i in range(max_exp_len):
#                 gt_item   = gt_exp_list[i]   if i < len(gt_exp_list)   else {}
#                 pred_item = pred_exp_list[i] if i < len(pred_exp_list) else {}
#
#                 for sub in ["title", "company", "duration", "location", "description"]:
#                     gt_sub   = gt_item.get(sub, "")
#                     pred_sub = pred_item.get(sub, "")
#
#                     if sub == "duration":
#                         gt_norm_date   = normalize_date_field(gt_sub)
#                         pred_norm_date = normalize_date_field(pred_sub)
#                         gt_has   = bool(gt_norm_date)
#                         pred_has = bool(pred_norm_date)
#                         matched  = (gt_norm_date == pred_norm_date)
#
#                     elif sub == "description":
#                         gt_clean   = clean_experience_description(gt_sub)
#                         pred_clean = clean_experience_description(pred_sub)
#                         gt_has   = bool(gt_clean)
#                         pred_has = bool(pred_clean)
#                         matched  = (gt_clean == pred_clean)
#
#                     else:  # title, company, location
#                         gt_has   = isinstance(gt_sub, str) and bool(gt_sub.strip())
#                         pred_has = isinstance(pred_sub, str) and bool(pred_sub.strip())
#                         matched  = simple_text_match(gt_sub, pred_sub, sub)
#
#                     field_name = f"experience.{sub}"
#                     if gt_has and pred_has:
#                         if matched:
#                             counts[field_name]['tp'] += 1
#                         else:
#                             counts[field_name]['fn'] += 1
#                     elif not gt_has and pred_has:
#                         counts[field_name]['fp'] += 1
#                     elif gt_has and not pred_has:
#                         counts[field_name]['fn'] += 1
#                     # else both empty: ignore
#
#         # — Compute per‐subfield metrics for this model
#         metrics = compute_field_metrics(counts)
#         all_model_metrics[model] = metrics
#         # — Print subfield results
#         print(f"🔢 {model} → Predictions loaded: {loaded_preds} | Missing files: {missing_preds}")
#         print("Field                          | Precision   | Recall      | F1-score    |  TP   FP   FN")
#         print("-------------------------------|-------------|-------------|-------------|------------------------")
#         for field in fields:
#             m = metrics[field]
#             print(f"{field:30}| "
#                   f"{m['precision']:.2f}        | "
#                   f"{m['recall']:.2f}        | "
#                   f"{m['f1']:.2f}        | "
#                   f"{m['tp']:>3}   {m['fp']:>3}   {m['fn']:>3}")
#         print()
#
#         # — Aggregate “education” block
#         edu_block = aggregate_block_metrics(metrics, "education")
#         print(f"Block: education            | "
#               f"{edu_block['precision']:.2f}        | "
#               f"{edu_block['recall']:.2f}        | "
#               f"{edu_block['f1']:.2f}        | "
#               f"{edu_block['tp']:>3}   {edu_block['fp']:>3}   {edu_block['fn']:>3}")
#
#         # — Aggregate “experience” block
#         exp_block = aggregate_block_metrics(metrics, "experience")
#         print(f"Block: experience           | "
#               f"{exp_block['precision']:.2f}        | "
#               f"{exp_block['recall']:.2f}        | "
#               f"{exp_block['f1']:.2f}        | "
#               f"{exp_block['tp']:>3}   {exp_block['fp']:>3}   {exp_block['fn']:>3}")
#
#         print("\n")  # blank line before next model
#         all_model_block_metrics[model] = {
#             "education": edu_block,
#             "experience": exp_block
#         }
#         output_path = os.path.join("evaluation", "metrics.json")
#         with open(output_path, "w", encoding="utf-8") as f_out:
#             json.dump(all_model_metrics, f_out, indent=2)
#
#         print(f"\n✔ Saved all metrics to {output_path}")