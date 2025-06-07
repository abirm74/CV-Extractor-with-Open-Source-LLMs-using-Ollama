import os
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
import unicodedata
from typing import Dict, List, Any, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Setup (keeping your original structure intact)
ground_truth_path = 'evaluation/ground_truth.json'
outputs_folder = 'outputs'
models = ['llama3_instruct', 'mistral', 'qwen_7b']
model_dirs = {
    'llama3_instruct': 'json_llama3_instruct',
    'mistral': 'json_mistral',
    'qwen_7b': 'json_qwen_7b'
}

# Focus only on the fields you care about
TARGET_FIELDS = ['name', 'email', 'linkedin', 'phone', 'location', 'skills', 'education', 'experience']

# Field alias mappings for normalization
FIELD_ALIASES = {
    'email': ['mail', 'e-mail', 'e_mail', 'email_address'],
    'phone': ['phone_number', 'telephone', 'mobile', 'contact_number'],
    'name': ['full_name', 'candidate_name', 'applicant_name'],
    'location': ['address', 'city', 'location_info'],
    'linkedin': ['linkedin_url', 'linkedin_profile', 'linkedin_link'],
    'skills': ['skill_set', 'technical_skills', 'competencies'],
    'education': ['educational_background', 'academic_history'],
    'experience': ['work_experience', 'professional_experience', 'employment_history'],
    'description': ['full_description', 'job_description', 'responsibilities'],
    'duration': ['period', 'dates', 'time_period'],
    'company': ['employer', 'organization', 'workplace'],
    'university': ['institution', 'school', 'college'],
    'degree': ['qualification', 'diploma', 'certification']
}


def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing extra spaces, special chars, and lowercasing."""
    if not isinstance(text, str):
        return str(text).lower().strip()

    # Remove unicode artifacts and normalize
    text = unicodedata.normalize('NFKD', text)
    # Convert to lowercase and remove extra whitespace
    text = re.sub(r'\s+', ' ', text.lower().strip())
    # Remove common punctuation that might vary
    text = re.sub(r'[.,;:()[\]{}"\'-]', '', text)
    return text


def clean_description_text(text: str) -> str:
    """Clean description text by removing bullet points and formatting symbols."""
    if not isinstance(text, str):
        return str(text)

    # Remove common bullet point symbols and formatting
    text = re.sub(r'^[\s]*[-*•·◦▪▫‣⁃]+[\s]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+\.[\s]*', '', text, flags=re.MULTILINE)  # Remove numbered lists
    text = re.sub(r'^[\s]*[a-zA-Z]\)[\s]*', '', text, flags=re.MULTILINE)  # Remove lettered lists

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def normalize_field_name(field_name: str) -> str:
    """Normalize field names and handle aliases."""
    normalized = normalize_text(field_name)

    # Check if this field matches any known aliases
    for canonical_name, aliases in FIELD_ALIASES.items():
        if normalized == canonical_name or normalized in [normalize_text(alias) for alias in aliases]:
            return canonical_name

    return normalized


def compute_string_similarity(str1: str, str2: str, method='sequence_matcher') -> float:
    """Compute similarity between two strings using various methods."""
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0

    # Clean descriptions if they contain bullet points
    if method == 'jaccard':
        str1 = clean_description_text(str1)
        str2 = clean_description_text(str2)

    str1_norm = normalize_text(str1)
    str2_norm = normalize_text(str2)

    if str1_norm == str2_norm:
        return 1.0

    if method == 'sequence_matcher':
        return SequenceMatcher(None, str1_norm, str2_norm).ratio()
    elif method == 'jaccard':
        set1 = set(str1_norm.split())
        set2 = set(str2_norm.split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    return 0.0


def extract_skills_list(skills_data: Union[str, List[str], Dict]) -> List[str]:
    """Extract and normalize skills from various formats."""
    if isinstance(skills_data, str):
        # Split by common delimiters
        skills = re.split(r'[,;|\n•·]', skills_data)
        return [normalize_text(skill) for skill in skills if skill.strip()]
    elif isinstance(skills_data, list):
        return [normalize_text(str(skill)) for skill in skills_data if skill]
    elif isinstance(skills_data, dict):
        # Handle nested skills structure
        all_skills = []
        for key, value in skills_data.items():
            if isinstance(value, list):
                all_skills.extend([normalize_text(str(skill)) for skill in value])
            elif isinstance(value, str):
                all_skills.extend([normalize_text(skill) for skill in re.split(r'[,;|\n•·]', value) if skill.strip()])
        return all_skills
    return []


def compute_list_similarity(list1: List[str], list2: List[str], threshold=0.8) -> Tuple[float, int, int, float, float]:
    """Compute similarity between two lists of strings with fuzzy matching and return precision/recall."""
    if not list1 and not list2:
        return 1.0, 0, 0, 1.0, 1.0
    if not list1 or not list2:
        return 0.0, 0, max(len(list1), len(list2)), 0.0, 0.0

    matches = 0
    used_indices = set()

    for item1 in list1:
        best_match_score = 0
        best_match_idx = -1

        for idx, item2 in enumerate(list2):
            if idx in used_indices:
                continue

            similarity = compute_string_similarity(item1, item2)
            if similarity > best_match_score and similarity >= threshold:
                best_match_score = similarity
                best_match_idx = idx

        if best_match_idx != -1:
            matches += 1
            used_indices.add(best_match_idx)

    total_items = max(len(list1), len(list2))
    precision = matches / len(list2) if list2 else 0  # TP / (TP + FP)
    recall = matches / len(list1) if list1 else 0  # TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1, matches, total_items - matches, precision, recall


def find_best_field_match(gt_entry: Dict, pred_entry: Dict, field_candidates: List[str]) -> Tuple[str, Any]:
    """Find the best matching field from candidates in prediction."""
    best_field = None
    best_value = None

    for candidate in field_candidates:
        normalized_candidate = normalize_field_name(candidate)
        if normalized_candidate in pred_entry:
            return normalized_candidate, pred_entry[normalized_candidate]

        # Try exact match in keys
        for key in pred_entry.keys():
            if normalize_field_name(key) == normalized_candidate:
                return normalized_candidate, pred_entry[key]

    return None, None


def compute_precision_recall_for_field(gt_value: str, pred_value: str, threshold=0.8) -> Tuple[float, float]:
    """Compute precision and recall for a simple field."""
    if not gt_value and not pred_value:
        return 1.0, 1.0
    if not gt_value:
        return 0.0, 0.0  # No ground truth to match
    if not pred_value:
        return 0.0, 0.0  # No prediction made

    similarity = compute_string_similarity(str(gt_value), str(pred_value))

    # Binary classification: if similarity > threshold, it's a match
    if similarity >= threshold:
        precision = 1.0  # Predicted correctly
        recall = 1.0  # Found the ground truth
    else:
        precision = 0.0  # Predicted incorrectly
        recall = 0.0  # Didn't find the ground truth

    return precision, recall


def compare_nested_experience_education(gt_list: List[Dict], pred_list: List[Dict], field_type: str) -> Dict[
    str, float]:
    """Compare nested lists of experience or education entries."""
    if not gt_list and not pred_list:
        return {'overall': 1.0, 'count_match': 1.0, 'precision': 1.0, 'recall': 1.0}

    results = defaultdict(list)
    precision_scores = []
    recall_scores = []

    # Define expected subfields for each type
    expected_subfields = {
        'experience': ['title', 'company', 'location', 'duration', 'description'],
        'education': ['degree', 'university', 'city', 'duration']
    }

    subfields = expected_subfields.get(field_type, [])

    # Match entries using best alignment
    used_pred_indices = set()
    matched_entries = 0

    for gt_entry in gt_list:
        # Normalize ground truth entry
        gt_normalized = {normalize_field_name(k): v for k, v in gt_entry.items()}

        best_match_score = 0
        best_match_idx = -1
        best_subfield_scores = {}

        for pred_idx, pred_entry in enumerate(pred_list):
            if pred_idx in used_pred_indices:
                continue

            # Normalize predicted entry
            pred_normalized = {normalize_field_name(k): v for k, v in pred_entry.items()}

            # Compute subfield similarities for this pair
            subfield_scores = {}
            total_score = 0

            for subfield in subfields:
                gt_val = gt_normalized.get(subfield, '')
                pred_val = pred_normalized.get(subfield, '')

                # Handle special cases
                if subfield in ['description'] and (gt_val or pred_val):
                    similarity = compute_string_similarity(str(gt_val), str(pred_val), method='jaccard')
                else:
                    similarity = compute_string_similarity(str(gt_val), str(pred_val))

                subfield_scores[subfield] = similarity
                total_score += similarity

            avg_score = total_score / len(subfields) if subfields else 0

            if avg_score > best_match_score:
                best_match_score = avg_score
                best_match_idx = pred_idx
                best_subfield_scores = subfield_scores

        # If we found a reasonable match (lowered threshold for better matching)
        if best_match_score > 0.2:
            matched_entries += 1
            used_pred_indices.add(best_match_idx)

            # Record subfield scores
            for subfield, score in best_subfield_scores.items():
                results[subfield].append(score)

    # Compute precision and recall for nested structures
    true_positives = matched_entries
    false_positives = len(pred_list) - matched_entries
    false_negatives = len(gt_list) - matched_entries

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    # Compute final metrics
    final_results = {}
    for subfield in subfields:
        if results[subfield]:
            final_results[subfield] = sum(results[subfield]) / len(results[subfield])
        else:
            final_results[subfield] = 0.0

    # Overall metrics
    count_accuracy = matched_entries / max(len(gt_list), len(pred_list)) if (gt_list or pred_list) else 1.0
    overall_score = sum(final_results.values()) / len(final_results) if final_results else 0.0

    final_results['overall'] = overall_score
    final_results['count_match'] = count_accuracy
    final_results['precision'] = precision
    final_results['recall'] = recall

    return final_results


def compare_fields(gt_data: Dict, pred_data: Dict) -> Dict[str, Any]:
    """Compare all fields between ground truth and prediction."""
    results = {}

    # Normalize both dictionaries
    gt_normalized = {normalize_field_name(k): v for k, v in gt_data.items() if normalize_field_name(k) in TARGET_FIELDS}
    pred_normalized = {normalize_field_name(k): v for k, v in pred_data.items()}

    # Only evaluate target fields
    for field in TARGET_FIELDS:
        gt_value = gt_normalized.get(field, '')

        # Find the best matching field in prediction
        pred_value = pred_normalized.get(field, '')
        if not pred_value and field in FIELD_ALIASES:
            # Try aliases
            for alias in FIELD_ALIASES[field]:
                alias_norm = normalize_field_name(alias)
                if alias_norm in pred_normalized:
                    pred_value = pred_normalized[alias_norm]
                    break

        if field == 'skills':
            # Special handling for skills
            gt_skills = extract_skills_list(gt_value)
            pred_skills = extract_skills_list(pred_value)
            f1_score, matches, mismatches, precision, recall = compute_list_similarity(gt_skills, pred_skills)
            results[field] = {
                'score': f1_score,
                'type': 'list',
                'matches': matches,
                'mismatches': mismatches,
                'gt_count': len(gt_skills),
                'pred_count': len(pred_skills),
                'precision': precision,
                'recall': recall
            }

        elif field in ['experience', 'education']:
            # Special handling for nested structures
            gt_list = gt_value if isinstance(gt_value, list) else []
            pred_list = pred_value if isinstance(pred_value, list) else []

            nested_results = compare_nested_experience_education(gt_list, pred_list, field)
            results[field] = {
                'score': nested_results['overall'],
                'type': 'nested',
                'subfields': {k: v for k, v in nested_results.items() if
                              k not in ['overall', 'count_match', 'precision', 'recall']},
                'count_accuracy': nested_results['count_match'],
                'gt_count': len(gt_list),
                'pred_count': len(pred_list),
                'precision': nested_results['precision'],
                'recall': nested_results['recall']
            }

        else:
            # Simple field comparison
            similarity = compute_string_similarity(str(gt_value), str(pred_value))
            precision, recall = compute_precision_recall_for_field(str(gt_value), str(pred_value))
            results[field] = {
                'score': similarity,
                'type': 'simple',
                'gt_value': str(gt_value)[:100] + '...' if len(str(gt_value)) > 100 else str(gt_value),
                'pred_value': str(pred_value)[:100] + '...' if len(str(pred_value)) > 100 else str(pred_value),
                'precision': precision,
                'recall': recall
            }

    return results


def create_visualizations(all_results: Dict[str, Dict], output_dir: str = 'evaluation'):
    """Create comprehensive visualizations for the evaluation results."""

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Aggregate results by model for different metrics
    model_scores = defaultdict(lambda: defaultdict(list))
    model_precision = defaultdict(lambda: defaultdict(list))
    model_recall = defaultdict(lambda: defaultdict(list))

    for filename, file_results in all_results.items():
        for model, model_results in file_results.items():
            if model_results:
                for field, field_result in model_results.items():
                    if isinstance(field_result, dict) and 'score' in field_result:
                        model_scores[model][field].append(field_result['score'])
                        if 'precision' in field_result:
                            model_precision[model][field].append(field_result['precision'])
                        if 'recall' in field_result:
                            model_recall[model][field].append(field_result['recall'])

    # 1. Overall Model Performance Comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # F1 Scores
    overall_scores = {}
    for model in models:
        if model in model_scores:
            all_scores = []
            for field_scores in model_scores[model].values():
                all_scores.extend(field_scores)
            if all_scores:
                overall_scores[model] = np.mean(all_scores)

    if overall_scores:
        bars1 = ax1.bar(overall_scores.keys(), overall_scores.values(), alpha=0.8,
                        color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('Overall F1 Score by Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('F1 Score')
        ax1.set_ylim(0, 1)
        for bar, score in zip(bars1, overall_scores.values()):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # Precision
    overall_precision = {}
    for model in models:
        if model in model_precision:
            all_precision = []
            for field_precision in model_precision[model].values():
                all_precision.extend(field_precision)
            if all_precision:
                overall_precision[model] = np.mean(all_precision)

    if overall_precision:
        bars2 = ax2.bar(overall_precision.keys(), overall_precision.values(), alpha=0.8,
                        color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_title('Overall Precision by Model', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Precision')
        ax2.set_ylim(0, 1)
        for bar, score in zip(bars2, overall_precision.values()):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    # Recall
    overall_recall = {}
    for model in models:
        if model in model_recall:
            all_recall = []
            for field_recall in model_recall[model].values():
                all_recall.extend(field_recall)
            if all_recall:
                overall_recall[model] = np.mean(all_recall)

    if overall_recall:
        bars3 = ax3.bar(overall_recall.keys(), overall_recall.values(), alpha=0.8,
                        color=['skyblue', 'lightcoral', 'lightgreen'])
        ax3.set_title('Overall Recall by Model', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Recall')
        ax3.set_ylim(0, 1)
        for bar, score in zip(bars3, overall_recall.values()):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_model_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Field-wise Performance Heatmap
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Create matrices for heatmaps
    score_matrix = np.zeros((len(models), len(TARGET_FIELDS)))
    precision_matrix = np.zeros((len(models), len(TARGET_FIELDS)))
    recall_matrix = np.zeros((len(models), len(TARGET_FIELDS)))

    for i, model in enumerate(models):
        for j, field in enumerate(TARGET_FIELDS):
            if field in model_scores[model]:
                score_matrix[i, j] = np.mean(model_scores[model][field])
            if field in model_precision[model]:
                precision_matrix[i, j] = np.mean(model_precision[model][field])
            if field in model_recall[model]:
                recall_matrix[i, j] = np.mean(model_recall[model][field])

    # F1 Score heatmap
    sns.heatmap(score_matrix, annot=True, fmt='.3f', xticklabels=TARGET_FIELDS,
                yticklabels=models, cmap='RdYlBu_r', ax=ax1, vmin=0, vmax=1)
    ax1.set_title('F1 Score by Model and Field', fontsize=14, fontweight='bold')

    # Precision heatmap
    sns.heatmap(precision_matrix, annot=True, fmt='.3f', xticklabels=TARGET_FIELDS,
                yticklabels=models, cmap='RdYlBu_r', ax=ax2, vmin=0, vmax=1)
    ax2.set_title('Precision by Model and Field', fontsize=14, fontweight='bold')

    # Recall heatmap
    sns.heatmap(recall_matrix, annot=True, fmt='.3f', xticklabels=TARGET_FIELDS,
                yticklabels=models, cmap='RdYlBu_r', ax=ax3, vmin=0, vmax=1)
    ax3.set_title('Recall by Model and Field', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'field_wise_performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Performance Distribution Violin Plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, field in enumerate(TARGET_FIELDS):
        field_data = []
        field_labels = []

        for model in models:
            if field in model_scores[model]:
                field_data.extend(model_scores[model][field])
                field_labels.extend([model] * len(model_scores[model][field]))

        if field_data:
            df_field = pd.DataFrame({'Score': field_data, 'Model': field_labels})
            sns.violinplot(data=df_field, x='Model', y='Score', ax=axes[idx])
            axes[idx].set_title(f'{field.capitalize()} Performance', fontweight='bold')
            axes[idx].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Precision vs Recall Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']

    for i, model in enumerate(models):
        model_prec = []
        model_rec = []

        for field in TARGET_FIELDS:
            if field in model_precision[model] and field in model_recall[model]:
                model_prec.extend(model_precision[model][field])
                model_rec.extend(model_recall[model][field])

        if model_prec and model_rec:
            ax.scatter(model_prec, model_rec, c=colors[i], marker=markers[i],
                       label=model, alpha=0.6, s=60)

    ax.set_xlabel('Precision', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Precision vs Recall by Model', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add diagonal line for F1 score reference
    x = np.linspace(0, 1, 100)
    for f1 in [0.2, 0.4, 0.6, 0.8]:
        y = (f1 * x) / (2 * x - f1)
        y = np.where(y >= 0, y, np.nan)
        ax.plot(x, y, 'k--', alpha=0.3, linewidth=0.5)
        # Add F1 labels
        if f1 == 0.8:
            ax.text(0.9, 0.72, f'F1={f1}', rotation=45, alpha=0.7)

    plt.savefig(os.path.join(output_dir, 'precision_recall_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Model Comparison Radar Chart
    angles = np.linspace(0, 2 * np.pi, len(TARGET_FIELDS), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for model in models:
        scores = []
        for field in TARGET_FIELDS:
            if field in model_scores[model]:
                scores.append(np.mean(model_scores[model][field]))
            else:
                scores.append(0)
        scores += scores[:1]  # Complete the circle

        ax.plot(angles, scores, 'o-', linewidth=2, label=model)
        ax.fill(angles, scores, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(TARGET_FIELDS)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)

    plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Visualizations saved to {output_dir}/ directory:")
    print("   - overall_model_performance.png")
    print("   - field_wise_performance_heatmap.png")
    print("   - performance_distribution.png")
    print("   - precision_recall_scatter.png")
    print("   - radar_chart.png")


def generate_evaluation_report(all_results: Dict[str, Dict]) -> str:
    """Generate a comprehensive evaluation report with precision/recall metrics."""
    report = []
    report.append("=" * 80)
    report.append("RESUME EXTRACTION EVALUATION REPORT")
    report.append("=" * 80)

    # Aggregate results by model
    model_aggregates = defaultdict(lambda: defaultdict(list))
    model_precision_aggregates = defaultdict(lambda: defaultdict(list))
    model_recall_aggregates = defaultdict(lambda: defaultdict(list))
    model_subfield_aggregates = defaultdict(lambda: defaultdict(list))

    for filename, file_results in all_results.items():
        for model, model_results in file_results.items():
            if model_results:  # Only process if we have results
                for field, field_result in model_results.items():
                    if isinstance(field_result, dict) and 'score' in field_result:
                        model_aggregates[model][field].append(field_result['score'])

                        if 'precision' in field_result:
                            model_precision_aggregates[model][field].append(field_result['precision'])
                        if 'recall' in field_result:
                            model_recall_aggregates[model][field].append(field_result['recall'])

                        # Collect subfield scores for nested fields
                        if field_result.get('type') == 'nested' and 'subfields' in field_result:
                            for subfield, subscore in field_result['subfields'].items():
                                model_subfield_aggregates[model][f"{field}_{subfield}"].append(subscore)

    # Overall Model Performance Summary
    report.append("\n=== OVERALL MODEL PERFORMANCE ===")
    report.append("-" * 50)

    model_overall_scores = {}
    model_overall_precision = {}
    model_overall_recall = {}

    for model in models:
        if model in model_aggregates:
            # Calculate overall averages
            all_scores = []
            all_precision = []
            all_recall = []

            for field_scores in model_aggregates[model].values():
                all_scores.extend(field_scores)
            for field_precision in model_precision_aggregates[model].values():
                all_precision.extend(field_precision)
            for field_recall in model_recall_aggregates[model].values():
                all_recall.extend(field_recall)

            if all_scores:
                overall_avg = sum(all_scores) / len(all_scores)
                model_overall_scores[model] = overall_avg

            if all_precision:
                precision_avg = sum(all_precision) / len(all_precision)
                model_overall_precision[model] = precision_avg

            if all_recall:
                recall_avg = sum(all_recall) / len(all_recall)
                model_overall_recall[model] = recall_avg

    # Display overall metrics
    report.append(f"{'Model':<20} {'F1 Score':<10} {'Precision':<10} {'Recall':<10}")
    report.append("-" * 50)

    for model in models:
        f1_score = model_overall_scores.get(model, 0.0)
        precision = model_overall_precision.get(model, 0.0)
        recall = model_overall_recall.get(model, 0.0)
        report.append(f"{model.upper():<20} {f1_score:<10.3f} {precision:<10.3f} {recall:<10.3f}")

    # Field-wise Performance Summary
    report.append(f"\n=== FIELD-WISE PERFORMANCE ===")
    report.append("-" * 50)

    for field in TARGET_FIELDS:
        report.append(f"\n{field.upper()}:")
        report.append(f"{'Model':<15} {'F1':<8} {'Precision':<10} {'Recall':<8}")
        report.append("-" * 45)

        field_best_model = None
        field_best_score = -1

        for model in models:
            f1_score = 0.0
            precision = 0.0
            recall = 0.0

            if field in model_aggregates[model]:
                scores = model_aggregates[model][field]
                f1_score = sum(scores) / len(scores)

            if field in model_precision_aggregates[model]:
                prec_scores = model_precision_aggregates[model][field]
                precision = sum(prec_scores) / len(prec_scores)

            if field in model_recall_aggregates[model]:
                rec_scores = model_recall_aggregates[model][field]
                recall = sum(rec_scores) / len(rec_scores)

            report.append(f"{model:<15} {f1_score:<8.3f} {precision:<10.3f} {recall:<8.3f}")

            if f1_score > field_best_score:
                field_best_score = f1_score
                field_best_model = model

        if field_best_model:
            report.append(f"  --> Best: {field_best_model} (F1: {field_best_score:.3f})")

        # Show subfield performance for nested fields
        if field in ['experience', 'education']:
            expected_subfields = {
                'experience': ['title', 'company', 'location', 'duration', 'description'],
                'education': ['degree', 'university', 'city', 'duration']
            }

            subfields = expected_subfields.get(field, [])
            for subfield in subfields:
                subfield_key = f"{field}_{subfield}"
                report.append(f"\n  {subfield.upper()}:")

                for model in models:
                    if subfield_key in model_subfield_aggregates[model]:
                        scores = model_subfield_aggregates[model][subfield_key]
                        avg_score = sum(scores) / len(scores)
                        report.append(f"    {model:<13} : {avg_score:.3f}")
                    else:
                        report.append(f"    {model:<13} : N/A")

    # Best Model Summary
    report.append(f"\n=== BEST MODEL SUMMARY ===")
    report.append("-" * 50)
    if model_overall_scores:
        best_model = max(model_overall_scores.items(), key=lambda x: x[1])
        best_precision_model = max(model_overall_precision.items(),
                                   key=lambda x: x[1]) if model_overall_precision else ("N/A", 0)
        best_recall_model = max(model_overall_recall.items(), key=lambda x: x[1]) if model_overall_recall else (
        "N/A", 0)

        report.append(f"Overall Best F1 Score: {best_model[0].upper()} ({best_model[1]:.3f})")
        report.append(f"Best Precision: {best_precision_model[0].upper()} ({best_precision_model[1]:.3f})")
        report.append(f"Best Recall: {best_recall_model[0].upper()} ({best_recall_model[1]:.3f})")

        # Show where each model excels
        report.append(f"\nModel Strengths (F1 Score):")
        for field in TARGET_FIELDS:
            field_scores = {}
            for model in models:
                if field in model_aggregates[model]:
                    scores = model_aggregates[model][field]
                    field_scores[model] = sum(scores) / len(scores)

            if field_scores:
                best_field_model = max(field_scores.items(), key=lambda x: x[1])
                report.append(f"  {field:<12} : {best_field_model[0]} ({best_field_model[1]:.3f})")

    # Detailed file-by-file results
    report.append(f"\n=== DETAILED RESULTS BY FILE ===")
    report.append("-" * 50)

    for filename, file_results in all_results.items():
        report.append(f"\n{filename}")
        for model in models:
            if model in file_results and file_results[model]:
                model_scores = []
                report.append(f"  {model}:")
                for field in TARGET_FIELDS:
                    if field in file_results[model]:
                        result = file_results[model][field]
                        score = result['score']
                        precision = result.get('precision', 0.0)
                        recall = result.get('recall', 0.0)
                        model_scores.append(score)

                        if result['type'] == 'nested':
                            report.append(
                                f"    {field:<12} : F1={score:.3f}, P={precision:.3f}, R={recall:.3f} (count: {result.get('count_accuracy', 0):.3f})")
                            for subfield, subscore in result.get('subfields', {}).items():
                                report.append(f"      {subfield:<10} : {subscore:.3f}")
                        elif result['type'] == 'list':
                            report.append(
                                f"    {field:<12} : F1={score:.3f}, P={precision:.3f}, R={recall:.3f} ({result.get('matches', 0)}/{result.get('gt_count', 0)} matches)")
                        else:
                            report.append(f"    {field:<12} : F1={score:.3f}, P={precision:.3f}, R={recall:.3f}")

                if model_scores:
                    avg_score = sum(model_scores) / len(model_scores)
                    report.append(f"    {'AVG':<12} : {avg_score:.3f}")
            else:
                report.append(f"  {model}: No output found")

    return "\n".join(report)


# Load ground truth (keeping your original structure)
with open(ground_truth_path, 'r', encoding='utf-8') as f:
    ground_truth_data = json.load(f)

# Store all results for final reporting
all_results = {}

# Evaluate (keeping your original loop structure)
for entry in ground_truth_data:
    gt_filename = entry['filename']  # e.g. "Avery Rodriguez Resume.png_scanned_clean_ocr_text.txt"
    base_filename = os.path.splitext(gt_filename)[0]  # remove .txt

    print(f"\n🔍 Checking model outputs for: {base_filename}")

    file_results = {}

    for model in models:
        model_folder = os.path.join(outputs_folder, model_dirs[model])
        model_output_filename = f"{base_filename}_{model}.json"
        model_output_path = os.path.join(model_folder, model_output_filename)

        if os.path.exists(model_output_path):
            print(f"✅ Found: {model_output_path}")

            # Load and compare
            try:
                with open(model_output_path, 'r', encoding='utf-8') as f:
                    pred_data = json.load(f)

                # Perform comparison
                comparison_results = compare_fields(entry, pred_data)
                file_results[model] = comparison_results

                # Print quick summary
                if comparison_results:
                    avg_score = sum(
                        result['score'] for result in comparison_results.values()
                        if isinstance(result, dict) and 'score' in result
                    ) / len(comparison_results)
                    print(f"   📊 Average F1 score: {avg_score:.3f}")

            except Exception as e:
                print(f"   ❌ Error processing {model_output_path}: {str(e)}")
                file_results[model] = {}
        else:
            print(f"❌ Missing: {model_output_path}")
            file_results[model] = {}

    all_results[base_filename] = file_results

# Generate and save final report
print("\n" + "=" * 80)
print("GENERATING FINAL EVALUATION REPORT...")
print("=" * 80)

final_report = generate_evaluation_report(all_results)
print(final_report)

# Save report to file with UTF-8 encoding
report_path = os.path.join('evaluation', 'evaluation_report.txt')
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(final_report)

print(f"\n💾 Full report saved to: {report_path}")

# Save detailed results as JSON for further analysis
results_path = os.path.join('evaluation', 'detailed_results.json')
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"💾 Detailed results saved to: {results_path}")

# Generate visualizations
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS...")
print("=" * 80)

try:
    create_visualizations(all_results, 'evaluation')
except Exception as e:
    print(f"❌ Error generating visualizations: {str(e)}")
    print("Make sure you have matplotlib, seaborn, and pandas installed:")
    print("pip install matplotlib seaborn pandas numpy")

print("\n🎉 Evaluation complete!")