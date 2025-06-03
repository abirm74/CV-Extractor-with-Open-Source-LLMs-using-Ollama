import json
import pandas as pd
import matplotlib.pyplot as plt

metrics_path = "evaluation/analysis.json"  # <-- changed from metrics.json
with open(metrics_path, "r", encoding="utf-8") as f:
    all_model_metrics = json.load(f)

atomic_fields = ["name", "location", "skills", "email", "linkedin", "phone"]
blocks = ["education", "experience"]

fields_to_keep = atomic_fields + blocks

models = list(all_model_metrics.keys())
df_precision = pd.DataFrame(index=fields_to_keep, columns=models, dtype=float)
df_recall = pd.DataFrame(index=fields_to_keep, columns=models, dtype=float)
df_f1 = pd.DataFrame(index=fields_to_keep, columns=models, dtype=float)

for model, mdata in all_model_metrics.items():
    for field in fields_to_keep:
        if field in mdata:
            # Assume each field has direct precision, recall, and f1 values at the top level
            df_precision.loc[field, model] = mdata[field].get("precision", 0.0)
            df_recall.loc[field, model] = mdata[field].get("recall", 0.0)
            df_f1.loc[field, model] = mdata[field].get("f1", 0.0)
        else:
            # If a field is missing, set to 0.0 explicitly
            df_precision.loc[field, model] = 0.0
            df_recall.loc[field, model] = 0.0
            df_f1.loc[field, model] = 0.0

# 7) Optional: Inspect the DataFrames
print("Precision by field/block and model:")
print(df_precision)
print("\nRecall by field/block and model:")
print(df_recall)
print("\nF₁ by field/block and model:")
print(df_f1)

plt.figure(figsize=(10, 5))
df_f1.plot(
    kind="bar",
    figsize=(10, 5),
    rot=45,
    ylabel="F₁-score",
    title="F₁-score for name, location, skills, email, linkedin, education, experience"
)
plt.xlabel("Field / Block")
plt.tight_layout()
plt.show()

# Save F1 scores to CSV
df_f1.to_csv("evaluation/f1_scores.csv")

# (Optional)
# plt.figure(figsize=(10, 5))
# df_precision.plot(kind="bar", figsize=(10, 5), rot=45, ylabel="Precision",
#                   title="Precision for name, location, ...")
# plt.xlabel("Field / Block")
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(10, 5))
# df_recall.plot(kind="bar", figsize=(10, 5), rot=45, ylabel="Recall",
#                title="Recall for name, location, ...")
# plt.xlabel("Field / Block")
# plt.tight_layout()
# plt.show()

df_f1.to_csv("evaluation/f1_scores.csv")
# df_precision.to_csv("evaluation/precision_scores.csv")
# df_recall.to_csv("evaluation/recall_scores.csv")
