import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

LOG_PATH = Path(__file__).parent / "metrics_log.json"
PDF_PATH = Path(__file__).parent / "evaluation_report.pdf"

def load_log():
    with open(LOG_PATH) as f: return json.load(f)

def plot_field_trends(pdf, log, field):
    timestamps = [entry["timestamp"] for entry in log]
    for model in log[-1]["scores"].keys():
        f1s = [
            sum(entry["scores"][model][field]["f1"]) /
            len(entry["scores"][model][field]["f1"]) if entry["scores"][model][field]["f1"] else 0
            for entry in log
        ]
        plt.plot(timestamps, f1s, marker='o', label=model)
    plt.title(f"F1 Trend for {field}")
    plt.xticks(rotation=45)
    plt.ylabel("F1 Score")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.clf()

def build_pdf():
    log = load_log()
    with PdfPages(PDF_PATH) as pdf:
        # Cover page
        plt.figure(figsize=(8,6))
        plt.text(0.5, 0.5, "Resume Extraction Evaluation Report",
                 ha="center", va="center", size=24)
        pdf.savefig(); plt.close()

        # One page per field
        for field in log[-1]["scores"].keys():
            plot_field_trends(pdf, log, field)

    print(f"▶ PDF report written to {PDF_PATH}")

if __name__=="__main__":
    build_pdf()
