import os
import csv
import uuid
import datetime
from cleaning_utils import clean_text  # your existing cleaner
from data_loader import load_text_file  # assume this reads a txt file

# === CONFIG ===
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw_policies"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "paragraphs.csv"


# === HELPERS ===
def split_into_paragraphs(text, min_len=100):
    """Split raw text into roughly coherent paragraphs."""
    parts = [p.strip() for p in text.split("\n\n") if len(p.strip()) > min_len]
    return parts

# === MAIN PIPELINE ===
def build_policy_dataset():
    rows = []
    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".txt"):
            continue
        source = filename.replace(".txt", "")
        raw_text = load_text_file(os.path.join(RAW_DIR, filename))
        cleaned = clean_text(raw_text)
        paragraphs = split_into_paragraphs(cleaned)

        for i, paragraph in enumerate(paragraphs, start=1):
            row = {
                "id": f"{source}_{i:03d}",
                "source": source,
                "section": "",
                "policy_text": paragraph,
                "gdpr_article": "",
                "gdpr_principle": "",
                "compliance_label": "",
                "explanation": "",
                "keywords_detected": "",
                "annotator": "",
                "confidence": "",
                "source_url": "",
                "date_annotated": datetime.date.today().isoformat(),
            }
            rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved {len(rows)} paragraphs to {OUTPUT_CSV}")

if __name__ == "__main__":
    build_policy_dataset()
