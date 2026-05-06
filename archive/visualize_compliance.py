import os
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
BASE_DIR = "/Users/davidj.silva/Desktop/Data Science/Thesis/Project/data"
INPUT_FILE = os.path.join(BASE_DIR, "annotated", "combined_compliance_scores.csv")
OUTPUT_IMG = os.path.join(BASE_DIR, "annotated", "compliance_distribution.png")

def main():
    # --- Load data ---
    df = pd.read_csv(INPUT_FILE)

    # --- Summary counts ---
    summary = df["compliance_label"].value_counts().reindex(
        ["Likely compliant", "Partially compliant", "Likely non-compliant"]
    ).fillna(0)

    print("📊 Compliance summary:")
    print(summary)

    # --- Bar chart ---
    plt.figure(figsize=(7, 5))
    bars = plt.bar(summary.index, summary.values, color=["#4CAF50", "#FFC107", "#F44336"])
    plt.title("GDPR Compliance Distribution")
    plt.xlabel("Compliance Category")
    plt.ylabel("Number of Paragraphs")

    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            height + 0.5, 
            f"{int(height)}", 
            ha='center', va='bottom', fontsize=11
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    plt.show()

    print(f"\n✅ Chart saved to: {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
