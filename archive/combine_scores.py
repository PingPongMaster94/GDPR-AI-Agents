import pandas as pd
from gdpr_agent import GDPRComplianceAgent

# === CONFIG ===
POLICY_FILE = "/Users/davidj.silva/Desktop/Data Science/Thesis/Project/data/annotated/paragraphs_with_articles.csv"
GDPR_FILE = "/Users/davidj.silva/Desktop/Data Science/Thesis/Project/data/reference_law_articles.csv"
OUTPUT_FILE = "/Users/davidj.silva/Desktop/Data Science/Thesis/Project/data/annotated/combined_compliance_scores.csv"

# === PARAMETERS ===
SEMANTIC_WEIGHT = 0.4   # weight for semantic similarity (0–1)
HEURISTIC_WEIGHT = 0.6  # weight for heuristic score (0–1)
THRESHOLDS = {
    "compliant": 0.55,
    "partial": 0.3
}

def label_from_score(score):
    if score >= THRESHOLDS["compliant"]:
        return "Likely compliant"
    elif score >= THRESHOLDS["partial"]:
        return "Partially compliant"
    else:
        return "Likely non-compliant"

def main():
    # --- Load data ---
    df = pd.read_csv(POLICY_FILE)
    gdpr_df = pd.read_csv(GDPR_FILE)
    agent = GDPRComplianceAgent(gdpr_articles_df=gdpr_df)

    combined = []

    # --- Iterate through policy paragraphs ---
    for i, row in df.iterrows():
        text = str(row["policy_text"])

        # Heuristic evaluation
        report = agent.evaluate_policy(text)
        if report.empty:
            heuristic_score = 0.0
        else:
            heuristic_score = agent.overall_score(report)

        # Semantic similarity (from previous step)
        semantic_score = float(row.get("best_similarity", 0.0))

        # Combine both
        combined_score = (HEURISTIC_WEIGHT * heuristic_score) + (SEMANTIC_WEIGHT * semantic_score)
        compliance_label = label_from_score(combined_score)

        # Compose output row
        combined.append({
            "id": row.get("id", i),
            "policy_text": text[:300] + ("..." if len(text) > 300 else ""),
            "best_article_number": row.get("best_article_number", ""),
            "best_article_title": row.get("best_article_title", ""),
            "heuristic_score": round(heuristic_score, 3),
            "semantic_score": round(semantic_score, 3),
            "combined_score": round(combined_score, 3),
            "compliance_label": compliance_label
        })

    out = pd.DataFrame(combined)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"✅ Combined scores saved to {OUTPUT_FILE}")
    print(out.head(10))

if __name__ == "__main__":
    main()
