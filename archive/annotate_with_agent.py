import os
import pandas as pd
import datetime
from gdpr_agent import GDPRComplianceAgent

# === CONFIG ===
INPUT_CSV = "data/processed/paragraphs.csv"
OUTPUT_CSV = "data/annotated/compliance_dataset.csv"

def annotate_dataset():
    df = pd.read_csv(INPUT_CSV)
    agent = GDPRComplianceAgent()

    annotated = []

    for i, row in df.iterrows():
        text = row["policy_text"]

        try:
            # Run the GDPR agent
            report = agent.evaluate_policy(text)

            # Aggregate the results across all checks
            if report.empty:
                gdpr_article = ""
                compliance_label = "Unclear"
                explanation = "No relevant content detected."
                confidence = 0.0
            else:
                # Compute overall compliance score (average of check scores)
                overall = agent.overall_score(report)
                confidence = round(overall, 3)

                # Get failing or partial checks to build explanation
                failed = report[report["status"] != "pass"]
                if failed.empty:
                    compliance_label = "Compliant"
                    explanation = "All checks passed."
                    gdpr_article = ", ".join(set(sum([c.split(", ") for c in report["gdpr_articles"].tolist()], [])))
                else:
                    compliance_label = "Non-compliant" if (failed["status"] == "fail").any() else "Partially compliant"
                    first_issue = failed.iloc[0]
                    gdpr_article = first_issue["gdpr_articles"]
                    explanation = f"Missing: {', '.join(first_issue['missing'])}. Recommendation: {first_issue['recommendation']}"

            annotated_row = row.to_dict()
            annotated_row.update({
                "gdpr_article": gdpr_article,
                "gdpr_principle": "",
                "compliance_label": compliance_label,
                "explanation": explanation,
                "annotator": "gdpr_agent_v1",
                "confidence": confidence,
                "date_annotated": datetime.date.today().isoformat(),
            })

        except Exception as e:
            print(f"⚠️ Error processing row {i}: {e}")
            annotated_row = row.to_dict()
            annotated_row.update({
                "gdpr_article": "",
                "gdpr_principle": "",
                "compliance_label": "error",
                "explanation": str(e),
                "annotator": "gdpr_agent_v1",
                "confidence": "",
                "date_annotated": datetime.date.today().isoformat(),
            })

        annotated.append(annotated_row)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame(annotated).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"✅ Saved annotated dataset to {OUTPUT_CSV}")

if __name__ == "__main__":
    annotate_dataset()
