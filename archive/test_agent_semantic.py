from gdpr_agent import GDPRComplianceAgent
import pandas as pd

# === Load data ===
df = pd.read_csv("/Users/davidj.silva/Desktop/Data Science/Thesis/Project/data/annotated/paragraphs_with_articles.csv")
gdpr_df = pd.read_csv("/Users/davidj.silva/Desktop/Data Science/Thesis/Project/data/reference_law_articles.csv")

# === Initialize the agent ===
agent = GDPRComplianceAgent(gdpr_articles_df=gdpr_df)

# === Pick one paragraph to test ===
text = df.iloc[0]["policy_text"]
print("Policy paragraph:")
print(text[:500], "...\n")

# === Run the agent’s heuristic evaluation ===
report = agent.evaluate_policy(text)
print("Heuristic evaluation:")
print(report[["check", "status", "gdpr_articles"]])

# === Show the semantic match from your linker ===
print("\nMost similar GDPR Article (semantic):")
print(f"Article {df.iloc[0]['best_article_number']} — {df.iloc[0]['best_article_title']}")
