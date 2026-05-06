import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_FILE = PROJECT_ROOT / "data/processed/paragraphs.csv"
GDPR_FILE = PROJECT_ROOT / "data/reference_law_articles.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/annotated/paragraphs_with_articles.csv"

TOP_K = 3  # number of articles to keep per paragraph

def main():
    # --- Load datasets ---
    df_policy = pd.read_csv(POLICY_FILE)
    df_gdpr = pd.read_csv(GDPR_FILE)
    df_gdpr_articles = df_gdpr[df_gdpr["section_type"] == "Article"].copy()

    # --- Build TF-IDF vector spaces ---
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1,2))
    corpus = df_gdpr_articles["text"].fillna("").tolist() + df_policy["policy_text"].fillna("").tolist()
    vectorizer.fit(corpus)

    X_gdpr = vectorizer.transform(df_gdpr_articles["text"].fillna("").tolist())
    X_policy = vectorizer.transform(df_policy["policy_text"].fillna("").tolist())

    # --- Compute similarities ---
    sims = cosine_similarity(X_policy, X_gdpr)

    # --- For each paragraph, get top-k most similar GDPR articles ---
    top_matches = []
    for i, row in df_policy.iterrows():
        sim_row = sims[i]
        top_idx = sim_row.argsort()[::-1][:TOP_K]
        matches = [
            {
                "article_number": df_gdpr_articles.iloc[j]["number"],
                "article_title": df_gdpr_articles.iloc[j]["title"],
                "score": float(sim_row[j]),
            }
            for j in top_idx
        ]
        top_matches.append(matches)

    # --- Attach to dataframe ---
    df_policy["related_articles"] = [
        "; ".join([f"Art. {m['article_number']} ({m['score']:.2f})" for m in ms])
        for ms in top_matches
    ]
    df_policy["best_article_number"] = [ms[0]["article_number"] for ms in top_matches]
    df_policy["best_article_title"] = [ms[0]["article_title"] for ms in top_matches]
    df_policy["best_similarity"] = [ms[0]["score"] for ms in top_matches]

    df_policy.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df_policy)} paragraphs with top-{TOP_K} article matches → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
