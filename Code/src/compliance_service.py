from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.gdpr_agent import GDPRComplianceAgent
from src.combine_scores_LLM import (
    llm_call_ollama,
    build_prompt_plain,
    extract_llm_verdict,
    verdict_to_score,
    label_from_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GDPR_FILE = PROJECT_ROOT / "data" / "processed" / "reference_law_articles.csv"

HEURISTIC_WEIGHT = 0.25
SEMANTIC_WEIGHT = 0.25
LLM_WEIGHT = 0.50


def split_into_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]

    if not paragraphs:
        return [text.strip()]

    return paragraphs


def lookup_article_text(gdpr_df: pd.DataFrame, number):
    mask = (
        (gdpr_df["section_type"] == "Article")
        & (gdpr_df["number"].astype(str) == str(number))
    )
    row = gdpr_df.loc[mask].head(1)

    if row.empty:
        return "", ""

    return str(row.iloc[0].get("title", "")), str(row.iloc[0].get("text", ""))


def semantic_match_paragraphs(paragraphs: list[str], gdpr_df: pd.DataFrame):
    articles = gdpr_df[gdpr_df["section_type"] == "Article"].copy()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2)
    )

    corpus = articles["text"].fillna("").tolist() + paragraphs
    vectorizer.fit(corpus)

    x_articles = vectorizer.transform(articles["text"].fillna("").tolist())
    x_policy = vectorizer.transform(paragraphs)

    sims = cosine_similarity(x_policy, x_articles)

    matches = []

    for i, sim_row in enumerate(sims):
        best_idx = sim_row.argsort()[::-1][0]
        article = articles.iloc[best_idx]

        matches.append({
            "paragraph": paragraphs[i],
            "best_article_number": str(article["number"]),
            "best_article_title": str(article["title"]),
            "semantic_score": float(sim_row[best_idx]),
        })

    return matches


def assess_policy_text(policy_text: str) -> dict:
    if not policy_text or not policy_text.strip():
        raise ValueError("No policy text provided.")

    if not GDPR_FILE.exists():
        raise FileNotFoundError(f"Missing GDPR reference file: {GDPR_FILE}")

    gdpr_df = pd.read_csv(GDPR_FILE)
    agent = GDPRComplianceAgent(gdpr_articles_df=gdpr_df)

    paragraphs = split_into_paragraphs(policy_text)
    semantic_rows = semantic_match_paragraphs(paragraphs, gdpr_df)

    paragraph_results = []

    for idx, row in enumerate(semantic_rows, start=1):
        paragraph = row["paragraph"]

        heuristic_report = agent.evaluate_policy(paragraph)
        heuristic_score = (
            float(agent.overall_score(heuristic_report))
            if heuristic_report is not None and not heuristic_report.empty
            else 0.0
        )

        semantic_score = float(row["semantic_score"])

        article_title, article_text = lookup_article_text(
            gdpr_df,
            row["best_article_number"]
        )

        prompt = build_prompt_plain(
            paragraph[:800],
            row["best_article_number"],
            article_title,
            article_text
        )

        llm_reply = llm_call_ollama(prompt)
        llm_verdict = extract_llm_verdict(llm_reply)
        llm_score = verdict_to_score(llm_verdict)

        combined_score = (
            HEURISTIC_WEIGHT * heuristic_score
            + SEMANTIC_WEIGHT * semantic_score
            + LLM_WEIGHT * llm_score
        )

        paragraph_results.append({
            "paragraph_id": idx,
            "policy_text": paragraph[:500],
            "best_article_number": row["best_article_number"],
            "best_article_title": row["best_article_title"],
            "heuristic_score": round(heuristic_score, 3),
            "semantic_score": round(semantic_score, 3),
            "llm_verdict": llm_verdict,
            "llm_score": round(llm_score, 3),
            "llm_assessment": llm_reply,
            "combined_score": round(combined_score, 3),
            "combined_label": label_from_score(combined_score),
        })

    overall_score = sum(r["combined_score"] for r in paragraph_results) / len(paragraph_results)
    overall_label = label_from_score(overall_score)

    issues = [
        {
            "title": f"Potential issue in paragraph {r['paragraph_id']}",
            "severity": "High" if r["combined_score"] < 0.3 else "Medium",
            "description": r["llm_assessment"],
        }
        for r in paragraph_results
        if r["combined_score"] < 0.55
    ]

    recommendations = [
        {
            "number": i + 1,
            "text": f"Review paragraph {r['paragraph_id']} against GDPR Article {r['best_article_number']}."
        }
        for i, r in enumerate(paragraph_results)
        if r["combined_score"] < 0.55
    ]

    return {
        "combined_score": round(overall_score, 3),
        "combined_score_percent": round(overall_score * 100),
        "combined_label": overall_label,
        "overall_status": overall_label.replace("Likely ", ""),
        "summary": build_summary(overall_label),
        "word_count": len(policy_text.split()),
        "paragraph_count": len(paragraphs),
        "paragraph_results": paragraph_results,
        "issues": issues,
        "recommendations": recommendations,
    }


def build_summary(label: str) -> str:
    if label == "Likely compliant":
        return "The submitted policy shows strong alignment with the GDPR checks performed by the system."
    if label == "Partially compliant":
        return "The submitted policy shows partial GDPR alignment but still contains relevant compliance gaps."
    return "The submitted policy shows significant gaps against GDPR requirements."