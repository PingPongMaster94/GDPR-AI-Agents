from pathlib import Path
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.gdpr_agent import GDPRComplianceAgent
from src.llm_client import call_llm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GDPR_FILE = PROJECT_ROOT / "data" / "processed" / "reference_law_articles.csv"

MAX_POLICY_CHARS = 14000
MAX_FINDINGS = 6
MAX_EVIDENCE_ITEMS = 10


COMPLIANCE_AREAS = {
    "lawful_basis": {
        "policy_section": "Lawful basis for processing",
        "gdpr_article": "Article 6",
        "keywords": ["lawful basis", "legal basis", "legitimate interest", "contract", "legal obligation", "consent"],
    },
    "consent": {
        "policy_section": "Consent and withdrawal",
        "gdpr_article": "Article 7",
        "keywords": ["consent", "withdraw", "opt out", "permission"],
    },
    "transparency": {
        "policy_section": "Transparency and privacy information",
        "gdpr_article": "Articles 12–14",
        "keywords": ["privacy notice", "information we collect", "personal information", "privacy policy", "data controller"],
    },
    "rights": {
        "policy_section": "Data subject rights",
        "gdpr_article": "Articles 15–22",
        "keywords": ["access", "rectification", "erasure", "delete", "portability", "object", "restriction", "rights"],
    },
    "retention": {
        "policy_section": "Data retention",
        "gdpr_article": "Article 5(1)(e) and Article 13(2)(a)",
        "keywords": ["retain", "retention", "storage period", "how long", "delete"],
    },
    "third_parties": {
        "policy_section": "Third-party sharing and processors",
        "gdpr_article": "Article 28 and Article 13(1)(e)",
        "keywords": ["third party", "processor", "service provider", "vendor", "partner", "share"],
    },
    "international_transfers": {
        "policy_section": "International data transfers",
        "gdpr_article": "Articles 44–49",
        "keywords": ["international transfer", "outside the eea", "outside eu", "standard contractual clauses", "scc", "adequacy"],
    },
    "security": {
        "policy_section": "Security of processing",
        "gdpr_article": "Article 32",
        "keywords": ["security", "encryption", "protect", "safeguards", "confidentiality", "access controls"],
    },
    "automated_decision_making": {
        "policy_section": "Automated decision-making and profiling",
        "gdpr_article": "Article 22",
        "keywords": ["automated decision", "profiling", "algorithm", "automated processing"],
    },
}


def clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(text: str) -> list[str]:
    text = text.strip()
    parts = re.split(r"\n\s*\n", text)

    if len(parts) <= 1:
        parts = re.split(
            r"(?=\n?\d+\.\s+[A-Z])|(?=\n?[A-Z][A-Za-z\s]{3,80}\n)",
            text,
        )

    parts = [p.strip() for p in parts if len(p.strip()) > 80]

    chunks = []

    for part in parts:
        if len(part) <= 1200:
            chunks.append(part)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", part)
            buffer = ""

            for sentence in sentences:
                if len(buffer) + len(sentence) <= 1000:
                    buffer = f"{buffer} {sentence}".strip()
                else:
                    if len(buffer) > 80:
                        chunks.append(buffer)
                    buffer = sentence

            if len(buffer) > 80:
                chunks.append(buffer)

    return chunks if chunks else [text]


def status_from_score(score: int) -> str:
    if score >= 75:
        return "Compliant"
    if score >= 45:
        return "Partially Compliant"
    return "Non-Compliant"


def safe_json_loads(raw: str) -> dict:
    raw_clean = clean_text(raw)

    try:
        return json.loads(raw_clean)
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw_clean, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {
        "overall_status": "Non-Compliant",
        "compliance_score": 0,
        "summary": "The hybrid assessment could not be parsed into structured JSON.",
        "findings": [
            {
                "policy_section": "General GDPR compliance",
                "gdpr_article": "Articles 12–14",
                "why": "The model response could not be parsed reliably.",
                "fix": "Review the policy manually against GDPR transparency and information requirements.",
            }
        ],
    }


def semantic_match_chunks(chunks: list[str], gdpr_df: pd.DataFrame) -> list[dict]:
    articles = gdpr_df[gdpr_df["section_type"] == "Article"].copy()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
    )

    corpus = articles["text"].fillna("").tolist() + chunks
    vectorizer.fit(corpus)

    x_articles = vectorizer.transform(articles["text"].fillna("").tolist())
    x_policy = vectorizer.transform(chunks)

    sims = cosine_similarity(x_policy, x_articles)

    matches = []

    for i, sim_row in enumerate(sims):
        best_idx = sim_row.argsort()[::-1][0]
        article = articles.iloc[best_idx]

        title = article.get("title", "")
        if pd.isna(title):
            title = ""

        matches.append({
            "chunk_id": i + 1,
            "text": chunks[i],
            "best_article_number": str(article["number"]),
            "best_article_title": str(title),
            "semantic_score": round(float(sim_row[best_idx]), 3),
        })

    return matches


def build_hybrid_evidence(policy_text: str, gdpr_df: pd.DataFrame) -> dict:
    agent = GDPRComplianceAgent(gdpr_articles_df=gdpr_df)

    chunks = split_into_chunks(policy_text)
    semantic_rows = semantic_match_chunks(chunks, gdpr_df)

    chunk_results = []

    for row in semantic_rows:
        report = agent.evaluate_policy(row["text"])
        heuristic_score = (
            float(agent.overall_score(report))
            if report is not None and not report.empty
            else 0.0
        )

        chunk_results.append({
            "chunk_id": row["chunk_id"],
            "best_article_number": row["best_article_number"],
            "best_article_title": row["best_article_title"],
            "semantic_score": row["semantic_score"],
            "heuristic_score": round(heuristic_score, 3),
            "text_preview": row["text"][:300],
        })

    area_summary = []

    policy_lower = policy_text.lower()

    for key, area in COMPLIANCE_AREAS.items():
        keyword_hits = [
            kw for kw in area["keywords"]
            if kw.lower() in policy_lower
        ]

        related_chunks = [
            c for c in chunk_results
            if any(kw.lower() in c["text_preview"].lower() for kw in area["keywords"])
        ]

        if related_chunks:
            avg_heuristic = sum(c["heuristic_score"] for c in related_chunks) / len(related_chunks)
            avg_semantic = sum(c["semantic_score"] for c in related_chunks) / len(related_chunks)
        else:
            avg_heuristic = 0.0
            avg_semantic = 0.0

        area_summary.append({
            "policy_section": area["policy_section"],
            "gdpr_article": area["gdpr_article"],
            "keyword_hits": keyword_hits,
            "coverage_detected": bool(keyword_hits or related_chunks),
            "average_heuristic_score": round(avg_heuristic, 3),
            "average_semantic_score": round(avg_semantic, 3),
        })

    weakest_evidence = sorted(
        chunk_results,
        key=lambda x: (x["heuristic_score"], x["semantic_score"])
    )[:MAX_EVIDENCE_ITEMS]

    return {
        "chunk_count": len(chunks),
        "area_summary": area_summary,
        "weakest_evidence": weakest_evidence,
    }


def build_hybrid_prompt(policy_text: str, evidence: dict) -> str:
    policy_text = policy_text[:MAX_POLICY_CHARS]

    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)

    return f"""
You are a GDPR compliance auditor.

Assess the whole privacy policy as one document.

You are given:
1. The full privacy policy text.
2. Structured evidence from a hybrid GDPR framework:
   - heuristic coverage checks
   - semantic GDPR article matching
   - weak evidence areas

Use the structured evidence to guide your assessment, but produce a clear final judgement on the full policy.

Return ONLY valid JSON. No markdown. No text outside JSON.

Use this exact JSON structure:

{{
  "overall_status": "Compliant | Partially Compliant | Non-Compliant",
  "compliance_score": 0,
  "summary": "Short overall explanation.",
  "findings": [
    {{
      "policy_section": "Policy section or compliance area that needs improvement",
      "gdpr_article": "Relevant GDPR article or article range",
      "why": "Why this area is incomplete, unclear, or non-compliant",
      "fix": "Concrete action needed to improve compliance"
    }}
  ]
}}

Rules:
- compliance_score must be an integer from 0 to 100.
- Return between 1 and {MAX_FINDINGS} findings.
- Do not mention chunk numbers or paragraph numbers.
- Do not list every GDPR article.
- Focus on the most important GDPR compliance gaps.
- Each finding must include policy_section, gdpr_article, why, and fix.

Hybrid evidence:
{evidence_json}

Privacy policy text:
\"\"\"
{policy_text}
\"\"\"
""".strip()


def normalise_findings(findings) -> list[dict]:
    if not isinstance(findings, list):
        findings = []

    cleaned = []

    for item in findings[:MAX_FINDINGS]:
        if not isinstance(item, dict):
            continue

        cleaned.append({
            "policy_section": clean_text(item.get("policy_section", "General GDPR compliance")),
            "gdpr_article": clean_text(item.get("gdpr_article", "Relevant GDPR provisions")),
            "why": clean_text(item.get("why", "The policy may require further review.")),
            "fix": clean_text(item.get("fix", "Clarify this section in the privacy policy.")),
        })

    if not cleaned:
        cleaned.append({
            "policy_section": "General GDPR compliance",
            "gdpr_article": "Articles 12–14",
            "why": "The model did not return specific structured findings.",
            "fix": "Review the policy manually and ensure key GDPR requirements are addressed.",
        })

    return cleaned


def build_issues(findings: list[dict], score: int) -> list[dict]:
    severity = "High" if score < 45 else "Medium"

    return [
        {
            "title": f["policy_section"],
            "severity": severity,
            "description": (
                f"GDPR article to address: {f['gdpr_article']}\n\n"
                f"Why: {f['why']}\n\n"
                f"Fix: {f['fix']}"
            ),
            "policy_section": f["policy_section"],
            "gdpr_article": f["gdpr_article"],
            "why": f["why"],
            "fix": f["fix"],
        }
        for f in findings
    ]


def build_sections(findings: list[dict], score: int) -> list[dict]:
    status = "missing" if score < 45 else "weak"

    return [
        {
            "name": f["policy_section"],
            "status": status,
            "note": f"GDPR article to address: {f['gdpr_article']}",
        }
        for f in findings
    ]


def build_paragraph_results(findings: list[dict], status: str, score: int) -> list[dict]:
    return [
        {
            "paragraph_id": i + 1,
            "policy_text": f["policy_section"],
            "best_article_number": f["gdpr_article"],
            "best_article_title": f["policy_section"],
            "section_name": f["policy_section"],
            "heuristic_score": 0.0,
            "semantic_score": 0.0,
            "llm_verdict": status,
            "llm_score": round(score / 100, 3),
            "llm_assessment": (
                f"GDPR article to address: {f['gdpr_article']}\n\n"
                f"Why: {f['why']}\n\n"
                f"Fix: {f['fix']}"
            ),
            "reviewed_by_llm": True,
            "combined_score": round(score / 100, 3),
            "combined_label": status,
        }
        for i, f in enumerate(findings)
    ]


def assess_policy_text(policy_text: str) -> dict:
    if not policy_text or not policy_text.strip():
        raise ValueError("No policy text provided.")

    if not GDPR_FILE.exists():
        raise FileNotFoundError(f"Missing GDPR reference file: {GDPR_FILE}")

    policy_text = policy_text.strip()
    word_count = len(policy_text.split())

    gdpr_df = pd.read_csv(GDPR_FILE)

    evidence = build_hybrid_evidence(policy_text, gdpr_df)
    prompt = build_hybrid_prompt(policy_text, evidence)

    raw_reply = call_llm(prompt)
    parsed = safe_json_loads(raw_reply)

    score = parsed.get("compliance_score", 0)

    try:
        score = int(float(score))
    except Exception:
        score = 0

    score = max(0, min(100, score))

    overall_status = clean_text(parsed.get("overall_status", "")) or status_from_score(score)

    if overall_status not in ["Compliant", "Partially Compliant", "Non-Compliant"]:
        overall_status = status_from_score(score)

    findings = normalise_findings(parsed.get("findings", []))
    summary = clean_text(parsed.get("summary", "")) or "Hybrid GDPR assessment completed."

    issues = build_issues(findings, score)
    sections = build_sections(findings, score)
    paragraph_results = build_paragraph_results(findings, overall_status, score)

    recommendations = [
        {
            "number": i + 1,
            "text": f"{f['policy_section']}: {f['fix']}",
        }
        for i, f in enumerate(findings)
    ]

    return {
        "combined_score": round(score / 100, 3),
        "combined_score_percent": score,
        "combined_label": overall_status,
        "overall_status": overall_status,
        "summary": summary,
        "word_count": word_count,
        "paragraph_count": evidence["chunk_count"],
        "llm_reviewed_paragraph_count": 1,
        "api_mode": "hybrid_evidence_guided",
        "sections": sections,
        "paragraph_results": paragraph_results,
        "issues": issues,
        "recommendations": recommendations,
        "hybrid_evidence": evidence,
        "raw_llm_response": clean_text(raw_reply),
    }