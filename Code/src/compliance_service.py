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

MAX_POLICY_CHARS = 8000
MAX_FINDINGS = 6
MAX_EVIDENCE_ITEMS = 8


GDPR_REQUIREMENTS = {
    "controller_identity": {
        "policy_section": "Controller identity and contact details",
        "gdpr_article": "Articles 13(1)(a), 14(1)(a)",
        "description": "The policy should identify the data controller and provide accessible contact details.",
        "patterns": [
            r"\b(controller|data controller|we are responsible|company responsible)\b",
            r"\b(contact us|privacy team|privacy office|privacy manager|dpo|data protection officer)\b",
            r"\b(email|address|postal address|registered office|contact details)\b",
        ],
    },
    "purposes": {
        "policy_section": "Purposes of processing",
        "gdpr_article": "Articles 5(1)(b), 13(1)(c), 14(1)(c)",
        "description": "The policy should explain why personal data is processed.",
        "patterns": [
            r"\b(purpose|purposes|we use|we process|used to|in order to|so that we can)\b",
            r"\b(provide|deliver|operate|improve|personalise|communicate|support|security|marketing)\b",
        ],
    },
    "lawful_basis": {
        "policy_section": "Lawful basis for processing",
        "gdpr_article": "Article 6",
        "description": "The policy should identify or explain the lawful basis for processing personal data.",
        "patterns": [
            r"\b(lawful basis|legal basis|legal ground|basis for processing)\b",
            r"\b(consent|contract|legitimate interest|legal obligation|vital interests|public task)\b",
            r"\b(necessary to perform|necessary for our legitimate interests|required by law|with your consent)\b",
        ],
    },
    "data_categories": {
        "policy_section": "Categories of personal data",
        "gdpr_article": "Articles 13(1)(c), 14(1)(d)",
        "description": "The policy should describe the categories of personal data collected or processed.",
        "patterns": [
            r"\b(personal data|personal information|information we collect|data we collect)\b",
            r"\b(name|email|address|phone|payment|account|device|ip address|location|usage data|cookies)\b",
            r"\b(categories of data|types of data|information such as)\b",
        ],
    },
    "data_subject_rights": {
        "policy_section": "Data subject rights",
        "gdpr_article": "Articles 12, 15-22",
        "description": "The policy should explain the rights available to data subjects.",
        "patterns": [
            r"\b(your rights|data subject rights|privacy rights|rights under)\b",
            r"\b(access|rectification|erasure|delete|restriction|portability|object|withdraw consent)\b",
            r"\b(request|exercise your rights|contact us to exercise)\b",
        ],
    },
    "retention": {
        "policy_section": "Data retention",
        "gdpr_article": "Article 5(1)(e), Article 13(2)(a), Article 14(2)(a)",
        "description": "The policy should explain how long data is retained or the criteria used to determine retention.",
        "patterns": [
            r"\b(retain|retention|keep|store|storage period|how long)\b",
            r"\b(as long as necessary|for as long as|retention period|deleted after|criteria used)\b",
            r"\b(legal obligation|business purpose|account closure|until no longer needed)\b",
        ],
    },
    "recipients": {
        "policy_section": "Recipients and third-party sharing",
        "gdpr_article": "Articles 13(1)(e), 14(1)(e), 28",
        "description": "The policy should explain whether data is shared with recipients, processors, or third parties.",
        "patterns": [
            r"\b(share|disclose|transfer|provide|recipient|third party|third parties)\b",
            r"\b(service provider|processor|vendor|partner|affiliate|supplier|contractor)\b",
            r"\b(hosting|analytics|payment|support|marketing|security provider)\b",
        ],
    },
    "international_transfers": {
        "policy_section": "International data transfers",
        "gdpr_article": "Articles 44-49",
        "description": "The policy should explain international transfers and safeguards where applicable.",
        "patterns": [
            r"\b(international transfer|transfer outside|outside the eea|outside the eu|outside the uk)\b",
            r"\b(standard contractual clauses|scc|adequacy|safeguards|data privacy framework)\b",
            r"\b(other countries|global|worldwide|internationally)\b",
        ],
    },
    "security": {
        "policy_section": "Security of processing",
        "gdpr_article": "Article 32",
        "description": "The policy should describe technical or organisational security measures.",
        "patterns": [
            r"\b(security|secure|protect|safeguard|confidentiality)\b",
            r"\b(encryption|access controls|authentication|monitoring|technical and organisational measures)\b",
            r"\b(unauthorised access|loss|misuse|breach|incident)\b",
        ],
    },
    "complaints": {
        "policy_section": "Right to lodge a complaint",
        "gdpr_article": "Article 13(2)(d), Article 14(2)(e), Article 77",
        "description": "The policy should mention the right to lodge a complaint with a supervisory authority.",
        "patterns": [
            r"\b(complaint|lodge a complaint|supervisory authority|data protection authority)\b",
            r"\b(ico|cnpd|dpc|edpb|regulator|authority)\b",
        ],
    },
    "automated_decision_making": {
        "policy_section": "Automated decision-making and profiling",
        "gdpr_article": "Article 13(2)(f), Article 14(2)(g), Article 22",
        "description": "The policy should state whether automated decision-making or profiling is used.",
        "patterns": [
            r"\b(automated decision|automated decision-making|solely automated|profiling)\b",
            r"\b(algorithm|automated processing|personalised recommendations|targeted advertising)\b",
            r"\b(legal effects|similarly significant effects|logic involved)\b",
        ],
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


def fallback_parse_response(raw_clean: str) -> dict:
    return {
        "overall_status": "Non-Compliant",
        "compliance_score": 0,
        "summary": "The hybrid assessment could not be parsed into structured JSON.",
        "findings": [
            {
                "policy_section": "General GDPR compliance",
                "gdpr_article": "Articles 12-14",
                "why": "The model response could not be parsed reliably.",
                "fix": "Review the policy manually against GDPR transparency and information requirements.",
            }
        ],
        "raw_response": raw_clean,
    }


def safe_json_loads(raw: str) -> dict:
    raw_clean = str(raw or "").strip()
    raw_clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw_clean)
    raw_clean = re.sub(r"^```(?:json)?", "", raw_clean, flags=re.IGNORECASE).strip()
    raw_clean = re.sub(r"```$", "", raw_clean).strip()

    try:
        parsed = json.loads(raw_clean)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = raw_clean.find("{")
    if start == -1:
        return fallback_parse_response(clean_text(raw_clean))

    brace_count = 0
    end = -1
    in_string = False
    escape = False

    for i in range(start, len(raw_clean)):
        char = raw_clean[i]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break

    if end != -1:
        candidate = raw_clean[start:end]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return fallback_parse_response(clean_text(raw_clean))


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


def evaluate_requirement(policy_text: str, chunks: list[str], requirement: dict) -> dict:
    pattern_results = []
    matched_chunks = []

    for pattern in requirement["patterns"]:
        matches = re.findall(pattern, policy_text, flags=re.IGNORECASE)
        pattern_results.append({
            "pattern": pattern,
            "matched": len(matches) > 0,
            "match_count": len(matches),
        })

    for i, chunk in enumerate(chunks):
        for pattern in requirement["patterns"]:
            if re.search(pattern, chunk, flags=re.IGNORECASE):
                matched_chunks.append({
                    "chunk_id": i + 1,
                    "text_preview": chunk[:250],
                })
                break

    matched_patterns = sum(1 for p in pattern_results if p["matched"])
    total_patterns = len(pattern_results)

    score = matched_patterns / total_patterns if total_patterns else 0.0

    if matched_chunks and score < 0.5:
        score = 0.5

    if score >= 0.75:
        coverage_status = "strong"
    elif score >= 0.35:
        coverage_status = "partial"
    else:
        coverage_status = "weak_or_missing"

    return {
        "policy_section": requirement["policy_section"],
        "gdpr_article": requirement["gdpr_article"],
        "description": requirement["description"],
        "heuristic_coverage_score": round(score, 3),
        "coverage_status": coverage_status,
        "matched_patterns": [
            p["pattern"] for p in pattern_results if p["matched"]
        ],
        "matched_chunk_count": len(matched_chunks),
        "matched_chunk_examples": matched_chunks[:3],
    }


def build_requirement_evidence(policy_text: str, chunks: list[str]) -> list[dict]:
    return [
        evaluate_requirement(policy_text, chunks, requirement)
        for requirement in GDPR_REQUIREMENTS.values()
    ]


def build_llm_requirement_review_prompt(policy_text: str, requirement_evidence: list[dict]) -> str:
    policy_text = policy_text[:MAX_POLICY_CHARS]

    requirements = [
        {
            "policy_section": item["policy_section"],
            "gdpr_article": item["gdpr_article"],
            "description": item["description"],
            "heuristic_coverage_score": item["heuristic_coverage_score"],
            "matched_chunk_examples": item.get("matched_chunk_examples", []),
        }
        for item in requirement_evidence
    ]

    requirements_json = json.dumps(requirements, ensure_ascii=False, indent=2)

    return f"""
You are reviewing GDPR privacy policy coverage.

For each GDPR requirement, decide whether the submitted privacy policy addresses it.

Return ONLY valid JSON. No markdown. No explanation outside JSON.

Use this exact structure:

{{
  "requirements": [
    {{
      "policy_section": "Controller identity and contact details",
      "status": "PRESENT",
      "gdpr_article": "Articles 13(1)(a), 14(1)(a)",
      "evidence": "Short quote or summary of the relevant evidence from the policy.",
      "reason": "Brief explanation of why the requirement is present, weak, or missing."
    }}
  ]
}}

Status must be exactly one of:
- PRESENT
- WEAK
- MISSING

Rules:
- Do not mark a requirement as missing only because exact GDPR wording is absent.
- If the policy addresses the idea using different wording, mark it PRESENT or WEAK.
- Use PRESENT only when the requirement is clearly addressed.
- Use WEAK when the requirement is mentioned but lacks detail, clarity, or specificity.
- Use MISSING when the requirement is not addressed.
- Review the actual policy text, not only the heuristic evidence.

Heuristic evidence:
{requirements_json}

Privacy policy text:
\"\"\"
{policy_text}
\"\"\"
""".strip()


def normalise_llm_requirement_reviews(parsed: dict, requirement_evidence: list[dict]) -> list[dict]:
    reviews = parsed.get("requirements", [])

    if not isinstance(reviews, list):
        reviews = []

    review_map = {}

    for item in reviews:
        if not isinstance(item, dict):
            continue

        section = clean_text(item.get("policy_section", ""))
        status = clean_text(item.get("status", "")).upper()

        if status not in ["PRESENT", "WEAK", "MISSING"]:
            status = "WEAK"

        review_map[section] = {
            "llm_status": status,
            "llm_evidence": clean_text(item.get("evidence", "")),
            "llm_reason": clean_text(item.get("reason", "")),
        }

    final = []

    for item in requirement_evidence:
        section = item["policy_section"]
        llm_review = review_map.get(section)

        if llm_review:
            merged = {**item, **llm_review}
        else:
            heuristic_score = item.get("heuristic_coverage_score", 0)

            if heuristic_score >= 0.75:
                fallback_status = "PRESENT"
            elif heuristic_score >= 0.35:
                fallback_status = "WEAK"
            else:
                fallback_status = "MISSING"

            merged = {
                **item,
                "llm_status": fallback_status,
                "llm_evidence": "",
                "llm_reason": "Fallback status derived from heuristic coverage evidence.",
            }

        final.append(merged)

    return final


def build_semantic_evidence(semantic_rows: list[dict]) -> dict:
    top_semantic_evidence = sorted(
        semantic_rows,
        key=lambda x: x["semantic_score"],
        reverse=True,
    )[:MAX_EVIDENCE_ITEMS]

    article_counts = {}

    for row in semantic_rows:
        article_number = row["best_article_number"]
        article_title = row["best_article_title"]

        key = f"Article {article_number}"

        if key not in article_counts:
            article_counts[key] = {
                "gdpr_article": key,
                "article_title": article_title,
                "matched_chunks": 0,
                "max_similarity": 0.0,
            }

        article_counts[key]["matched_chunks"] += 1
        article_counts[key]["max_similarity"] = max(
            article_counts[key]["max_similarity"],
            row["semantic_score"],
        )

    article_summary = sorted(
        article_counts.values(),
        key=lambda x: (x["matched_chunks"], x["max_similarity"]),
        reverse=True,
    )[:MAX_EVIDENCE_ITEMS]

    return {
        "top_semantic_matches": [
            {
                "chunk_id": row["chunk_id"],
                "gdpr_article": f"Article {row['best_article_number']}",
                "article_title": row["best_article_title"],
                "semantic_score": row["semantic_score"],
                "text_preview": row["text"][:250],
            }
            for row in top_semantic_evidence
        ],
        "article_summary": article_summary,
    }


def build_hybrid_evidence(policy_text: str, gdpr_df: pd.DataFrame) -> dict:
    chunks = split_into_chunks(policy_text)
    semantic_rows = semantic_match_chunks(chunks, gdpr_df)

    requirement_evidence = build_requirement_evidence(policy_text, chunks)

    review_prompt = build_llm_requirement_review_prompt(policy_text, requirement_evidence)
    raw_requirement_review = call_llm(review_prompt)

    print("\n\n===== RAW REQUIREMENT REVIEW RESPONSE =====")
    print(raw_requirement_review)
    print("===== END RAW REQUIREMENT REVIEW RESPONSE =====\n\n")

    parsed_requirement_review = safe_json_loads(raw_requirement_review)

    requirement_evidence = normalise_llm_requirement_reviews(
        parsed_requirement_review,
        requirement_evidence,
    )

    semantic_evidence = build_semantic_evidence(semantic_rows)

    try:
        agent = GDPRComplianceAgent(gdpr_articles_df=gdpr_df)
        heuristic_reports = []

        for chunk in chunks:
            report = agent.evaluate_policy(chunk)
            heuristic_score = (
                float(agent.overall_score(report))
                if report is not None and not report.empty
                else 0.0
            )
            heuristic_reports.append(heuristic_score)

        legacy_heuristic_average = (
            sum(heuristic_reports) / len(heuristic_reports)
            if heuristic_reports
            else 0.0
        )
    except Exception:
        legacy_heuristic_average = 0.0

    status_to_score = {
        "PRESENT": 1.0,
        "WEAK": 0.5,
        "MISSING": 0.0,
    }

    requirement_scores = [
        status_to_score.get(item.get("llm_status", "MISSING"), 0.0)
        for item in requirement_evidence
    ]

    requirement_coverage_score = (
        sum(requirement_scores) / len(requirement_scores)
        if requirement_scores
        else 0.0
    )

    strong_areas = [
        item["policy_section"]
        for item in requirement_evidence
        if item.get("llm_status") == "PRESENT"
    ]

    partial_areas = [
        item["policy_section"]
        for item in requirement_evidence
        if item.get("llm_status") == "WEAK"
    ]

    weak_areas = [
        item["policy_section"]
        for item in requirement_evidence
        if item.get("llm_status") == "MISSING"
    ]

    return {
        "chunk_count": len(chunks),
        "requirement_coverage_score": round(requirement_coverage_score, 3),
        "legacy_heuristic_average": round(legacy_heuristic_average, 3),
        "strong_areas": strong_areas,
        "partial_areas": partial_areas,
        "weak_or_missing_areas": weak_areas,
        "requirement_evidence": requirement_evidence,
        "semantic_evidence": semantic_evidence,
        "important_note": (
            "This evidence is advisory and is used for traceability. "
            "Requirement coverage is reviewed by the LLM using both the policy text and heuristic evidence. "
            "The final compliance judgement is made separately by the LLM using the full hybrid context."
        ),
    }


def build_hybrid_prompt(policy_text: str, evidence: dict) -> str:
    was_truncated = len(policy_text) > MAX_POLICY_CHARS
    policy_text = policy_text[:MAX_POLICY_CHARS]

    truncation_note = (
        "The policy text was truncated because of deployment limits. Do not mark it as non-compliant only because it ends abruptly. Assess the visible policy content and identify substantive GDPR gaps."
        if was_truncated
        else "The full submitted policy text is included below."
    )

    compact_evidence = {
        "requirement_coverage_score": evidence["requirement_coverage_score"],
        "strong_areas": evidence["strong_areas"],
        "partial_areas": evidence["partial_areas"],
        "weak_or_missing_areas": evidence["weak_or_missing_areas"],
        "requirement_evidence": evidence["requirement_evidence"],
        "semantic_article_summary": evidence["semantic_evidence"]["article_summary"],
        "important_note": evidence["important_note"],
    }

    evidence_json = json.dumps(compact_evidence, ensure_ascii=False, indent=2)

    return f"""
You are a GDPR compliance auditor.

Assess the privacy policy as a whole document.

This is a hybrid system, but the LLM is the final compliance judge.

The hybrid evidence is provided to support traceability:
- requirement coverage evidence checks whether expected GDPR disclosure elements are present, weak, or missing
- semantic evidence links policy content to likely relevant GDPR articles
- the evidence should guide your review but should not mechanically determine the final score
- base the final verdict primarily on the actual privacy policy text

Your job:
1. Read the privacy policy.
2. Use the hybrid evidence as supporting context.
3. Produce a final GDPR compliance assessment.
4. Return concise findings with relevant GDPR articles, why, and fix.

Return ONLY valid JSON. No markdown. No text outside JSON.

Use this exact JSON structure:

{{
  "overall_status": "Compliant",
  "compliance_score": 85,
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
- overall_status must be exactly one of: "Compliant", "Partially Compliant", "Non-Compliant".
- compliance_score must be an integer from 0 to 100.
- Return between 1 and {MAX_FINDINGS} findings.
- Do not mention chunk numbers.
- Do not mention paragraph numbers.
- Do not list every GDPR article.
- Focus only on the most important GDPR compliance gaps.
- Do not penalise the policy simply because it does not use exact GDPR wording.
- Do not mark the policy as non-compliant solely because a pattern was not matched.
- Do not mark the policy as non-compliant solely because the submitted text may be truncated.
- Each finding must include policy_section, gdpr_article, why, and fix.

Truncation note:
{truncation_note}

Hybrid evidence:
{evidence_json}

Privacy policy text:
\"\"\"
{policy_text}
\"\"\"
""".strip()


def normalise_status(value: str, score: int) -> str:
    value = clean_text(value)

    if value in ["Compliant", "Partially Compliant", "Non-Compliant"]:
        return value

    lower = value.lower()

    if "non" in lower:
        return "Non-Compliant"
    if "partial" in lower:
        return "Partially Compliant"
    if "compliant" in lower:
        return "Compliant"

    return status_from_score(score)


def normalise_findings(findings) -> list[dict]:
    if not isinstance(findings, list):
        findings = []

    cleaned = []

    for item in findings[:MAX_FINDINGS]:
        if not isinstance(item, dict):
            continue

        policy_section = clean_text(item.get("policy_section", "General GDPR compliance"))
        gdpr_article = clean_text(item.get("gdpr_article", "Relevant GDPR provisions"))
        why = clean_text(item.get("why", "The policy may require further review."))
        fix = clean_text(item.get("fix", "Clarify this section in the privacy policy."))

        cleaned.append({
            "policy_section": policy_section or "General GDPR compliance",
            "gdpr_article": gdpr_article or "Relevant GDPR provisions",
            "why": why or "The policy may require further review.",
            "fix": fix or "Clarify this section in the privacy policy.",
        })

    if not cleaned:
        cleaned.append({
            "policy_section": "General GDPR compliance",
            "gdpr_article": "Articles 12-14",
            "why": "The model did not return specific structured findings.",
            "fix": "Review the policy manually and ensure key GDPR requirements are addressed.",
        })

    return cleaned


def build_issues(findings: list[dict], score: int) -> list[dict]:
    severity = "High" if score < 45 else "Medium"

    return [
        {
            "title": finding["policy_section"],
            "severity": severity,
            "description": (
                f"GDPR article to address: {finding['gdpr_article']}\n\n"
                f"Why: {finding['why']}\n\n"
                f"Fix: {finding['fix']}"
            ),
            "policy_section": finding["policy_section"],
            "gdpr_article": finding["gdpr_article"],
            "why": finding["why"],
            "fix": finding["fix"],
        }
        for finding in findings
    ]


def build_sections_from_evidence(evidence: dict) -> list[dict]:
    sections = []

    status_map = {
        "PRESENT": "present",
        "WEAK": "weak",
        "MISSING": "missing",
    }

    for item in evidence.get("requirement_evidence", []):
        llm_status = item.get("llm_status", "MISSING")
        score = item.get("heuristic_coverage_score", 0)

        note_parts = [
            item.get("description", ""),
            f"GDPR reference: {item.get('gdpr_article', '')}.",
            f"Coverage status: {llm_status}.",
        ]

        if item.get("llm_evidence"):
            note_parts.append(f"Evidence: {item['llm_evidence']}.")

        if item.get("llm_reason"):
            note_parts.append(f"Reason: {item['llm_reason']}.")

        note_parts.append(f"Heuristic evidence score: {score}.")

        sections.append({
            "name": item["policy_section"],
            "status": status_map.get(llm_status, "missing"),
            "note": " ".join(note_parts),
        })

    return sections


def build_paragraph_results(findings: list[dict], status: str, score: int) -> list[dict]:
    return [
        {
            "paragraph_id": i + 1,
            "policy_text": finding["policy_section"],
            "best_article_number": finding["gdpr_article"],
            "best_article_title": finding["policy_section"],
            "section_name": finding["policy_section"],
            "heuristic_score": 0.0,
            "semantic_score": 0.0,
            "llm_verdict": status,
            "llm_score": round(score / 100, 3),
            "llm_assessment": (
                f"GDPR article to address: {finding['gdpr_article']}\n\n"
                f"Why: {finding['why']}\n\n"
                f"Fix: {finding['fix']}"
            ),
            "reviewed_by_llm": True,
            "combined_score": round(score / 100, 3),
            "combined_label": status,
        }
        for i, finding in enumerate(findings)
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

    print("\n\n===== RAW HYBRID MODEL RESPONSE =====")
    print(raw_reply)
    print("===== END RAW HYBRID MODEL RESPONSE =====\n\n")

    parsed = safe_json_loads(raw_reply)

    score = parsed.get("compliance_score", 0)

    try:
        score = int(float(score))
    except Exception:
        score = 0

    score = max(0, min(100, score))

    overall_status = status_from_score(score)
    findings = normalise_findings(parsed.get("findings", []))
    summary = clean_text(parsed.get("summary", "")) or "Hybrid GDPR assessment completed."

    issues = build_issues(findings, score)
    sections = build_sections_from_evidence(evidence)
    paragraph_results = build_paragraph_results(findings, overall_status, score)

    recommendations = [
        {
            "number": i + 1,
            "text": f"{finding['policy_section']}: {finding['fix']}",
        }
        for i, finding in enumerate(findings)
    ]

    return {
        "combined_score": round(score / 100, 3),
        "combined_score_percent": score,
        "combined_label": overall_status,
        "overall_status": overall_status,
        "summary": summary,
        "word_count": word_count,
        "paragraph_count": evidence["chunk_count"],
        "llm_reviewed_paragraph_count": 2,
        "api_mode": "hybrid_llm_requirement_semantic_traceability",
        "sections": sections,
        "paragraph_results": paragraph_results,
        "issues": issues,
        "recommendations": recommendations,
        "hybrid_evidence": evidence,
        "raw_llm_response": clean_text(raw_reply),
    }