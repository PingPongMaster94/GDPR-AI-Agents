from pathlib import Path
import json
import re

from src.llm_client import call_llm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAX_POLICY_CHARS = 18000
MAX_FINDINGS = 6
MIN_POLICY_WORDS = 50


def clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def status_from_score(score: int) -> str:
    if score >= 75:
        return "Compliant"
    if score >= 45:
        return "Partially Compliant"
    return "Non-Compliant"


def invalid_policy_response(policy_text: str) -> dict:
    word_count = len(policy_text.split())

    why = (
        "The submitted text is not a complete privacy policy and does not provide "
        "the information required under GDPR transparency obligations."
    )
    fix = (
        "Submit a full privacy policy containing information about data collection, "
        "purposes, lawful basis, rights, retention, sharing, security, and contact details."
    )

    return {
        "combined_score": 0.0,
        "combined_score_percent": 0,
        "combined_label": "Non-Compliant",
        "overall_status": "Non-Compliant",
        "summary": "The submitted text is too short to be assessed as a valid privacy policy.",
        "word_count": word_count,
        "paragraph_count": 1,
        "llm_reviewed_paragraph_count": 0,
        "api_mode": "llm_only_whole_policy",
        "sections": [
            {
                "name": "General privacy policy completeness",
                "status": "missing",
                "note": "The submitted text does not contain enough information to evaluate GDPR compliance.",
            }
        ],
        "paragraph_results": [
            {
                "paragraph_id": 1,
                "policy_text": "General privacy policy completeness",
                "best_article_number": "Articles 12–14",
                "best_article_title": "Transparency and information duties",
                "section_name": "General privacy policy completeness",
                "heuristic_score": 0.0,
                "semantic_score": 0.0,
                "llm_verdict": "Non-Compliant",
                "llm_score": 0.0,
                "llm_assessment": (
                    f"GDPR article to address: Articles 12–14\n\n"
                    f"Why: {why}\n\n"
                    f"Fix: {fix}"
                ),
                "reviewed_by_llm": False,
                "combined_score": 0.0,
                "combined_label": "Non-Compliant",
            }
        ],
        "issues": [
            {
                "title": "General privacy policy completeness",
                "severity": "High",
                "description": (
                    f"GDPR article to address: Articles 12–14\n\n"
                    f"Why: {why}\n\n"
                    f"Fix: {fix}"
                ),
                "policy_section": "General privacy policy completeness",
                "gdpr_article": "Articles 12–14",
                "why": why,
                "fix": fix,
            }
        ],
        "recommendations": [
            {
                "number": 1,
                "text": "Submit a complete privacy policy before running the GDPR compliance check.",
            }
        ],
        "raw_llm_response": "",
    }


def safe_json_loads(raw: str) -> dict:
    raw_clean = clean_text(raw)

    try:
        return json.loads(raw_clean)
    except Exception:
        pass

    match = re.search(r"\{.*\}", raw_clean, flags=re.DOTALL)
    if match:
        candidate = match.group(0)

        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {
        "overall_status": "Non-Compliant",
        "compliance_score": 0,
        "summary": "The model response could not be parsed into the required structured JSON format.",
        "findings": [
            {
                "policy_section": "General GDPR compliance",
                "gdpr_article": "Articles 12–14",
                "why": (
                    "The assessment could not be parsed reliably. The submitted policy "
                    "requires manual review against GDPR transparency requirements."
                ),
                "fix": (
                    "Review the privacy policy manually and ensure GDPR information duties "
                    "are clearly addressed."
                ),
            }
        ],
        "raw_response": raw_clean,
    }


def build_prompt(policy_text: str) -> str:
    policy_text = policy_text[:MAX_POLICY_CHARS]

    return f"""
You are a GDPR compliance auditor reviewing a privacy policy.

Assess the whole policy as one document. Do not review paragraph by paragraph.

Return ONLY valid JSON. Do not include markdown. Do not include explanations outside JSON.

Use this exact JSON structure:

{{
  "overall_status": "Compliant",
  "compliance_score": 85,
  "summary": "Short overall explanation in plain English.",
  "findings": [
    {{
      "policy_section": "Name of the policy section or compliance area that needs improvement",
      "gdpr_article": "Relevant GDPR article or article range",
      "why": "Why this section is incomplete, unclear, or non-compliant",
      "fix": "Concrete action needed to improve compliance"
    }}
  ]
}}

Important rules:
- overall_status must be exactly one of: "Compliant", "Partially Compliant", "Non-Compliant".
- compliance_score must be an integer from 0 to 100.
- Return between 1 and {MAX_FINDINGS} findings.
- Do not mention paragraph numbers.
- Do not list every GDPR article.
- Focus only on the most important compliance gaps.
- Each finding must include policy_section, gdpr_article, why, and fix.
- If the submitted text is not a real privacy policy, return "Non-Compliant" and score 0.
- If the text is offensive, meaningless, or too short to be a privacy policy, return "Non-Compliant" and score 0.
- Use clear, concise language suitable for a thesis prototype.

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
            "gdpr_article": "Articles 12–14",
            "why": "The model did not return specific structured findings.",
            "fix": "Review the policy manually and ensure key GDPR transparency requirements are addressed.",
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


def build_sections(findings: list[dict], score: int) -> list[dict]:
    status = "missing" if score < 45 else "weak"

    return [
        {
            "name": finding["policy_section"],
            "status": status,
            "note": f"GDPR article to address: {finding['gdpr_article']}",
        }
        for finding in findings
    ]


def build_paragraph_results(findings: list[dict], overall_status: str, score: int) -> list[dict]:
    return [
        {
            "paragraph_id": i + 1,
            "policy_text": finding["policy_section"],
            "best_article_number": finding["gdpr_article"],
            "best_article_title": finding["policy_section"],
            "section_name": finding["policy_section"],
            "heuristic_score": 0.0,
            "semantic_score": 0.0,
            "llm_verdict": overall_status,
            "llm_score": round(score / 100, 3),
            "llm_assessment": (
                f"GDPR article to address: {finding['gdpr_article']}\n\n"
                f"Why: {finding['why']}\n\n"
                f"Fix: {finding['fix']}"
            ),
            "reviewed_by_llm": True,
            "combined_score": round(score / 100, 3),
            "combined_label": overall_status,
        }
        for i, finding in enumerate(findings)
    ]


def assess_policy_text(policy_text: str) -> dict:
    if not policy_text or not policy_text.strip():
        raise ValueError("No policy text provided.")

    policy_text = policy_text.strip()
    word_count = len(policy_text.split())

    if word_count < MIN_POLICY_WORDS:
        return invalid_policy_response(policy_text)

    prompt = build_prompt(policy_text)
    raw_reply = call_llm(prompt)

    print("\n\n===== RAW MODEL RESPONSE =====")
    print(raw_reply)
    print("===== END RAW MODEL RESPONSE =====\n\n")

    parsed = safe_json_loads(raw_reply)

    score = parsed.get("compliance_score", 0)

    try:
        score = int(float(score))
    except Exception:
        score = 0

    score = max(0, min(100, score))

    overall_status = normalise_status(parsed.get("overall_status", ""), score)
    findings = normalise_findings(parsed.get("findings", []))

    summary = clean_text(parsed.get("summary", "")) or (
        "The submitted policy was reviewed for high-level GDPR alignment."
    )

    issues = build_issues(findings, score)
    sections = build_sections(findings, score)
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
        "paragraph_count": 1,
        "llm_reviewed_paragraph_count": 1,
        "api_mode": "llm_only_whole_policy",
        "sections": sections,
        "paragraph_results": paragraph_results,
        "issues": issues,
        "recommendations": recommendations,
        "raw_llm_response": clean_text(raw_reply),
    }