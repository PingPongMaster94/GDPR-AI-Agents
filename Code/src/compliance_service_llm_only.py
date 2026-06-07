import json
import re

from src.llm_client import call_llm


MAX_POLICY_CHARS = 60000
MAX_STRENGTHS = 4
MAX_FINDINGS = 6
MIN_POLICY_WORDS = 50


def clean_text(text: str) -> str:
    text = str(text or "")

    text = re.sub(
        r"\x1b\[[0-9;]*[A-Za-z]",
        "",
        text,
    )
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"```", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def status_from_score(score: int) -> str:
    if score >= 75:
        return "Compliant"

    if score >= 45:
        return "Partially Compliant"

    return "Non-Compliant"


def safe_json_loads(raw: str) -> dict:
    raw_clean = str(raw or "").strip()

    raw_clean = re.sub(
        r"\x1b\[[0-9;]*[A-Za-z]",
        "",
        raw_clean,
    )
    raw_clean = re.sub(
        r"^```(?:json)?",
        "",
        raw_clean,
        flags=re.IGNORECASE,
    ).strip()
    raw_clean = re.sub(
        r"```$",
        "",
        raw_clean,
    ).strip()

    try:
        parsed = json.loads(raw_clean)

        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw_clean.find("{")

    if start == -1:
        raise ValueError(
            "The model response did not contain a JSON object."
        )

    brace_count = 0
    end = -1
    in_string = False
    escaped = False

    for index in range(start, len(raw_clean)):
        character = raw_clean[index]

        if escaped:
            escaped = False
            continue

        if character == "\\":
            escaped = True
            continue

        if character == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if character == "{":
            brace_count += 1

        elif character == "}":
            brace_count -= 1

            if brace_count == 0:
                end = index + 1
                break

    if end == -1:
        raise ValueError(
            "The model returned an incomplete JSON object."
        )

    candidate = raw_clean[start:end]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The model response could not be parsed into "
            "the required structured JSON format."
        ) from error

    if not isinstance(parsed, dict):
        raise ValueError(
            "The model response was not a JSON object."
        )

    return parsed


def count_policy_sections(policy_text: str) -> int:
    paragraphs = re.split(
        r"\n\s*\n",
        policy_text.strip(),
    )

    meaningful_paragraphs = [
        paragraph
        for paragraph in paragraphs
        if len(paragraph.strip()) > 50
    ]

    return max(
        1,
        len(meaningful_paragraphs),
    )


def invalid_policy_response(
    policy_text: str,
) -> dict:
    word_count = len(
        policy_text.split()
    )

    why = (
        "The submitted text does not contain enough information "
        "to be evaluated as a complete privacy policy."
    )

    recommendation = (
        "Submit a complete privacy policy covering personal data "
        "collection, processing purposes, lawful bases, data subject "
        "rights, retention, sharing, security, international transfers, "
        "and controller contact details."
    )

    issue = {
        "title": "Privacy policy completeness",
        "severity": "high",
        "description": (
            "GDPR article to address: Articles 12-14\n\n"
            f"Why: {why}"
        ),
        "policy_section": "Privacy policy completeness",
        "gdpr_article": "Articles 12-14",
        "why": why,
    }

    return {
        "combined_score": 0.0,
        "combined_score_percent": 0,
        "combined_label": "Non-Compliant",
        "overall_status": "Non-Compliant",
        "summary": (
            "The submitted text is too short to be assessed as a "
            "complete privacy policy."
        ),
        "analysis_method": {
            "title": "LLM-only whole-policy review",
            "description": (
                "The submitted text would normally be assessed directly "
                "by the language model without heuristic requirement "
                "checks or semantic GDPR article matching."
            ),
            "limitations": (
                "The submitted text was too short for the model analysis "
                "to be performed."
            ),
        },
        "strengths": [],
        "word_count": word_count,
        "paragraph_count": 1,
        "llm_reviewed_paragraph_count": 0,
        "llm_call_count": 0,
        "api_mode": "llm_only_whole_policy",
        "sections": [],
        "paragraph_results": [],
        "issues": [issue],
        "recommendations": [
            {
                "number": 1,
                "text": recommendation,
            }
        ],
        "raw_llm_response": "",
        "policy_was_truncated": False,
    }


def build_prompt(
    policy_text: str,
) -> str:
    was_truncated = (
        len(policy_text) > MAX_POLICY_CHARS
    )

    visible_policy = (
        policy_text[:MAX_POLICY_CHARS]
    )

    truncation_note = (
        "The submitted document exceeded the current analysis limit and "
        "was truncated. Do not treat an abrupt ending as evidence that the "
        "original privacy notice is incomplete. Assess only the substantive "
        "content that is visible."
        if was_truncated
        else (
            "The submitted privacy notice is included within the current "
            "analysis limit. Do not claim that sections or sentences were "
            "cut off unless the submitted text itself clearly demonstrates it."
        )
    )

    return f"""
You are reviewing a privacy notice for its likely alignment with GDPR
transparency and information requirements.

Assess the submitted privacy notice as one complete document.

The main purpose of this assessment is to evaluate the completeness,
clarity, accessibility, and likely GDPR alignment of the privacy notice,
with particular emphasis on Articles 12, 13, and 14.

This is not an audit of the organisation's entire GDPR compliance
programme. Distinguish carefully between:

1. Information that must or may need to be disclosed in a privacy notice;
2. Broader operational, governance, security, and accountability duties
   that the controller may need to perform internally.

This is an LLM-only assessment. No deterministic GDPR requirement
checklist, heuristic score, or semantic GDPR article retrieval evidence
is available.

Your assessment must be balanced. Identify both the main strengths of
the privacy notice and the most important disclosure gaps.

Return ONLY valid JSON.
Do not use markdown.
Do not include text outside the JSON object.

Use this exact structure:

{{
  "overall_status": "Partially Compliant",
  "compliance_score": 65,
  "summary": "Concise overall explanation balancing strengths and weaknesses.",
  "strengths": [
    {{
      "title": "Clear processing purposes",
      "evidence": "A short quotation or concise summary of the relevant privacy-notice content.",
      "gdpr_relevance": "Why this contributes to GDPR transparency or information duties."
    }}
  ],
  "findings": [
    {{
      "policy_section": "Data retention",
      "gdpr_article": "Articles 5(1)(e), 13(2)(a), 14(2)(a)",
      "why": "Why this disclosure is incomplete, unclear, or potentially non-compliant.",
      "recommendation": "Concrete action required to improve the privacy notice."
    }}
  ]
}}

Score thresholds:

- 75 to 100: Compliant
- 45 to 74: Partially Compliant
- 0 to 44: Non-Compliant

Core assessment rules:

- compliance_score must be an integer from 0 to 100.
- overall_status must be consistent with the score thresholds.
- Return between 1 and {MAX_STRENGTHS} strengths when genuine strengths exist.
- Return between 0 and {MAX_FINDINGS} findings.
- Do not invent strengths merely to fill the strengths array.
- Do not invent compliance gaps.
- Each strength must include title, evidence, and gdpr_relevance.
- Each finding must include policy_section, gdpr_article, why, and recommendation.
- Focus on substantive privacy-notice disclosure requirements.
- Give primary attention to Articles 12, 13, and 14 and to other provisions
  that directly create information duties relevant to the privacy notice.
- Do not treat every obligation contained in the GDPR as information that
  must appear in the privacy notice.
- Do not assess the organisation's complete internal GDPR governance,
  accountability, or operational compliance based only on its privacy notice.
- Do not penalise the notice for company descriptions, addresses, contact
  information, navigation text, introductory material, or contextual content.
- Evaluate whether required information is meaningfully present, not what
  percentage of the document it occupies.
- Do not require exact GDPR terminology when equivalent meaning is present.
- Consider the privacy notice as a whole before assigning the score.
- Minor drafting improvements should not automatically make an otherwise
  complete privacy notice non-compliant.
- Do not mention paragraph numbers or chunk numbers.
- Do not state that content was cut off unless supported by the document
  handling note below.

Data Protection Officer rules:

- DPO contact details are privacy-notice information only where a DPO is
  applicable or has been appointed.
- Do not assume that every organisation is required to appoint a DPO.
- Do not require the privacy notice to justify why a DPO has not been appointed.
- Do not create a finding merely because a general privacy contact, privacy
  manager, or privacy lead is used instead of the title "Data Protection Officer".
- Create a DPO-related finding only when the policy indicates that a DPO exists
  or is applicable but does not provide an adequate way to contact that DPO.

Personal-data breach rules:

- Articles 33 and 34 establish breach-response and notification duties.
- Do not treat the absence of internal breach procedures as a privacy-notice gap.
- Do not require the privacy notice to promise or explain notification to the
  supervisory authority or affected data subjects following a breach.
- Do not create a finding merely because breach-response procedures are not
  described in the privacy notice.
- If the policy voluntarily describes breach procedures, assess whether those
  statements are clear and non-misleading, but do not require their inclusion.

Other operational-compliance rules:

- Do not require descriptions of internal processor contracts under Article 28.
- Do not require records of processing activities, DPIA procedures, staff
  training, audit arrangements, or internal accountability documentation.
- Do not require a detailed description of all technical and organisational
  security measures under Article 32.
- General security information may be treated as a positive feature, but the
  absence of detailed internal security controls is not automatically a
  privacy-notice deficiency.
- For international transfers, assess whether applicable transfers, destinations
  or categories of destination, safeguards, and methods for obtaining further
  information are adequately explained.
- For automated decision-making, evaluate the disclosure only where such
  processing is stated, suggested, or applicable. Do not assume it occurs merely
  because analytics, marketing, or personalisation are mentioned.
- Do not require disclosure about processing activities that genuinely do not
  apply, unless the notice would be materially unclear without clarification.

Document handling note:

{truncation_note}

Privacy notice:

\"\"\"
{visible_policy}
\"\"\"
""".strip()


def normalise_strengths(
    strengths,
) -> list[dict]:
    if not isinstance(strengths, list):
        strengths = []

    cleaned_strengths: list[dict] = []

    for strength in strengths[:MAX_STRENGTHS]:
        if not isinstance(strength, dict):
            continue

        title = clean_text(
            strength.get(
                "title",
                "",
            )
        )

        evidence = clean_text(
            strength.get(
                "evidence",
                "",
            )
        )

        gdpr_relevance = clean_text(
            strength.get(
                "gdpr_relevance",
                strength.get(
                    "relevance",
                    "",
                ),
            )
        )

        if not title or not evidence:
            continue

        cleaned_strengths.append({
            "title": title,
            "evidence": evidence,
            "gdpr_relevance": (
                gdpr_relevance
                or (
                    "This contributes to the transparency and "
                    "information requirements of the GDPR."
                )
            ),
        })

    return cleaned_strengths


def normalise_findings(
    findings,
    score: int,
) -> list[dict]:
    if not isinstance(findings, list):
        findings = []

    cleaned_findings: list[dict] = []

    for finding in findings[:MAX_FINDINGS]:
        if not isinstance(finding, dict):
            continue

        policy_section = clean_text(
            finding.get(
                "policy_section",
                "",
            )
        )

        gdpr_article = clean_text(
            finding.get(
                "gdpr_article",
                "",
            )
        )

        why = clean_text(
            finding.get(
                "why",
                "",
            )
        )

        recommendation = clean_text(
            finding.get(
                "recommendation",
                finding.get(
                    "fix",
                    "",
                ),
            )
        )

        if not policy_section or not why:
            continue

        cleaned_findings.append({
            "policy_section": policy_section,
            "gdpr_article": (
                gdpr_article
                or "Relevant GDPR provisions"
            ),
            "why": why,
            "recommendation": (
                recommendation
                or (
                    "Review and clarify this part of the "
                    "privacy policy."
                )
            ),
        })

    if (
        score < 75
        and not cleaned_findings
    ):
        cleaned_findings.append({
            "policy_section": (
                "General GDPR transparency"
            ),
            "gdpr_article": "Articles 12-14",
            "why": (
                "The assessment indicates that the policy requires "
                "improvement, but the model did not provide a specific "
                "structured finding."
            ),
            "recommendation": (
                "Review the policy against the GDPR transparency "
                "and information requirements."
            ),
        })

    return cleaned_findings


def build_issues(
    findings: list[dict],
    score: int,
) -> list[dict]:
    severity = (
        "high"
        if score < 45
        else "medium"
    )

    return [
        {
            "title": finding[
                "policy_section"
            ],
            "severity": severity,
            "description": (
                f"GDPR article to address: "
                f"{finding['gdpr_article']}\n\n"
                f"Why: {finding['why']}"
            ),
            "policy_section": (
                finding["policy_section"]
            ),
            "gdpr_article": (
                finding["gdpr_article"]
            ),
            "why": finding["why"],
        }
        for finding in findings
    ]


def build_recommendations(
    findings: list[dict],
) -> list[dict]:
    return [
        {
            "number": index + 1,
            "text": (
                f"{finding['policy_section']}: "
                f"{finding['recommendation']}"
            ),
        }
        for index, finding
        in enumerate(findings)
    ]


def build_paragraph_results(
    findings: list[dict],
    overall_status: str,
    score: int,
) -> list[dict]:
    return [
        {
            "paragraph_id": index + 1,
            "policy_text": (
                finding["policy_section"]
            ),
            "best_article_number": (
                finding["gdpr_article"]
            ),
            "best_article_title": (
                finding["policy_section"]
            ),
            "section_name": (
                finding["policy_section"]
            ),
            "heuristic_score": 0.0,
            "semantic_score": 0.0,
            "llm_verdict": overall_status,
            "llm_score": round(
                score / 100,
                3,
            ),
            "llm_assessment": (
                f"GDPR article to address: "
                f"{finding['gdpr_article']}\n\n"
                f"Why: {finding['why']}"
            ),
            "reviewed_by_llm": True,
            "combined_score": round(
                score / 100,
                3,
            ),
            "combined_label": overall_status,
        }
        for index, finding
        in enumerate(findings)
    ]


def assess_policy_text(
    policy_text: str,
) -> dict:
    if not policy_text or not policy_text.strip():
        raise ValueError(
            "No policy text provided."
        )

    policy_text = policy_text.strip()

    word_count = len(
        policy_text.split()
    )

    if word_count < MIN_POLICY_WORDS:
        return invalid_policy_response(
            policy_text
        )

    was_truncated = (
        len(policy_text) > MAX_POLICY_CHARS
    )

    prompt = build_prompt(
        policy_text
    )

    raw_reply = call_llm(
        prompt
    )

    print(
        "\n===== RAW LLM-ONLY ASSESSMENT ====="
    )
    print(raw_reply)
    print(
        "===== END LLM-ONLY ASSESSMENT =====\n"
    )

    parsed = safe_json_loads(
        raw_reply
    )

    try:
        score = int(
            float(
                parsed.get(
                    "compliance_score"
                )
            )
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "The model did not return a valid compliance score."
        ) from error

    score = max(
        0,
        min(100, score),
    )

    overall_status = (
        status_from_score(score)
    )

    summary = clean_text(
        parsed.get(
            "summary",
            "",
        )
    )

    if not summary:
        raise ValueError(
            "The model did not return an assessment summary."
        )

    strengths = normalise_strengths(
        parsed.get(
            "strengths",
            [],
        )
    )

    findings = normalise_findings(
        parsed.get(
            "findings",
            [],
        ),
        score,
    )

    issues = build_issues(
        findings,
        score,
    )

    recommendations = (
        build_recommendations(
            findings
        )
    )

    paragraph_results = (
        build_paragraph_results(
            findings,
            overall_status,
            score,
        )
    )

    return {
        "combined_score": round(
            score / 100,
            3,
        ),
        "combined_score_percent": score,
        "combined_label": overall_status,
        "overall_status": overall_status,
        "summary": summary,
        "analysis_method": {
            "title": "LLM-only privacy-notice review",
            "description": (
                "The complete submitted privacy notice was assessed directly "
                "by the language model, with primary emphasis on the GDPR "
                "transparency and information requirements applicable to "
                "privacy notices."
    ),
            "limitations": (
                "This mode does not use heuristic requirement checks, semantic "
                "GDPR article matching, or the structured traceability evidence "
                "available in Hybrid mode. It evaluates the submitted notice "
                "rather than the organisation's complete GDPR compliance programme."
    ),
},
        "strengths": strengths,
        "word_count": word_count,
        "paragraph_count": (
            count_policy_sections(
                policy_text
            )
        ),
        "llm_reviewed_paragraph_count": 1,
        "llm_call_count": 1,
        "api_mode": "llm_only_whole_policy",
        "sections": [],
        "paragraph_results": (
            paragraph_results
        ),
        "issues": issues,
        "recommendations": (
            recommendations
        ),
        "raw_llm_response": clean_text(
            raw_reply
        ),
        "policy_was_truncated": (
            was_truncated
        ),
    }