from pathlib import Path
import json
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm_client import call_llm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GDPR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reference_law_articles.csv"
)

MAX_POLICY_CHARS = 60000
MAX_EVIDENCE_ITEMS = 8


GDPR_REQUIREMENTS = {
    "controller_identity": {
        "policy_section": (
            "Controller identity and contact details"
        ),
        "gdpr_article": (
            "Articles 13(1)(a), 14(1)(a)"
        ),
        "description": (
            "The policy should identify the data controller "
            "and provide accessible contact details."
        ),
        "patterns": [
            r"\b(controller|data controller|we are responsible|company responsible)\b",
            r"\b(contact us|privacy team|privacy office|privacy manager|dpo|data protection officer)\b",
            r"\b(email|address|postal address|registered office|contact details)\b",
        ],
    },
    "purposes": {
        "policy_section": "Purposes of processing",
        "gdpr_article": (
            "Articles 5(1)(b), 13(1)(c), 14(1)(c)"
        ),
        "description": (
            "The policy should explain why personal data "
            "is processed."
        ),
        "patterns": [
            r"\b(purpose|purposes|we use|we process|used to|in order to|so that we can)\b",
            r"\b(provide|deliver|operate|improve|personalise|communicate|support|security|marketing)\b",
        ],
    },
    "lawful_basis": {
        "policy_section": "Lawful basis for processing",
        "gdpr_article": "Article 6",
        "description": (
            "The policy should identify or explain the lawful "
            "basis for processing personal data."
        ),
        "patterns": [
            r"\b(lawful basis|legal basis|legal ground|basis for processing)\b",
            r"\b(consent|contract|legitimate interest|legal obligation|vital interests|public task)\b",
            r"\b(necessary to perform|necessary for our legitimate interests|required by law|with your consent)\b",
        ],
    },
    "data_categories": {
        "policy_section": (
            "Categories of personal data"
        ),
        "gdpr_article": (
            "Articles 13(1)(c), 14(1)(d)"
        ),
        "description": (
            "The policy should describe the categories of "
            "personal data collected or processed."
        ),
        "patterns": [
            r"\b(personal data|personal information|information we collect|data we collect)\b",
            r"\b(name|email|address|phone|payment|account|device|ip address|location|usage data|cookies)\b",
            r"\b(categories of data|types of data|information such as)\b",
        ],
    },
    "data_subject_rights": {
        "policy_section": "Data subject rights",
        "gdpr_article": "Articles 12, 15-22",
        "description": (
            "The policy should explain the rights available "
            "to data subjects and how those rights may be "
            "exercised."
        ),
        "patterns": [
            r"\b(your rights|data subject rights|privacy rights|rights under)\b",
            r"\b(access|rectification|erasure|delete|restriction|portability|object|withdraw consent)\b",
            r"\b(request|exercise your rights|contact us to exercise)\b",
        ],
    },
    "retention": {
        "policy_section": "Data retention",
        "gdpr_article": (
            "Article 5(1)(e), Article 13(2)(a), "
            "Article 14(2)(a)"
        ),
        "description": (
            "The policy should explain how long data is "
            "retained or the criteria used to determine "
            "retention."
        ),
        "patterns": [
            r"\b(retain|retention|keep|store|storage period|how long)\b",
            r"\b(as long as necessary|for as long as|retention period|deleted after|criteria used)\b",
            r"\b(legal obligation|business purpose|account closure|until no longer needed)\b",
        ],
    },
    "recipients": {
        "policy_section": (
            "Recipients and third-party sharing"
        ),
        "gdpr_article": (
            "Articles 13(1)(e), 14(1)(e), 28"
        ),
        "description": (
            "The policy should explain whether data is shared "
            "with recipients, processors, or third parties."
        ),
        "patterns": [
            r"\b(share|disclose|transfer|provide|recipient|third party|third parties)\b",
            r"\b(service provider|processor|vendor|partner|affiliate|supplier|contractor)\b",
            r"\b(hosting|analytics|payment|support|marketing|security provider)\b",
        ],
    },
    "international_transfers": {
        "policy_section": (
            "International data transfers"
        ),
        "gdpr_article": "Articles 44-49",
        "description": (
            "The policy should explain international "
            "transfers and the safeguards used where "
            "applicable."
        ),
        "patterns": [
            r"\b(international transfer|transfer outside|outside the eea|outside the eu|outside the uk)\b",
            r"\b(standard contractual clauses|scc|adequacy|safeguards|data privacy framework)\b",
            r"\b(other countries|global|worldwide|internationally)\b",
        ],
    },
    "security": {
        "policy_section": "Security of processing",
        "gdpr_article": "Article 32",
        "description": (
            "The policy should describe relevant technical "
            "or organisational security measures."
        ),
        "patterns": [
            r"\b(security|secure|protect|safeguard|confidentiality)\b",
            r"\b(encryption|access controls|authentication|monitoring|technical and organisational measures)\b",
            r"\b(unauthorised access|loss|misuse|breach|incident)\b",
        ],
    },
    "complaints": {
        "policy_section": (
            "Right to lodge a complaint"
        ),
        "gdpr_article": (
            "Article 13(2)(d), Article 14(2)(e), "
            "Article 77"
        ),
        "description": (
            "The policy should mention the right to lodge "
            "a complaint with a supervisory authority."
        ),
        "patterns": [
            r"\b(complaint|lodge a complaint|supervisory authority|data protection authority)\b",
            r"\b(ico|cnpd|dpc|edpb|regulator|authority)\b",
        ],
    },
    "automated_decision_making": {
        "policy_section": (
            "Automated decision-making and profiling"
        ),
        "gdpr_article": (
            "Article 13(2)(f), Article 14(2)(g), "
            "Article 22"
        ),
        "description": (
            "The policy should state whether automated "
            "decision-making or profiling is used."
        ),
        "patterns": [
            r"\b(automated decision|automated decision-making|solely automated|profiling)\b",
            r"\b(algorithm|automated processing|personalised recommendations|targeted advertising)\b",
            r"\b(legal effects|similarly significant effects|logic involved)\b",
        ],
    },
}


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


def split_into_chunks(text: str) -> list[str]:
    parts = re.split(
        r"\n\s*\n",
        text.strip(),
    )

    if len(parts) <= 1:
        parts = re.split(
            r"(?=\n?\d+\.\s+[A-Z])|"
            r"(?=\n?[A-Z][A-Za-z\s]{3,80}\n)",
            text,
        )

    parts = [
        part.strip()
        for part in parts
        if len(part.strip()) > 80
    ]

    chunks: list[str] = []

    for part in parts:
        if len(part) <= 1200:
            chunks.append(part)
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+",
            part,
        )

        buffer = ""

        for sentence in sentences:
            candidate = (
                f"{buffer} {sentence}"
            ).strip()

            if len(candidate) <= 1000:
                buffer = candidate
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


def semantic_match_chunks(
    chunks: list[str],
    gdpr_df: pd.DataFrame,
) -> list[dict]:
    articles = gdpr_df[
        gdpr_df["section_type"] == "Article"
    ].copy()

    if articles.empty:
        return []

    article_texts = (
        articles["text"]
        .fillna("")
        .tolist()
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
    )

    vectorizer.fit(article_texts + chunks)

    article_vectors = vectorizer.transform(
        article_texts
    )

    policy_vectors = vectorizer.transform(
        chunks
    )

    similarity_matrix = cosine_similarity(
        policy_vectors,
        article_vectors,
    )

    matches: list[dict] = []

    for index, similarity_row in enumerate(
        similarity_matrix
    ):
        best_index = int(
            similarity_row.argmax()
        )

        article = articles.iloc[best_index]

        article_title = article.get(
            "title",
            "",
        )

        if pd.isna(article_title):
            article_title = ""

        matches.append({
            "chunk_id": index + 1,
            "text": chunks[index],
            "best_article_number": str(
                article["number"]
            ),
            "best_article_title": str(
                article_title
            ),
            "semantic_score": round(
                float(
                    similarity_row[best_index]
                ),
                3,
            ),
        })

    return matches


def evaluate_requirement(
    policy_text: str,
    chunks: list[str],
    requirement: dict,
) -> dict:
    matched_patterns: list[str] = []
    matched_chunks: list[dict] = []

    for pattern in requirement["patterns"]:
        if re.search(
            pattern,
            policy_text,
            flags=re.IGNORECASE,
        ):
            matched_patterns.append(pattern)

    for index, chunk in enumerate(chunks):
        if any(
            re.search(
                pattern,
                chunk,
                flags=re.IGNORECASE,
            )
            for pattern in requirement["patterns"]
        ):
            matched_chunks.append({
                "chunk_id": index + 1,
                "text_preview": chunk[:300],
            })

    total_patterns = len(
        requirement["patterns"]
    )

    heuristic_score = (
        len(matched_patterns) / total_patterns
        if total_patterns
        else 0.0
    )

    if matched_chunks and heuristic_score < 0.5:
        heuristic_score = 0.5

    return {
        "policy_section": (
            requirement["policy_section"]
        ),
        "gdpr_article": (
            requirement["gdpr_article"]
        ),
        "description": (
            requirement["description"]
        ),
        "heuristic_coverage_score": round(
            heuristic_score,
            3,
        ),
        "matched_patterns": matched_patterns,
        "matched_chunk_count": len(
            matched_chunks
        ),
        "matched_chunk_examples": (
            matched_chunks[:3]
        ),
    }


def build_requirement_evidence(
    policy_text: str,
    chunks: list[str],
) -> list[dict]:
    return [
        evaluate_requirement(
            policy_text,
            chunks,
            requirement,
        )
        for requirement
        in GDPR_REQUIREMENTS.values()
    ]


def build_requirement_review_prompt(
    policy_text: str,
    requirement_evidence: list[dict],
) -> str:
    visible_policy = (
        policy_text[:MAX_POLICY_CHARS]
    )

    evidence_json = json.dumps(
        requirement_evidence,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are reviewing GDPR requirement coverage in a privacy policy.

Review every supplied GDPR requirement against the actual policy text.

Return ONLY valid JSON. Do not use markdown or add text outside JSON.

Use this exact structure:

{{
  "requirements": [
    {{
      "policy_section": "Controller identity and contact details",
      "status": "PRESENT",
      "gdpr_article": "Articles 13(1)(a), 14(1)(a)",
      "evidence": "Short quotation or concise summary of relevant policy evidence.",
      "reason": "Why the requirement is present, weak, or missing.",
      "recommendation": "Concrete corrective action, or an empty string when present."
    }}
  ]
}}

Status must be exactly:
- PRESENT
- WEAK
- MISSING

Rules:
- Preserve every policy_section name exactly.
- Review every supplied requirement.
- Recognise equivalent wording and meaning.
- Do not depend on exact GDPR terminology.
- PRESENT means clearly and sufficiently addressed.
- WEAK means addressed but incomplete, vague, or insufficiently specific.
- MISSING means not meaningfully addressed.
- For PRESENT requirements, recommendation may be empty.
- For WEAK or MISSING requirements, provide a practical recommendation.
- Extra company information must not reduce coverage.
- Evaluate what the policy contains, not how much text is devoted to it.
- Preliminary heuristic evidence is advisory only.

Preliminary evidence:
{evidence_json}

Privacy policy:
\"\"\"
{visible_policy}
\"\"\"
""".strip()


def normalise_requirement_reviews(
    parsed: dict,
    requirement_evidence: list[dict],
) -> list[dict]:
    reviews = parsed.get(
        "requirements",
        [],
    )

    if not isinstance(reviews, list):
        raise ValueError(
            "The requirement review did not return "
            "a requirements list."
        )

    review_map: dict[str, dict] = {}

    for review in reviews:
        if not isinstance(review, dict):
            continue

        section = clean_text(
            review.get(
                "policy_section",
                "",
            )
        )

        if not section:
            continue

        status = clean_text(
            review.get(
                "status",
                "",
            )
        ).upper()

        if status not in {
            "PRESENT",
            "WEAK",
            "MISSING",
        }:
            status = "WEAK"

        review_map[section.casefold()] = {
            "llm_status": status,
            "llm_evidence": clean_text(
                review.get(
                    "evidence",
                    "",
                )
            ),
            "llm_reason": clean_text(
                review.get(
                    "reason",
                    "",
                )
            ),
            "llm_recommendation": clean_text(
                review.get(
                    "recommendation",
                    "",
                )
            ),
        }

    normalised: list[dict] = []

    for item in requirement_evidence:
        section_key = (
            item["policy_section"].casefold()
        )

        review = review_map.get(
            section_key
        )

        if not review:
            raise ValueError(
                "The requirement review omitted: "
                f"{item['policy_section']}."
            )

        normalised.append({
            **item,
            **review,
        })

    return normalised


def build_semantic_evidence(
    semantic_rows: list[dict],
) -> dict:
    top_matches = sorted(
        semantic_rows,
        key=lambda row: row["semantic_score"],
        reverse=True,
    )[:MAX_EVIDENCE_ITEMS]

    article_counts: dict[str, dict] = {}

    for row in semantic_rows:
        key = (
            f"Article "
            f"{row['best_article_number']}"
        )

        if key not in article_counts:
            article_counts[key] = {
                "gdpr_article": key,
                "article_title": (
                    row["best_article_title"]
                ),
                "matched_chunks": 0,
                "max_similarity": 0.0,
            }

        article_counts[key][
            "matched_chunks"
        ] += 1

        article_counts[key][
            "max_similarity"
        ] = max(
            article_counts[key][
                "max_similarity"
            ],
            row["semantic_score"],
        )

    article_summary = sorted(
        article_counts.values(),
        key=lambda item: (
            item["matched_chunks"],
            item["max_similarity"],
        ),
        reverse=True,
    )[:MAX_EVIDENCE_ITEMS]

    return {
        "top_semantic_matches": [
            {
                "chunk_id": row["chunk_id"],
                "gdpr_article": (
                    f"Article "
                    f"{row['best_article_number']}"
                ),
                "article_title": (
                    row["best_article_title"]
                ),
                "semantic_score": (
                    row["semantic_score"]
                ),
                "text_preview": (
                    row["text"][:300]
                ),
            }
            for row in top_matches
        ],
        "article_summary": article_summary,
    }


def build_hybrid_evidence(
    policy_text: str,
    gdpr_df: pd.DataFrame,
) -> dict:
    chunks = split_into_chunks(
        policy_text
    )

    semantic_rows = semantic_match_chunks(
        chunks,
        gdpr_df,
    )

    preliminary_evidence = (
        build_requirement_evidence(
            policy_text,
            chunks,
        )
    )

    requirement_prompt = (
        build_requirement_review_prompt(
            policy_text,
            preliminary_evidence,
        )
    )

    raw_requirement_review = call_llm(
        requirement_prompt
    )

    print(
        "\n===== RAW REQUIREMENT REVIEW ====="
    )
    print(raw_requirement_review)
    print(
        "===== END REQUIREMENT REVIEW =====\n"
    )

    parsed_requirement_review = (
        safe_json_loads(
            raw_requirement_review
        )
    )

    requirement_evidence = (
        normalise_requirement_reviews(
            parsed_requirement_review,
            preliminary_evidence,
        )
    )

    semantic_evidence = (
        build_semantic_evidence(
            semantic_rows
        )
    )

    status_scores = {
        "PRESENT": 1.0,
        "WEAK": 0.5,
        "MISSING": 0.0,
    }

    requirement_coverage_score = (
        sum(
            status_scores[
                item["llm_status"]
            ]
            for item
            in requirement_evidence
        )
        / len(requirement_evidence)
    )

    return {
        "chunk_count": len(chunks),
        "requirement_coverage_score": round(
            requirement_coverage_score,
            3,
        ),
        "present_areas": [
            item["policy_section"]
            for item
            in requirement_evidence
            if item["llm_status"]
            == "PRESENT"
        ],
        "weak_areas": [
            item["policy_section"]
            for item
            in requirement_evidence
            if item["llm_status"]
            == "WEAK"
        ],
        "missing_areas": [
            item["policy_section"]
            for item
            in requirement_evidence
            if item["llm_status"]
            == "MISSING"
        ],
        "requirement_evidence": (
            requirement_evidence
        ),
        "semantic_evidence": (
            semantic_evidence
        ),
    }


def build_final_assessment_prompt(
    policy_text: str,
    evidence: dict,
) -> str:
    visible_policy = (
        policy_text[:MAX_POLICY_CHARS]
    )

    compact_evidence = {
        "requirement_coverage_score": (
            evidence[
                "requirement_coverage_score"
            ]
        ),
        "present_areas": (
            evidence["present_areas"]
        ),
        "weak_areas": (
            evidence["weak_areas"]
        ),
        "missing_areas": (
            evidence["missing_areas"]
        ),
        "requirements": [
            {
                "policy_section": (
                    item["policy_section"]
                ),
                "gdpr_article": (
                    item["gdpr_article"]
                ),
                "status": (
                    item["llm_status"]
                ),
                "evidence": (
                    item["llm_evidence"]
                ),
                "reason": (
                    item["llm_reason"]
                ),
            }
            for item
            in evidence[
                "requirement_evidence"
            ]
        ],
        "semantic_article_summary": (
            evidence[
                "semantic_evidence"
            ]["article_summary"]
        ),
    }

    evidence_json = json.dumps(
        compact_evidence,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are a GDPR compliance auditor.

Assess the privacy policy as one complete document.

Use the structured evidence for traceability and legal grounding.
The final score must reflect the overall quality, completeness,
clarity, and transparency of the policy.

Return ONLY valid JSON. Do not use markdown or add text outside JSON.

Use this exact structure:

{{
  "overall_status": "Compliant",
  "compliance_score": 78,
  "summary": "Concise explanation of the overall assessment."
}}

Score thresholds:
- 75 to 100: Compliant
- 45 to 74: Partially Compliant
- 0 to 44: Non-Compliant

Rules:
- compliance_score must be an integer from 0 to 100.
- overall_status must match the score threshold.
- Additional company information must not reduce the score.
- Do not score based on keyword density.
- Assess what the policy meaningfully contains.
- Do not penalise semantically equivalent wording.
- Consider both strengths and weaknesses.
- Do not repeat detailed findings because they are already represented in the requirement evidence.

Hybrid evidence:
{evidence_json}

Privacy policy:
\"\"\"
{visible_policy}
\"\"\"
""".strip()


def build_sections_from_evidence(
    evidence: dict,
) -> list[dict]:
    status_map = {
        "PRESENT": "present",
        "WEAK": "weak",
        "MISSING": "missing",
    }

    return [
        {
            "name": item["policy_section"],
            "status": status_map[
                item["llm_status"]
            ],
            "note": (
                f"{item['description']}\n"
                f"GDPR reference: "
                f"{item['gdpr_article']}\n"
                f"Evidence: "
                f"{item['llm_evidence'] or 'No specific evidence was identified.'}\n"
                f"Assessment: "
                f"{item['llm_reason'] or 'No explanation was provided.'}"
            ),
        }
        for item
        in evidence["requirement_evidence"]
    ]


def build_gap_outputs(
    evidence: dict,
) -> tuple[list[dict], list[dict]]:
    status_order = {
        "MISSING": 0,
        "WEAK": 1,
    }

    gap_items = [
        item
        for item
        in evidence["requirement_evidence"]
        if item["llm_status"]
        in {"MISSING", "WEAK"}
    ]

    gap_items.sort(
        key=lambda item: (
            status_order[
                item["llm_status"]
            ],
            item["policy_section"],
        )
    )

    issues: list[dict] = []
    recommendations: list[dict] = []

    for index, item in enumerate(
        gap_items
    ):
        reason = (
            item["llm_reason"]
            or (
                "The policy does not provide "
                "sufficient information for "
                "this requirement."
            )
        )

        recommendation = (
            item["llm_recommendation"]
            or (
                "Review and improve the "
                f"{item['policy_section'].lower()} "
                "section."
            )
        )

        issues.append({
            "title": item["policy_section"],
            "severity": (
                "high"
                if item["llm_status"]
                == "MISSING"
                else "medium"
            ),
            "description": (
                f"GDPR article to address: "
                f"{item['gdpr_article']}\n\n"
                f"Why: {reason}"
            ),
            "policy_section": (
                item["policy_section"]
            ),
            "gdpr_article": (
                item["gdpr_article"]
            ),
            "why": reason,
            "coverage_status": (
                item["llm_status"]
            ),
        })

        recommendations.append({
            "number": index + 1,
            "text": (
                f"{item['policy_section']}: "
                f"{recommendation}"
            ),
        })

    return issues, recommendations


def build_paragraph_results(
    issues: list[dict],
    overall_status: str,
    score: int,
) -> list[dict]:
    return [
        {
            "paragraph_id": index + 1,
            "policy_text": (
                issue["policy_section"]
            ),
            "best_article_number": (
                issue["gdpr_article"]
            ),
            "best_article_title": (
                issue["policy_section"]
            ),
            "section_name": (
                issue["policy_section"]
            ),
            "heuristic_score": (
                0.0
                if issue["coverage_status"]
                == "MISSING"
                else 0.5
            ),
            "semantic_score": 0.0,
            "llm_verdict": overall_status,
            "llm_score": round(
                score / 100,
                3,
            ),
            "llm_assessment": (
                f"GDPR article to address: "
                f"{issue['gdpr_article']}\n\n"
                f"Why: {issue['why']}"
            ),
            "reviewed_by_llm": True,
            "combined_score": round(
                score / 100,
                3,
            ),
            "combined_label": overall_status,
        }
        for index, issue
        in enumerate(issues)
    ]


def assess_policy_text(
    policy_text: str,
) -> dict:
    if not policy_text or not policy_text.strip():
        raise ValueError(
            "No policy text provided."
        )

    if not GDPR_FILE.exists():
        raise FileNotFoundError(
            f"Missing GDPR reference file: "
            f"{GDPR_FILE}"
        )

    policy_text = policy_text.strip()
    word_count = len(
        policy_text.split()
    )

    gdpr_df = pd.read_csv(
        GDPR_FILE
    )

    evidence = build_hybrid_evidence(
        policy_text,
        gdpr_df,
    )

    final_prompt = (
        build_final_assessment_prompt(
            policy_text,
            evidence,
        )
    )

    raw_reply = call_llm(
        final_prompt
    )

    print(
        "\n===== RAW HYBRID ASSESSMENT ====="
    )
    print(raw_reply)
    print(
        "===== END HYBRID ASSESSMENT =====\n"
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
            "The model did not return a valid "
            "compliance score."
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
            "The model did not return an "
            "assessment summary."
        )

    sections = (
        build_sections_from_evidence(
            evidence
        )
    )

    issues, recommendations = (
        build_gap_outputs(
            evidence
        )
    )

    paragraph_results = (
        build_paragraph_results(
            issues,
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
        "word_count": word_count,
        "paragraph_count": (
            evidence["chunk_count"]
        ),
        "llm_reviewed_paragraph_count": 2,
        "llm_call_count": 2,
        "api_mode": (
            "hybrid_llm_requirement_"
            "semantic_traceability"
        ),
        "sections": sections,
        "paragraph_results": (
            paragraph_results
        ),
        "issues": issues,
        "recommendations": (
            recommendations
        ),
        "hybrid_evidence": evidence,
        "raw_llm_response": clean_text(
            raw_reply
        ),
    }