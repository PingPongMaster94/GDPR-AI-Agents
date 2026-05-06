# combine_scores_LLM.py
# Complete GDPR compliance pipeline with:
# - document-type classification
# - strict privacy-policy allowlist
# - rule-based rejection of obvious non-policy text
# - heuristic scoring
# - semantic scoring
# - local LLM compliance reasoning via Ollama
# - JSONL caching
# - CSV output

import os
import time
import json
import hashlib
import subprocess
from pathlib import Path

import pandas as pd

from gdpr_agent import GDPRComplianceAgent


# ========== PATHS (RELATIVE TO PROJECT ROOT) ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_FILE = PROJECT_ROOT / "data/annotated/paragraphs_with_articles.csv"
GDPR_FILE   = PROJECT_ROOT / "data/reference_law_articles.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/annotated/combined_compliance_scoresV3.csv"
CACHE_FILE  = PROJECT_ROOT / "data/annotated/llm_assessment_cache.jsonl"


# ========== SCORING PARAMS ==========
HEURISTIC_WEIGHT = 0.25
SEMANTIC_WEIGHT  = 0.25
LLM_WEIGHT       = 0.50

THRESHOLDS = {
    "compliant": 0.55,
    "partial": 0.30,
}


# ========== PIPELINE PARAMS ==========
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
MAX_CHARS = 800
TIMEOUT_S = 420
RETRIES = 2
PREWARM_PROMPT = "Say 'ready' once."
MIN_POLICY_CHARS = 120

ALLOWED_POLICY_TYPES = {"PrivacyPolicy", "PrivacyNotice"}

REJECT_TYPES = {
    "TermsOfService",
    "CookieBanner",
    "Marketing",
    "FAQ",
    "ProductPage",
    "GenericWebpage",
    "Contract",
    "Other",
}

OPTIONAL_REVIEW_TYPES = {
    "CookiePolicy",
    "DataProcessingAgreement",
}


# ========== GENERIC HELPERS ==========
def label_from_score(score: float) -> str:
    if score >= THRESHOLDS["compliant"]:
        return "Likely compliant"
    elif score >= THRESHOLDS["partial"]:
        return "Partially compliant"
    else:
        return "Likely non-compliant"


def local_llm_call(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Call a local Ollama model robustly.
    Requires:
      - ollama installed
      - `ollama pull <model>`
      - Ollama daemon running
    """
    if not isinstance(prompt, str):
        prompt = str(prompt)

    cmd = ["ollama", "run", model]
    last_err = ""

    for attempt in range(RETRIES + 1):
        try:
            result = subprocess.run(
                cmd,
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                timeout=TIMEOUT_S,
            )
            out = result.stdout.decode("utf-8", errors="ignore").strip()
            err = result.stderr.decode("utf-8", errors="ignore").strip()
            return out if out else f"(Ollama error: {err})"

        except subprocess.TimeoutExpired:
            last_err = f"timeout after {TIMEOUT_S}s (attempt {attempt + 1}/{RETRIES + 1})"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        time.sleep(1 + attempt * 2)

    return f"(Local LLM error: {last_err})"


def read_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "key" in obj and "value" in obj:
                        cache[obj["key"]] = obj["value"]
                except Exception:
                    continue
    return cache


def append_cache(path: Path, key: str, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def make_key(text: str, related: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(text.encode("utf-8"))
    h.update(related.encode("utf-8"))
    return h.hexdigest()


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def lookup_article_text(gdpr_df: pd.DataFrame, number):
    """
    Return (title, text) for a given Article number if available.
    """
    try:
        mask = (
            (gdpr_df["section_type"].astype(str) == "Article")
            & (gdpr_df["number"].astype(str) == str(number))
        )
        row = gdpr_df.loc[mask].head(1)
        if len(row):
            return str(row.iloc[0].get("title", "")), str(row.iloc[0].get("text", ""))
    except Exception:
        pass
    return "", ""


# ========== RULE-BASED DOCUMENT SIGNALS ==========
def detect_document_signals(text: str) -> dict:
    text_l = (text or "").lower()

    signals = {
        "privacy": [
            "privacy policy", "privacy notice", "personal data", "personal information",
            "legal basis", "lawful basis", "data controller", "data processor",
            "retention", "your rights", "supervisory authority", "contact us",
            "international transfers", "third party", "data subject", "gdpr",
            "processing of personal data", "how we collect", "how we use",
        ],
        "terms": [
            "terms of service", "terms and conditions", "acceptable use",
            "limitation of liability", "governing law", "binding arbitration",
            "prohibited conduct", "termination", "indemnification",
            "by using this service", "you agree to these terms",
        ],
        "cookie_policy": [
            "cookie policy", "types of cookies", "strictly necessary cookies",
            "performance cookies", "analytics cookies", "advertising cookies",
            "browser settings", "cookie preferences",
        ],
        "cookie_banner": [
            "accept all cookies", "reject all", "manage preferences",
            "cookie settings", "consent preferences", "we use cookies",
            "allow all", "save preferences",
        ],
        "marketing": [
            "buy now", "book a demo", "get started", "free trial",
            "our product", "features", "pricing", "customer stories",
            "sign up today", "trusted by", "grow your business",
        ],
        "faq": [
            "frequently asked questions", "faq", "q:", "a:",
            "how do i", "what happens if",
        ],
        "contract": [
            "agreement", "party", "parties", "hereinafter", "warranty",
            "breach", "obligations", "this contract", "shall", "whereas",
        ],
        "dpa": [
            "data processing agreement", "subprocessor", "controller",
            "processor", "standard contractual clauses", "audit rights",
            "annex", "instructions of the controller",
        ],
        "product_page": [
            "product overview", "key features", "specifications",
            "compare plans", "try now", "pricing plans",
        ],
    }

    counts = {k: sum(1 for term in terms if term in text_l) for k, terms in signals.items()}
    return counts


def looks_like_privacy_policy(text: str) -> bool:
    counts = detect_document_signals(text)
    return counts["privacy"] >= 2


# ========== DOCUMENT CLASSIFICATION PROMPT ==========
def build_document_type_prompt(text: str) -> str:
    return f"""
You are a document classifier.

Classify the text into exactly one of these document types:

- PrivacyPolicy
- PrivacyNotice
- TermsOfService
- CookiePolicy
- CookieBanner
- DataProcessingAgreement
- Contract
- Marketing
- FAQ
- ProductPage
- GenericWebpage
- Other

Definitions:
- PrivacyPolicy / PrivacyNotice: explains personal data collection, purposes, legal basis, sharing, retention, rights, contact details, international transfers, or similar privacy topics.
- TermsOfService: rules for using a service, account obligations, liability, dispute resolution, prohibited conduct.
- CookiePolicy: detailed explanation of cookies or trackers specifically.
- CookieBanner: short consent banner or pop-up text about accepting or managing cookies.
- DataProcessingAgreement: processor/controller obligations, security, subprocessors, audit, SCCs, DPA clauses.
- Contract: legal agreement terms between parties.
- Marketing: promotional or sales content.
- FAQ: question-answer help content.
- ProductPage: describes a product or feature.
- GenericWebpage: navigation, homepage, or general informational page.
- Other: anything else.

Text:
{text}

Return plain text only with exactly these sections:
- DocumentType: <one label from the list above>
- IsPrivacyPolicy: <Yes / No>
- Confidence: <High / Medium / Low>
- Why: <1 short sentence>
- Error: <empty if IsPrivacyPolicy is Yes, otherwise a short error message>
""".strip()


def extract_document_type_check(llm_reply: str) -> dict:
    """
    Parse document-type classification output.
    """
    result = {
        "document_type": None,
        "is_privacy_policy": None,
        "confidence": "",
        "why": "",
        "error": "",
    }

    if not isinstance(llm_reply, str) or not llm_reply.strip():
        result["error"] = "Could not determine document type."
        return result

    for line in llm_reply.splitlines():
        lower = line.lower().strip()

        if lower.startswith("- documenttype:") or lower.startswith("documenttype:"):
            result["document_type"] = line.split(":", 1)[1].strip()

        elif lower.startswith("- isprivacypolicy:") or lower.startswith("isprivacypolicy:"):
            value = line.split(":", 1)[1].strip().lower()
            if value == "yes":
                result["is_privacy_policy"] = True
            elif value == "no":
                result["is_privacy_policy"] = False

        elif lower.startswith("- confidence:") or lower.startswith("confidence:"):
            result["confidence"] = line.split(":", 1)[1].strip()

        elif lower.startswith("- why:") or lower.startswith("why:"):
            result["why"] = line.split(":", 1)[1].strip()

        elif lower.startswith("- error:") or lower.startswith("error:"):
            result["error"] = line.split(":", 1)[1].strip()

    if result["is_privacy_policy"] is False and not result["error"]:
        doc_type = result["document_type"] or "unknown document"
        result["error"] = f"Input text appears to be {doc_type}, not a privacy policy or privacy notice."

    if result["is_privacy_policy"] is True:
        result["error"] = ""

    return result


def should_reject_document_type(doc_info: dict, text: str):
    """
    Returns:
        (reject: bool, reason: str)
    """
    counts = detect_document_signals(text)
    doc_type = doc_info.get("document_type")
    is_privacy = doc_info.get("is_privacy_policy")
    confidence = (doc_info.get("confidence") or "").lower()

    if is_privacy is False:
        return True, doc_info.get("error") or f"Rejected: detected as {doc_type}."

    if doc_type in REJECT_TYPES:
        return True, f"Rejected: detected as {doc_type}."

    if doc_type in OPTIONAL_REVIEW_TYPES:
        return True, f"Rejected: detected as {doc_type}, which is not a privacy policy or privacy notice."

    if doc_type in ALLOWED_POLICY_TYPES and is_privacy is True:
        return False, ""

    if counts["terms"] >= 2 and counts["privacy"] == 0:
        return True, "Rejected: text looks like Terms of Service, not a privacy policy."

    if counts["cookie_banner"] >= 2 and counts["privacy"] <= 1:
        return True, "Rejected: text looks like a cookie banner, not a privacy policy."

    if counts["cookie_policy"] >= 2 and counts["privacy"] <= 1:
        return True, "Rejected: text looks like a cookie policy, not a full privacy policy."

    if counts["marketing"] >= 2 and counts["privacy"] == 0:
        return True, "Rejected: text looks like marketing content, not a privacy policy."

    if counts["faq"] >= 2 and counts["privacy"] == 0:
        return True, "Rejected: text looks like FAQ content, not a privacy policy."

    if counts["product_page"] >= 2 and counts["privacy"] == 0:
        return True, "Rejected: text looks like a product page, not a privacy policy."

    if counts["dpa"] >= 2 and counts["privacy"] <= 1:
        return True, "Rejected: text looks like a data processing agreement, not a privacy policy."

    if is_privacy is None or confidence == "low":
        if counts["privacy"] < 2:
            return True, "Rejected: could not confidently identify this text as a privacy policy."

    return True, "Rejected: document is not clearly a privacy policy or privacy notice."


# ========== COMPLIANCE PROMPT ==========
def build_prompt_plain(paragraph: str, best_no: str, art_title: str, art_text: str) -> str:
    law_snippet = (art_text[:1200] + "...") if art_text else ""
    article_label = f"Article {best_no} — {art_title}".strip(" —")

    return f"""
You are a GDPR compliance expert. Assess the paragraph’s compliance.

Paragraph (truncated):
{paragraph}

Relevant GDPR context:
{article_label}
{law_snippet}

Return a short, plain-text note with exactly these sections:
- Article: <number and title>
- Verdict: <Compliant / Partially compliant / Non-compliant>
- Why: <1–2 concise sentences>
- Fix: <1 concrete action if not fully compliant>
""".strip()


def extract_llm_verdict(llm_reply: str) -> str:
    if not isinstance(llm_reply, str) or not llm_reply.strip():
        return "Unknown"

    verdict_line = None
    for line in llm_reply.splitlines():
        if "verdict" in line.lower():
            verdict_line = line.strip()
            break

    if not verdict_line:
        return "Unknown"

    v = verdict_line.lower()

    if "non" in v and "compliant" in v:
        return "Non-compliant"
    if "part" in v and "compliant" in v:
        return "Partially compliant"
    if "compliant" in v:
        return "Compliant"

    return "Unknown"


def verdict_to_score(verdict: str) -> float:
    v = (verdict or "").strip().lower()
    if v == "compliant":
        return 1.0
    if v == "partially compliant":
        return 0.5
    if v == "non-compliant":
        return 0.0
    return 0.5


# ========== ROW BUILDERS ==========
def build_error_row(row, idx, raw_text, best_no, doc_info, doc_reply, error_message):
    return {
        "id": row.get("id", idx),
        "source": row.get("source", ""),
        "policy_text": raw_text[:300] + ("..." if len(raw_text) > 300 else ""),
        "best_article_number": best_no,

        "document_type": doc_info.get("document_type"),
        "is_privacy_policy": False,
        "document_confidence": doc_info.get("confidence", ""),
        "policy_check_reason": doc_info.get("why", ""),
        "policy_check_raw": doc_reply,
        "error_message": error_message,

        "heuristic_score": None,
        "semantic_score": None,
        "llm_verdict": None,
        "llm_score": None,
        "llm_assessment": None,

        "combined_score": None,
        "combined_label": "Error",
    }


# ========== MAIN ==========
def main():
    wsum = HEURISTIC_WEIGHT + SEMANTIC_WEIGHT + LLM_WEIGHT
    if abs(wsum - 1.0) > 1e-9:
        raise ValueError(f"Weights must sum to 1.0. Current sum={wsum}")

    df = pd.read_csv(POLICY_FILE)
    gdpr_df = pd.read_csv(GDPR_FILE)
    agent = GDPRComplianceAgent(gdpr_articles_df=gdpr_df)

    # ---- TEST FILTER: first 5 policies/docs ----
    if "source" in df.columns:
        first5 = df["source"].dropna().unique()[:30]
        df = df[df["source"].isin(first5)].copy()
        print("Testing sources:", first5, "→ rows:", len(df))
    else:
        df["doc"] = df["id"].astype(str).str.split("_", n=1).str[0]
        first5 = df["doc"].dropna().unique()[:5]
        df = df[df["doc"].isin(first5)].copy()
        print("Testing docs:", first5, "→ rows:", len(df))

    total = len(df)
    print(f"Using Ollama model: {OLLAMA_MODEL}")
    print(f"Evaluating {total} paragraphs...")

    # Prewarm model
    _ = local_llm_call(PREWARM_PROMPT)

    cache = read_cache(CACHE_FILE)
    combined_rows = []

    for idx, row in df.iterrows():
        raw_text = safe_str(row.get("policy_text", "")).strip()
        text = raw_text[:MAX_CHARS]
        best_no = safe_str(row.get("best_article_number", "")).strip()

        # ---------- STEP 0: basic empty/short rejection ----------
        if not raw_text:
            doc_info = {
                "document_type": "Empty",
                "confidence": "High",
                "why": "No text was provided.",
            }
            combined_rows.append(
                build_error_row(
                    row=row,
                    idx=idx,
                    raw_text=raw_text,
                    best_no=best_no,
                    doc_info=doc_info,
                    doc_reply="",
                    error_message="Rejected: empty input text.",
                )
            )
            processed = len(combined_rows)
            if processed % 5 == 0 or processed == total:
                print(f"   → Processed {processed}/{total}")
            continue

        if len(raw_text.strip()) < MIN_POLICY_CHARS:
            doc_info = {
                "document_type": "TooShort",
                "confidence": "High",
                "why": "Text is too short to reliably assess as a privacy policy.",
            }
            combined_rows.append(
                build_error_row(
                    row=row,
                    idx=idx,
                    raw_text=raw_text,
                    best_no=best_no,
                    doc_info=doc_info,
                    doc_reply="",
                    error_message="Rejected: text is too short to classify as a privacy policy.",
                )
            )
            processed = len(combined_rows)
            if processed % 5 == 0 or processed == total:
                print(f"   → Processed {processed}/{total}")
            continue

        # ---------- STEP 1: rule-based signals ----------
        rule_policy_guess = looks_like_privacy_policy(text)
        signal_counts = detect_document_signals(text)

        # ---------- STEP 2: LLM document classification ----------
        doc_prompt = build_document_type_prompt(text)
        doc_related = f"document_type_check_v1|rule_guess={rule_policy_guess}|signals={json.dumps(signal_counts, sort_keys=True)}"
        doc_ckey = make_key(text, doc_related, OLLAMA_MODEL)

        if doc_ckey in cache:
            doc_reply = cache[doc_ckey]
        else:
            doc_reply = local_llm_call(doc_prompt)
            append_cache(CACHE_FILE, doc_ckey, doc_reply)
            cache[doc_ckey] = doc_reply

        doc_info = extract_document_type_check(doc_reply)
        reject_doc, reject_reason = should_reject_document_type(doc_info, text)

        if reject_doc:
            combined_rows.append(
                build_error_row(
                    row=row,
                    idx=idx,
                    raw_text=raw_text,
                    best_no=best_no,
                    doc_info=doc_info,
                    doc_reply=doc_reply,
                    error_message=reject_reason,
                )
            )
            processed = len(combined_rows)
            if processed % 5 == 0 or processed == total:
                print(f"   → Processed {processed}/{total}")
            continue

        # ---------- STEP 3: heuristic GDPR scoring ----------
        try:
            report = agent.evaluate_policy(raw_text)
            heuristic_score = (
                float(agent.overall_score(report))
                if report is not None and not report.empty
                else 0.0
            )
        except Exception:
            heuristic_score = 0.0

        # ---------- STEP 4: semantic score ----------
        try:
            semantic_score = float(row.get("best_similarity", 0.0))
        except Exception:
            semantic_score = 0.0

        # ---------- STEP 5: article grounding ----------
        art_title, art_text = ("", "")
        if best_no:
            art_title, art_text = lookup_article_text(gdpr_df, best_no)

        # ---------- STEP 6: LLM compliance reasoning ----------
        prompt = build_prompt_plain(text, best_no, art_title, art_text)
        ckey = make_key(text, (best_no + art_title + art_text[:500]), OLLAMA_MODEL)

        if ckey in cache:
            llm_reply = cache[ckey]
        else:
            llm_reply = local_llm_call(prompt)
            append_cache(CACHE_FILE, ckey, llm_reply)
            cache[ckey] = llm_reply

        llm_verdict = extract_llm_verdict(llm_reply)
        llm_score = verdict_to_score(llm_verdict)

        # ---------- STEP 7: combine ----------
        combined_score = (
            HEURISTIC_WEIGHT * heuristic_score
            + SEMANTIC_WEIGHT * semantic_score
            + LLM_WEIGHT * llm_score
        )
        combined_label = label_from_score(combined_score)

        combined_rows.append({
            "id": row.get("id", idx),
            "source": row.get("source", ""),
            "policy_text": raw_text[:300] + ("..." if len(raw_text) > 300 else ""),
            "best_article_number": best_no,

            "document_type": doc_info.get("document_type"),
            "is_privacy_policy": True,
            "document_confidence": doc_info.get("confidence", ""),
            "policy_check_reason": doc_info.get("why", ""),
            "policy_check_raw": doc_reply,
            "error_message": "",

            "heuristic_score": round(heuristic_score, 3),
            "semantic_score": round(semantic_score, 3),
            "llm_verdict": llm_verdict,
            "llm_score": round(llm_score, 3),
            "llm_assessment": llm_reply,

            "combined_score": round(combined_score, 3),
            "combined_label": combined_label,
        })

        processed = len(combined_rows)
        if processed % 5 == 0 or processed == total:
            print(f"   → Processed {processed}/{total}")

    out = pd.DataFrame(combined_rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\nSaved: {OUTPUT_FILE}")
    print(out.head(5))


if __name__ == "__main__":
    main()