# src/gdpr_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===============================================================
#  GDPR CHECKLIST
# ===============================================================
@dataclass
class GDPRCheck:
    key: str
    title: str
    query: str
    article_refs: List[str]
    heuristics: Dict[str, Any]
    recommendation: str

CHECKLIST: List[GDPRCheck] = [
    GDPRCheck(
        key="controller_identity",
        title="Controller identity & contact",
        query="data controller identity name address contact email",
        article_refs=["Art. 13(1)(a)", "Art. 14(1)(a)"],
        heuristics={"must_contain": ["controller", "contact", "email|telephone|phone"]},
        recommendation="State who the controller is and provide a working email or phone contact."
    ),
    GDPRCheck(
        key="dpo_contact",
        title="Data Protection Officer (DPO) contact",
        query="data protection officer contact details dpo",
        article_refs=["Art. 37–39", "Art. 13(1)(b)"],
        heuristics={"should_contain_one": ["data protection officer", r"\bDPO\b"]},
        recommendation="If a DPO exists, identify the role and publish a contact address or form."
    ),
    GDPRCheck(
        key="purposes_legal_basis",
        title="Purposes & legal bases",
        query="purpose of processing legal basis lawful basis consent legitimate interest contract legal obligation",
        article_refs=["Art. 5(1)(b)", "Art. 6"],
        heuristics={"must_contain": ["purpose|purposes", "legal basis|lawful basis|consent|legitimate interest|contract|legal obligation"]},
        recommendation="List each processing purpose with its corresponding legal basis."
    ),
    GDPRCheck(
        key="data_categories",
        title="Categories of personal data",
        query="categories of personal data data we collect identifiers contact data usage data cookies",
        article_refs=["Art. 14(1)(d)", "Recital 39"],
        heuristics={"must_contain": ["data we collect|categories|personal data"]},
        recommendation="Describe the categories of data collected (identifiers, contact, usage, etc.)."
    ),
    GDPRCheck(
        key="retention",
        title="Retention periods or criteria",
        query="data retention storage period how long we keep data deletion schedule",
        article_refs=["Art. 5(1)(e)", "Art. 13(2)(a)"],
        heuristics={"must_contain": ["retain|retention|how long|storage period|keep"]},
        recommendation="Specify how long data is kept or the criteria used for retention."
    ),
    GDPRCheck(
        key="recipients",
        title="Recipients / processors",
        query="recipients processors service providers sharing with third parties sub-processors",
        article_refs=["Art. 13(1)(e)", "Art. 28"],
        heuristics={"should_contain_one": ["processor|service provider", "third part(y|ies)|vendors|partners"]},
        recommendation="Identify categories of recipients and whether they act as processors or controllers."
    ),
    GDPRCheck(
        key="international_transfers",
        title="International data transfers & safeguards",
        query="international transfers outside EEA EU UK standard contractual clauses adequacy",
        article_refs=["Art. 44–49"],
        heuristics={"should_contain_one": ["standard contractual clauses|SCC", "adequacy|adequate", "EEA|EU|UK|outside"]},
        recommendation="Explain if data leaves the EEA/UK and the safeguard used (Adequacy, SCCs, BCRs)."
    ),
    GDPRCheck(
        key="rights",
        title="Data subject rights",
        query="rights access rectification erasure restriction objection portability how to exercise request",
        article_refs=["Art. 15–22"],
        heuristics={"must_contain": ["right|rights", "access|rectification|erasure|deletion|portability|object|restriction"]},
        recommendation="List rights (access, rectification, erasure, restriction, portability, objection) and how to request them."
    ),
    GDPRCheck(
        key="withdraw_consent",
        title="Right to withdraw consent",
        query="withdraw consent at any time opt out revoke",
        article_refs=["Art. 7(3)"],
        heuristics={"should_contain_one": ["withdraw|revoke", "consent"]},
        recommendation="Explain that consent can be withdrawn anytime and provide an opt-out mechanism."
    ),
    GDPRCheck(
        key="complaint_authority",
        title="Right to complain to supervisory authority",
        query="complaint supervisory authority data protection authority",
        article_refs=["Art. 13(2)(d)"],
        heuristics={"must_contain": ["complaint", "authority|supervisory"]},
        recommendation="Name the authority and clarify how a complaint can be lodged."
    ),
    GDPRCheck(
        key="automated_decisions",
        title="Automated decision-making / profiling",
        query="automated decision making profiling meaningful information logic significance consequences",
        article_refs=["Art. 22"],
        heuristics={"should_contain_one": ["automated decision|profiling|inferences"]},
        recommendation="State if automated decisions or profiling occur and their consequences."
    ),
    GDPRCheck(
        key="cookies_tracking",
        title="Cookies / tracking & consent",
        query="cookies tracking analytics advertising consent preferences banner",
        article_refs=["ePrivacy", "Art. 6(1)(a)"],
        heuristics={"should_contain_one": ["cookies|cookie", "consent|preferences|opt"]},
        recommendation="Explain cookie categories, purposes, and consent/opt-out options."
    ),
]

# ===============================================================
#  INTERNAL HELPERS
# ===============================================================
def _sentences(text: str) -> List[str]:
    chunks = re.split(r'(?<=[.!?])\s+', text.strip())
    return [c for c in chunks if c]

def _build_retriever(corpus: List[str]) -> Tuple[TfidfVectorizer, Any]:
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X = vec.fit_transform(corpus)
    return vec, X

def _top_k(query: str, vec: TfidfVectorizer, X, corpus: List[str], k: int = 5):
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X)[0]
    idx = sims.argsort()[::-1][:k]
    return [(corpus[i], float(sims[i])) for i in idx]

def _passes_heuristics(text: str, heuristics: Dict[str, Any]):
    t = text.lower()
    missing = []

    def any_match(patterns):
        return any(re.search(p, t, re.I) for p in patterns)

    if "must_contain" in heuristics:
        pats = heuristics["must_contain"]
        for p in pats:
            if not re.search(p, t, re.I):
                missing.append(p)
    if "should_contain_one" in heuristics:
        pats = heuristics["should_contain_one"]
        if not any_match(pats):
            missing.append(" OR ".join(pats))

    if not heuristics:
        return "pass", []
    status = "pass" if not missing else ("partial" if len(missing) <= 1 else "fail")
    return status, missing

# ===============================================================
#  MAIN AGENT
# ===============================================================
class GDPRComplianceAgent:
    def __init__(self, gdpr_articles_df: pd.DataFrame | None = None):
        self.gdpr = gdpr_articles_df

    def evaluate_policy(self, text: str) -> pd.DataFrame:
        # --- guard clauses for empties ---
        if not text or not text.strip():
            return pd.DataFrame()
        sents = _sentences(text)
        if not sents:
            return pd.DataFrame()

        vec, X = _build_retriever(sents)
        rows = []
        for chk in CHECKLIST:
            candidates = _top_k(chk.query, vec, X, sents, k=5)
            evidence = candidates[0][0] if candidates else ""
            status, missing = _passes_heuristics(evidence, chk.heuristics)
            rows.append({
                "check": chk.key,
                "title": chk.title,
                "status": status,
                "evidence": evidence[:600],
                "missing": missing,
                "gdpr_articles": ", ".join(chk.article_refs),
                "recommendation": chk.recommendation
            })
        df = pd.DataFrame(rows)
        df["score"] = df["status"].map({"pass":1.0,"partial":0.5,"fail":0.0})
        return df

    def overall_score(self, report_df: pd.DataFrame) -> float:
        return float(report_df["score"].mean()) if not report_df.empty else 0.0