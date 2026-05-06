# combine_scores_LLM.py
# Heuristic + semantic + local LLM (Ollama) reasoning with caching and plain-text output

import os
import time
import json
import hashlib
import subprocess
import pandas as pd
from pathlib import Path

from gdpr_agent import GDPRComplianceAgent

# ========== PATHS (RELATIVE TO PROJECT ROOT) ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_FILE = PROJECT_ROOT / "data/annotated/paragraphs_with_articles.csv"
GDPR_FILE   = PROJECT_ROOT / "data/reference_law_articles.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/annotated/combined_compliance_scores.csv"
CACHE_FILE  = PROJECT_ROOT / "data/annotated/llm_assessment_cache.jsonl"  # JSONL cache

# ========== SCORING PARAMS ==========
# You requested: heuristic 0.25, semantic 0.25, LLM 0.50
HEURISTIC_WEIGHT = 0.25
SEMANTIC_WEIGHT  = 0.25
LLM_WEIGHT       = 0.50

THRESHOLDS = {"compliant": 0.55, "partial": 0.30}

# ========== OLLAMA / RUNTIME PARAMS ==========
# Change via env: export OLLAMA_MODEL=mistral
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
MAX_CHARS = 800          # truncate paragraph to speed up local inference
TIMEOUT_S = 420          # generous timeout for CPU inference
RETRIES = 2              # retry attempts on failure/timeout
PREWARM_PROMPT = "Say 'ready' once."

# ========== HELPERS ==========
def label_from_score(score: float) -> str:
    if score >= THRESHOLDS["compliant"]:
        return "Likely compliant"
    elif score >= THRESHOLDS["partial"]:
        return "Partially compliant"
    else:
        return "Likely non-compliant"

def local_llm_call(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Call a local Ollama model robustly (bytes I/O, retries, timeout).
    Requires: `ollama pull <model>` and the Ollama daemon running.
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
                timeout=TIMEOUT_S
            )
            out = result.stdout.decode("utf-8", errors="ignore").strip()
            err = result.stderr.decode("utf-8", errors="ignore").strip()
            return out if out else f"(Ollama error: {err})"
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {TIMEOUT_S}s (attempt {attempt+1}/{RETRIES+1})"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1 + attempt * 2)  # simple backoff
    return f"(Local LLM error: {last_err})"

def read_cache(path: Path) -> dict:
    """Read a JSONL cache into a dict keyed by hash."""
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
    """Append a single JSONL record to cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

def make_key(text: str, related: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(text.encode("utf-8"))
    h.update(related.encode("utf-8"))
    return h.hexdigest()

def lookup_article_text(gdpr_df: pd.DataFrame, number):
    """Return (title, text) for a given Article number if available."""
    try:
        mask = (gdpr_df["section_type"] == "Article") & (gdpr_df["number"].astype(str) == str(number))
        row = gdpr_df.loc[mask].head(1)
        if len(row):
            return str(row.iloc[0].get("title", "")), str(row.iloc[0].get("text", ""))
    except Exception:
        pass
    return "", ""

def build_prompt_plain(paragraph: str, best_no: str, art_title: str, art_text: str) -> str:
    """
    Compact, grounded prompt that asks for plain-text (no JSON) with fixed sections.
    """
    law_snippet = (art_text[:1200] + "...") if art_text else ""
    return f"""
You are a GDPR compliance expert. Assess the paragraph’s compliance.

Paragraph (truncated):
{paragraph}

Relevant GDPR context:
Article {best_no} — {art_title}
{law_snippet}

Return a short, plain-text note (no JSON, no braces) with exactly these sections:
- Article: <number and title>
- Verdict: <Compliant / Partially compliant / Non-compliant>
- Why: <1–2 concise sentences>
- Fix: <1 concrete action if not fully compliant>
""".strip()

def extract_llm_verdict(llm_reply: str) -> str:
    """
    Extract the verdict label from the LLM plain-text reply.
    Returns one of: 'Compliant', 'Partially compliant', 'Non-compliant', or 'Unknown'
    """
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

    # Handle "Non-compliant" first (contains "compliant")
    if "non" in v and "compliant" in v:
        return "Non-compliant"
    if "part" in v and "compliant" in v:
        return "Partially compliant"
    if "compliant" in v:
        return "Compliant"

    return "Unknown"

def verdict_to_score(verdict: str) -> float:
    """
    Map verdict to ordinal score.
    """
    v = (verdict or "").strip().lower()
    if v == "compliant":
        return 1.0
    if v == "partially compliant":
        return 0.5
    if v == "non-compliant":
        return 0.0
    # If unclear, treat as neutral/partial (safer than defaulting to compliant)
    return 0.5

# ========== MAIN ==========
def main():
    # Sanity: weights should sum to 1.0
    wsum = HEURISTIC_WEIGHT + SEMANTIC_WEIGHT + LLM_WEIGHT
    if abs(wsum - 1.0) > 1e-9:
        raise ValueError(f"Weights must sum to 1.0. Current sum={wsum}")

    # Load data
    df = pd.read_csv(POLICY_FILE)
    gdpr_df = pd.read_csv(GDPR_FILE)
    agent = GDPRComplianceAgent(gdpr_articles_df=gdpr_df)

    # ---- TEST FILTER: only the first 5 documents (policies) ----
    if "source" in df.columns:
        first5 = df["source"].dropna().unique()[:]  # [:] meaning "all"
        df = df[df["source"].isin(first5)].copy()
        print("Testing sources:", first5, "→ rows:", len(df))
    else:
        # Derive doc from id prefix if 'source' missing
        df["doc"] = df["id"].astype(str).str.split("_", n=1).str[0]
        first5 = df["doc"].dropna().unique()[:5]
        df = df[df["doc"].isin(first5)].copy()
        print("Testing docs:", first5, "→ rows:", len(df))

    total = len(df)
    print(f"Using Ollama model: {OLLAMA_MODEL}")
    print(f"🔍 Evaluating {total} paragraphs...")

    # Pre-warm model once to reduce first-call latency
    _ = local_llm_call(PREWARM_PROMPT)

    # Prepare cache
    cache = read_cache(CACHE_FILE)

    combined_rows = []

    for idx, row in df.iterrows():
        raw_text = str(row.get("policy_text", ""))
        text = raw_text[:MAX_CHARS]  # truncate for speed
        best_no = str(row.get("best_article_number", "")).strip()

        # Heuristic evaluation (run on full raw_text)
        report = agent.evaluate_policy(raw_text)
        heuristic_score = float(agent.overall_score(report)) if report is not None and not report.empty else 0.0

        # Semantic score
        try:
            semantic_score = float(row.get("best_similarity", 0.0))
        except Exception:
            semantic_score = 0.0

        # Look up best article title/text for grounding
        art_title, art_text = ("", "")
        if best_no:
            art_title, art_text = lookup_article_text(gdpr_df, best_no)

        # Build plain-text prompt
        prompt = build_prompt_plain(text, best_no, art_title, art_text)

        # Cache key (prompt + model + a bit of law text to avoid collisions)
        ckey = make_key(text, (best_no + art_title + art_text[:500]), OLLAMA_MODEL)

        # LLM reasoning (with cache)
        if ckey in cache:
            llm_reply = cache[ckey]
        else:
            llm_reply = local_llm_call(prompt)
            append_cache(CACHE_FILE, ckey, llm_reply)
            cache[ckey] = llm_reply

        # LLM verdict -> numeric score
        llm_verdict = extract_llm_verdict(llm_reply)
        llm_score = verdict_to_score(llm_verdict)

        # Combine scores (NOW includes LLM)
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

    # Save
    out = pd.DataFrame(combined_rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n✅ Saved: {OUTPUT_FILE}")
    print(out.head(5))

if __name__ == "__main__":
    main()