#!/usr/bin/env python3
"""
combine_scores_LLM.py

Hybrid scoring: Heuristic (0.25) + Semantic (0.25) + LLM Verdict (0.50)

LLM backend switch (env var):
  - LLM_BACKEND=ollama  (default)
  - LLM_BACKEND=hf

Notes for macOS:
- If you use HF on Mac, quantization (bitsandbytes 4bit/8bit) is usually not supported.
  Leave HF_LOAD_IN_4BIT/HF_LOAD_IN_8BIT = false.
- This script tries to use MPS (Apple Silicon) automatically if available.
"""

import os
import re
import time
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from src.gdpr_agent import GDPRComplianceAgent


# =========================
# PATHS (RELATIVE TO PROJECT ROOT)
# =========================
# Expect this file to live in: <project_root>/src/combine_scores_LLM.py
# PROJECT_ROOT becomes: <project_root>
PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_FILE = PROJECT_ROOT / "data/annotated/paragraphs_with_articles.csv"
GDPR_FILE = PROJECT_ROOT / "data/reference_law_articles.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/results/compliance_results.csv"
CACHE_FILE = PROJECT_ROOT / "data/annotated/llm_assessment_cache.jsonl"


# =========================
# SCORING PARAMS
# =========================
HEURISTIC_WEIGHT = 0.25
SEMANTIC_WEIGHT = 0.25
LLM_WEIGHT = 0.50

THRESHOLDS = {"compliant": 0.55, "partial": 0.30}


# =========================
# EXPERIMENT CONTROLS (OPTIONAL)
# =========================
LIMIT_SOURCES = int(os.getenv("LIMIT_SOURCES", "0"))  # 0 = no limit
LIMIT_ROWS = int(os.getenv("LIMIT_ROWS", "0"))        # 0 = no limit
MAX_CHARS = int(os.getenv("MAX_CHARS", "800"))        # prompt truncation only

# HuggingFace input truncation (token-level)
HF_MAX_INPUT_TOKENS = int(os.getenv("HF_MAX_INPUT_TOKENS", "2048"))


# =========================
# LLM BACKEND SWITCH
# =========================
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()
# Supported: "ollama", "hf"


# ----- Ollama params -----
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "420"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))
OLLAMA_PREWARM_PROMPT = os.getenv("OLLAMA_PREWARM_PROMPT", "Say 'ready' once.")


# ----- HuggingFace params -----
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_REVISION = os.getenv("HF_REVISION", "").strip() or None
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "").strip() or None

HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "220"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.0"))
HF_DO_SAMPLE = os.getenv("HF_DO_SAMPLE", "false").strip().lower() in ("1", "true", "yes")
HF_TOP_P = float(os.getenv("HF_TOP_P", "1.0"))
HF_REPETITION_PENALTY = float(os.getenv("HF_REPETITION_PENALTY", "1.0"))

# Optional: quantization flags (usually NOT supported on macOS)
HF_LOAD_IN_8BIT = os.getenv("HF_LOAD_IN_8BIT", "false").strip().lower() in ("1", "true", "yes")
HF_LOAD_IN_4BIT = os.getenv("HF_LOAD_IN_4BIT", "false").strip().lower() in ("1", "true", "yes")

# device_map:
# - "auto" works well on Linux+CUDA, but on macOS it's safer to let the script pick MPS/CPU.
HF_DEVICE_MAP = os.getenv("HF_DEVICE_MAP", "auto").strip()

# torch dtype: "auto" | "float16" | "bfloat16" | "float32"
HF_TORCH_DTYPE = os.getenv("HF_TORCH_DTYPE", "auto").strip().lower()

# Use chat template (recommended)
HF_USE_CHAT_TEMPLATE = os.getenv("HF_USE_CHAT_TEMPLATE", "true").strip().lower() in ("1", "true", "yes")

# Lazy-initialized HF model objects
_HF_TOKENIZER = None
_HF_MODEL_OBJ = None
_HF_RUNTIME_DEVICE = None  # "cuda" | "mps" | "cpu"


# =========================
# HELPERS
# =========================
def label_from_score(score: float) -> str:
    if score >= THRESHOLDS["compliant"]:
        return "Likely compliant"
    if score >= THRESHOLDS["partial"]:
        return "Partially compliant"
    return "Likely non-compliant"


def preflight() -> None:
    missing = [p for p in [POLICY_FILE, GDPR_FILE] if not p.exists()]
    if missing:
        msg = "\n".join([f"- Missing: {m}" for m in missing])
        raise FileNotFoundError(
            "Preflight failed. Required input files not found:\n"
            f"{msg}\n\n"
            "Check project root + that earlier pipeline steps produced these files."
        )


def read_cache(path: Path) -> Dict[str, str]:
    cache: Dict[str, str] = {}
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


def append_cache(path: Path, key: str, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def make_cache_key(
    backend: str,
    model_name: str,
    prompt_text: str,
    related: str,
    gen_params: Dict[str, str],
) -> str:
    h = hashlib.sha256()
    h.update(backend.encode("utf-8"))
    h.update(model_name.encode("utf-8"))
    for k in sorted(gen_params.keys()):
        h.update(f"{k}={gen_params[k]}".encode("utf-8"))
    h.update(prompt_text.encode("utf-8"))
    h.update(related.encode("utf-8"))
    return h.hexdigest()


def lookup_article_text(gdpr_df: pd.DataFrame, number) -> Tuple[str, str]:
    try:
        mask = (gdpr_df["section_type"] == "Article") & (gdpr_df["number"].astype(str) == str(number))
        row = gdpr_df.loc[mask].head(1)
        if len(row):
            return str(row.iloc[0].get("title", "")), str(row.iloc[0].get("text", ""))
    except Exception:
        pass
    return "", ""


def build_prompt_plain(paragraph: str, best_no: str, art_title: str, art_text: str) -> str:
    law_snippet = (art_text[:1200] + "...") if art_text else ""
    return f"""
You are a GDPR compliance expert. Assess the paragraph’s compliance.

Paragraph (truncated):
{paragraph}

Relevant GDPR context:
Article {best_no} — {art_title}
{law_snippet}

Return a short, plain-text note with exactly these sections:
- Article: <number and title>
- Verdict: <Compliant / Partially compliant / Non-compliant>
- Why: <1–2 concise sentences>
- Fix: <1 concrete action if not fully compliant>
""".strip()


def _normalize_verdict_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def extract_llm_verdict(llm_reply: str) -> str:
    if not isinstance(llm_reply, str) or not llm_reply.strip():
        return "Unknown"

    for line in llm_reply.splitlines():
        if re.search(r"^\s*verdict\s*:", line, flags=re.IGNORECASE):
            v = _normalize_verdict_text(line.split(":", 1)[1])
            if "non" in v and "compliant" in v:
                return "Non-compliant"
            if "partial" in v and "compliant" in v:
                return "Partially compliant"
            if "compliant" in v:
                return "Compliant"
            return "Unknown"

    t = _normalize_verdict_text(llm_reply)
    if re.search(r"\bnon[- ]compliant\b", t):
        return "Non-compliant"
    if re.search(r"\bpartially[- ]compliant\b", t):
        return "Partially compliant"
    if re.search(r"\bcompliant\b", t):
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


# =========================
# LLM CALLERS
# =========================
def llm_call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    if not isinstance(prompt, str):
        prompt = str(prompt)

    cmd = ["ollama", "run", model]
    last_err = ""
    for attempt in range(OLLAMA_RETRIES + 1):
        try:
            result = subprocess.run(
                cmd,
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                timeout=OLLAMA_TIMEOUT_S,
            )
            out = result.stdout.decode("utf-8", errors="ignore").strip()
            err = result.stderr.decode("utf-8", errors="ignore").strip()
            return out if out else f"(Ollama error: {err})"
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {OLLAMA_TIMEOUT_S}s (attempt {attempt+1}/{OLLAMA_RETRIES+1})"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1 + attempt * 2)

    return f"(Local LLM error: {last_err})"


def _torch_dtype_from_env():
    import torch  # type: ignore

    if HF_TORCH_DTYPE == "auto":
        return "auto"
    if HF_TORCH_DTYPE == "float16":
        return torch.float16
    if HF_TORCH_DTYPE == "bfloat16":
        return torch.bfloat16
    if HF_TORCH_DTYPE == "float32":
        return torch.float32
    return "auto"


def _pick_runtime_device():
    """
    Prefer: CUDA > MPS > CPU
    """
    import torch  # type: ignore

    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon (Metal)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _init_hf() -> None:
    global _HF_TOKENIZER, _HF_MODEL_OBJ, _HF_RUNTIME_DEVICE
    if _HF_TOKENIZER is not None and _HF_MODEL_OBJ is not None:
        return

    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError(
            "HuggingFace backend selected, but required packages are missing.\n"
            "Install: pip install transformers accelerate safetensors torch"
        ) from e

    # Guard: bitsandbytes quantization generally doesn't work on macOS
    runtime_device = _pick_runtime_device()
    _HF_RUNTIME_DEVICE = runtime_device
    if runtime_device in ("mps", "cpu") and (HF_LOAD_IN_4BIT or HF_LOAD_IN_8BIT):
        raise ValueError(
            "HF_LOAD_IN_4BIT/HF_LOAD_IN_8BIT is enabled, but quantization is typically not supported on macOS/CPU/MPS.\n"
            "Set HF_LOAD_IN_4BIT=false and HF_LOAD_IN_8BIT=false."
        )

    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

    _HF_TOKENIZER = AutoTokenizer.from_pretrained(
        HF_MODEL,
        revision=HF_REVISION,
        cache_dir=HF_CACHE_DIR,
        use_fast=True,
    )

    load_kwargs: Dict[str, object] = {}

    torch_dtype = _torch_dtype_from_env()
    if torch_dtype != "auto":
        load_kwargs["torch_dtype"] = torch_dtype

    # Only use device_map="auto" when CUDA is present; on MPS/CPU it's safer to load normally then .to(device)
    if HF_DEVICE_MAP.lower() != "auto":
        load_kwargs["device_map"] = HF_DEVICE_MAP
    else:
        if runtime_device == "cuda":
            load_kwargs["device_map"] = "auto"

    if HF_LOAD_IN_4BIT and HF_LOAD_IN_8BIT:
        raise ValueError("Set only one of HF_LOAD_IN_4BIT or HF_LOAD_IN_8BIT.")
    if HF_LOAD_IN_4BIT:
        load_kwargs["load_in_4bit"] = True
    if HF_LOAD_IN_8BIT:
        load_kwargs["load_in_8bit"] = True

    _HF_MODEL_OBJ = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        revision=HF_REVISION,
        cache_dir=HF_CACHE_DIR,
        **load_kwargs,
    )

    # Move model to MPS/CPU if needed (when not using device_map)
    import torch  # type: ignore
    if runtime_device in ("mps", "cpu") and "device_map" not in load_kwargs:
        _HF_MODEL_OBJ = _HF_MODEL_OBJ.to(torch.device(runtime_device))

    if _HF_TOKENIZER.pad_token_id is None and _HF_TOKENIZER.eos_token_id is not None:
        _HF_TOKENIZER.pad_token = _HF_TOKENIZER.eos_token


def _build_hf_prompt(prompt: str) -> str:
    assert _HF_TOKENIZER is not None

    if HF_USE_CHAT_TEMPLATE and hasattr(_HF_TOKENIZER, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a GDPR compliance expert. Be concise and grounded."},
            {"role": "user", "content": prompt},
        ]
        try:
            return _HF_TOKENIZER.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    return prompt


def llm_call_hf(prompt: str) -> str:
    _init_hf()
    assert _HF_TOKENIZER is not None and _HF_MODEL_OBJ is not None

    text_prompt = _build_hf_prompt(prompt)

    inputs = _HF_TOKENIZER(
        text_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=HF_MAX_INPUT_TOKENS,
    )

    # Send inputs to same device as model
    try:
        model_device = next(_HF_MODEL_OBJ.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
    except Exception:
        pass

    gen_kwargs = dict(
        max_new_tokens=HF_MAX_NEW_TOKENS,
        do_sample=HF_DO_SAMPLE,
        temperature=HF_TEMPERATURE,
        top_p=HF_TOP_P,
        repetition_penalty=HF_REPETITION_PENALTY,
        pad_token_id=_HF_TOKENIZER.pad_token_id,
        eos_token_id=_HF_TOKENIZER.eos_token_id,
    )

    outputs = _HF_MODEL_OBJ.generate(**inputs, **gen_kwargs)
    out = _HF_TOKENIZER.decode(outputs[0], skip_special_tokens=True)

    # Try to strip the prompt
    if out.startswith(text_prompt):
        out = out[len(text_prompt):].lstrip()

    return out.strip()


def llm_call(prompt: str) -> Tuple[str, str, Dict[str, str]]:
    if LLM_BACKEND == "ollama":
        reply = llm_call_ollama(prompt, model=OLLAMA_MODEL)
        model_name = OLLAMA_MODEL
        gen_params = {"timeout_s": str(OLLAMA_TIMEOUT_S), "retries": str(OLLAMA_RETRIES)}
        return reply, model_name, gen_params

    if LLM_BACKEND == "hf":
        reply = llm_call_hf(prompt)
        model_name = HF_MODEL
        gen_params = {
            "max_new_tokens": str(HF_MAX_NEW_TOKENS),
            "max_input_tokens": str(HF_MAX_INPUT_TOKENS),
            "do_sample": str(HF_DO_SAMPLE),
            "temperature": str(HF_TEMPERATURE),
            "top_p": str(HF_TOP_P),
            "repetition_penalty": str(HF_REPETITION_PENALTY),
            "use_chat_template": str(HF_USE_CHAT_TEMPLATE),
            "device_map": str(HF_DEVICE_MAP),
            "dtype": str(HF_TORCH_DTYPE),
            "load_in_4bit": str(HF_LOAD_IN_4BIT),
            "load_in_8bit": str(HF_LOAD_IN_8BIT),
        }
        return reply, model_name, gen_params

    raise ValueError(f"Unsupported LLM_BACKEND='{LLM_BACKEND}'. Use 'ollama' or 'hf'.")


# =========================
# MAIN
# =========================
def main() -> None:
    wsum = HEURISTIC_WEIGHT + SEMANTIC_WEIGHT + LLM_WEIGHT
    if abs(wsum - 1.0) > 1e-9:
        raise ValueError(f"Weights must sum to 1.0. Current sum={wsum}")

    preflight()

    df = pd.read_csv(POLICY_FILE)
    gdpr_df = pd.read_csv(GDPR_FILE)
    agent = GDPRComplianceAgent(gdpr_articles_df=gdpr_df)

    if LIMIT_SOURCES > 0 and "source" in df.columns:
        sources = df["source"].dropna().unique()[:LIMIT_SOURCES]
        df = df[df["source"].isin(sources)].copy()
        print(f"Testing first {LIMIT_SOURCES} sources: {list(sources)} → rows: {len(df)}")

    if LIMIT_ROWS > 0:
        df = df.head(LIMIT_ROWS).copy()
        print(f"Testing first {LIMIT_ROWS} rows → rows: {len(df)}")

    total = len(df)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"LLM backend: {LLM_BACKEND}")

    if LLM_BACKEND == "ollama":
        print(f"LLM model: {OLLAMA_MODEL} (Ollama)")
        print(f"Ollama timeout={OLLAMA_TIMEOUT_S}s retries={OLLAMA_RETRIES}")
    else:
        print(f"LLM model: {HF_MODEL} (HuggingFace)")
        print(f"HF max_new_tokens={HF_MAX_NEW_TOKENS} max_input_tokens={HF_MAX_INPUT_TOKENS}")
        print(f"HF do_sample={HF_DO_SAMPLE} temperature={HF_TEMPERATURE} top_p={HF_TOP_P}")
        print(f"HF chat_template={HF_USE_CHAT_TEMPLATE} device_map={HF_DEVICE_MAP} dtype={HF_TORCH_DTYPE}")
        if HF_LOAD_IN_4BIT or HF_LOAD_IN_8BIT:
            print(f"HF quantization: 4bit={HF_LOAD_IN_4BIT} 8bit={HF_LOAD_IN_8BIT}")

    print(f"🔍 Evaluating {total} paragraphs...")

    if LLM_BACKEND == "ollama":
        _ = llm_call_ollama(OLLAMA_PREWARM_PROMPT)

    cache = read_cache(CACHE_FILE)
    combined_rows = []
    processed = 0

    for idx, row in df.iterrows():
        raw_text = str(row.get("policy_text", ""))
        text_for_prompt = raw_text[:MAX_CHARS]
        best_no = str(row.get("best_article_number", "")).strip()

        report = agent.evaluate_policy(raw_text)
        heuristic_score = float(agent.overall_score(report)) if report is not None and not report.empty else 0.0

        try:
            semantic_score = float(row.get("best_similarity", 0.0))
        except Exception:
            semantic_score = 0.0

        art_title, art_text = ("", "")
        if best_no:
            art_title, art_text = lookup_article_text(gdpr_df, best_no)

        prompt = build_prompt_plain(text_for_prompt, best_no, art_title, art_text)
        related = best_no + art_title + (art_text[:500] if isinstance(art_text, str) else "")

        # Cache key includes generation settings
        if LLM_BACKEND == "ollama":
            model_name = OLLAMA_MODEL
            gen_params = {"timeout_s": str(OLLAMA_TIMEOUT_S), "retries": str(OLLAMA_RETRIES)}
        else:
            model_name = HF_MODEL
            gen_params = {
                "max_new_tokens": str(HF_MAX_NEW_TOKENS),
                "max_input_tokens": str(HF_MAX_INPUT_TOKENS),
                "do_sample": str(HF_DO_SAMPLE),
                "temperature": str(HF_TEMPERATURE),
                "top_p": str(HF_TOP_P),
                "repetition_penalty": str(HF_REPETITION_PENALTY),
                "use_chat_template": str(HF_USE_CHAT_TEMPLATE),
                "device_map": str(HF_DEVICE_MAP),
                "dtype": str(HF_TORCH_DTYPE),
                "load_in_4bit": str(HF_LOAD_IN_4BIT),
                "load_in_8bit": str(HF_LOAD_IN_8BIT),
            }

        ckey = make_cache_key(LLM_BACKEND, model_name, prompt, related, gen_params)

        if ckey in cache:
            llm_reply = cache[ckey]
        else:
            llm_reply, _, _ = llm_call(prompt)
            append_cache(CACHE_FILE, ckey, llm_reply)
            cache[ckey] = llm_reply

        llm_verdict = extract_llm_verdict(llm_reply)
        llm_score = verdict_to_score(llm_verdict)

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
            "llm_backend": LLM_BACKEND,
            "llm_model": model_name,
            "llm_verdict": llm_verdict,
            "llm_score": round(llm_score, 3),
            "llm_assessment": llm_reply,
            "combined_score": round(combined_score, 3),
            "combined_label": combined_label,
        })

        processed += 1
        if processed % 25 == 0 or processed == total:
            print(f"   → Processed {processed}/{total}")

    out = pd.DataFrame(combined_rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n✅ Saved: {OUTPUT_FILE}")
    print(out.head(5))


if __name__ == "__main__":
    main()