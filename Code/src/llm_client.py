import os
import subprocess
import time


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "420"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def call_llm(prompt: str) -> str:
    print(f"DEBUG: LLM_PROVIDER = {LLM_PROVIDER}")

    if LLM_PROVIDER == "gemini":
        return call_gemini(prompt)

    return call_ollama(prompt)


def call_ollama(prompt: str) -> str:
    if not isinstance(prompt, str):
        prompt = str(prompt)

    cmd = ["ollama", "run", OLLAMA_MODEL]
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
            last_err = f"timeout after {OLLAMA_TIMEOUT_S}s"

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        time.sleep(1 + attempt * 2)

    return f"(Local LLM error: {last_err})"


def call_gemini(prompt: str) -> str:
    try:
        from google import genai
    except Exception as e:
        return (
            "Gemini client is not installed. "
            "Install it with: pip install google-genai. "
            f"Original error: {type(e).__name__}: {e}"
        )

    try:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            return "(Gemini error: missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable)"

        client = genai.Client(api_key=api_key)

        models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ]

        last_error = None

        for model_name in models_to_try:
            try:
                print(f"DEBUG: Trying Gemini model: {model_name}")

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.1,
                    },
                )

                print(f"DEBUG: Success with model: {model_name}")

                return response.text or ""

            except Exception as model_error:
                print(f"DEBUG: Failed model {model_name}: {model_error}")
                last_error = model_error

        return f"(Gemini error: {type(last_error).__name__}: {last_error})"

    except Exception as e:
        return f"(Gemini error: {type(e).__name__}: {e})"