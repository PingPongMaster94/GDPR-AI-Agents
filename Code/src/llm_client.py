import os
import subprocess
import time


class LLMClientError(RuntimeError):
    """Raised when the configured LLM provider cannot return a valid response."""


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "420"))
OLLAMA_MAX_ATTEMPTS = int(os.getenv("OLLAMA_MAX_ATTEMPTS", "2"))

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MAX_ATTEMPTS = int(os.getenv("GEMINI_MAX_ATTEMPTS", "2"))
GEMINI_RETRY_DELAY_S = float(os.getenv("GEMINI_RETRY_DELAY_S", "5"))

GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",")
    if model.strip()
]


def call_llm(prompt: str) -> str:
    if not isinstance(prompt, str):
        prompt = str(prompt)

    if not prompt.strip():
        raise LLMClientError("The model prompt was empty.")

    print(f"DEBUG: LLM_PROVIDER = {LLM_PROVIDER}")

    if LLM_PROVIDER == "gemini":
        return call_gemini(prompt)

    if LLM_PROVIDER == "ollama":
        return call_ollama(prompt)

    raise LLMClientError(
        f"Unsupported LLM provider: {LLM_PROVIDER}. "
        "Use 'gemini' or 'ollama'."
    )


def call_ollama(prompt: str) -> str:
    cmd = ["ollama", "run", OLLAMA_MODEL]
    last_error = "Unknown Ollama error."

    for attempt in range(1, OLLAMA_MAX_ATTEMPTS + 1):
        try:
            print(
                f"DEBUG: Ollama attempt {attempt}/"
                f"{OLLAMA_MAX_ATTEMPTS} using {OLLAMA_MODEL}"
            )

            result = subprocess.run(
                cmd,
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                timeout=OLLAMA_TIMEOUT_S,
            )

            output = result.stdout.decode(
                "utf-8",
                errors="ignore",
            ).strip()

            error_output = result.stderr.decode(
                "utf-8",
                errors="ignore",
            ).strip()

            if result.returncode == 0 and output:
                return output

            last_error = error_output or (
                f"Ollama exited with status {result.returncode}."
            )

        except subprocess.TimeoutExpired:
            last_error = (
                f"Ollama timed out after {OLLAMA_TIMEOUT_S} seconds."
            )

        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"

        if attempt < OLLAMA_MAX_ATTEMPTS:
            time.sleep(2 * attempt)

    raise LLMClientError(
        f"Ollama could not complete the analysis: {last_error}"
    )


def is_retryable_gemini_error(error: Exception) -> bool:
    error_text = str(error).upper()

    retryable_markers = (
        "429",
        "RESOURCE_EXHAUSTED",
        "503",
        "UNAVAILABLE",
        "504",
        "DEADLINE_EXCEEDED",
        "TIMEOUT",
        "CONNECTION",
        "TEMPORARILY",
        "HIGH DEMAND",
    )

    return any(
        marker in error_text
        for marker in retryable_markers
    )


def call_gemini(prompt: str) -> str:
    try:
        from google import genai
    except Exception as error:
        raise LLMClientError(
            "The Gemini client is not installed. "
            "Install it with: pip install google-genai. "
            f"Original error: {type(error).__name__}: {error}"
        ) from error

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not api_key:
        raise LLMClientError(
            "Missing GEMINI_API_KEY or GOOGLE_API_KEY "
            "environment variable."
        )

    client = genai.Client(api_key=api_key)

    models_to_try = list(
        dict.fromkeys(
            [GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]
        )
    )

    last_error: Exception | None = None

    for model_name in models_to_try:
        for attempt in range(1, GEMINI_MAX_ATTEMPTS + 1):
            try:
                print(
                    f"DEBUG: Trying Gemini model {model_name}, "
                    f"attempt {attempt}/{GEMINI_MAX_ATTEMPTS}"
                )

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.1,
                    },
                )

                response_text = (
                    response.text or ""
                ).strip()

                if not response_text:
                    raise LLMClientError(
                        f"Gemini model {model_name} returned "
                        "an empty response."
                    )

                print(
                    f"DEBUG: Success with model: {model_name}"
                )

                return response_text

            except Exception as error:
                last_error = error

                print(
                    f"DEBUG: Gemini model {model_name} failed "
                    f"on attempt {attempt}: {error}"
                )

                retryable = is_retryable_gemini_error(
                    error
                )

                if (
                    not retryable
                    or attempt >= GEMINI_MAX_ATTEMPTS
                ):
                    break

                delay = GEMINI_RETRY_DELAY_S * attempt

                print(
                    f"DEBUG: Retrying Gemini in {delay:.1f}s"
                )

                time.sleep(delay)

    error_message = (
        str(last_error)
        if last_error
        else "Unknown Gemini error."
    )

    raise LLMClientError(
        "Gemini could not complete the analysis after "
        f"trying: {', '.join(models_to_try)}. "
        f"Last error: {error_message}"
    )