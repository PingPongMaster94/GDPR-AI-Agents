import os
import re
import time
from typing import Optional


class LLMClientError(RuntimeError):
    """Raised when Gemini cannot return a usable response."""


# ---------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------

# Primary model used for normal assessments.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
).strip()

# Models attempted when the primary model is unavailable.
# Multiple fallback models can be separated by commas.
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.5-flash",
    ).split(",")
    if model.strip()
]

# Number of attempts made for temporary failures such as 503 errors.
GEMINI_MAX_ATTEMPTS = max(
    1,
    int(
        os.getenv(
            "GEMINI_MAX_ATTEMPTS",
            "3",
        )
    ),
)

# Base delay used between retries.
GEMINI_RETRY_DELAY_S = max(
    1.0,
    float(
        os.getenv(
            "GEMINI_RETRY_DELAY_S",
            "6",
        )
    ),
)

# Low temperature improves consistency for structured compliance analysis.
GEMINI_TEMPERATURE = float(
    os.getenv(
        "GEMINI_TEMPERATURE",
        "0.1",
    )
)

# Maximum response length allowed for structured JSON output.
GEMINI_MAX_OUTPUT_TOKENS = max(
    1024,
    int(
        os.getenv(
            "GEMINI_MAX_OUTPUT_TOKENS",
            "8192",
        )
    ),
)


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    """
    Send a prompt to Gemini and return its text response.

    Raises:
        LLMClientError:
            When no configured Gemini model can complete the request.
    """

    if not isinstance(prompt, str):
        prompt = str(prompt)

    prompt = prompt.strip()

    if not prompt:
        raise LLMClientError(
            "The model prompt was empty."
        )

    return call_gemini(prompt)


# ---------------------------------------------------------------------
# Gemini error handling
# ---------------------------------------------------------------------

def classify_gemini_error(
    error: Exception,
) -> str:
    """
    Classify a Gemini API error.

    Returns one of:
        authentication
        model_unavailable
        quota
        retryable
        fatal
    """

    error_text = str(
        error or ""
    ).upper()

    authentication_markers = (
        "401",
        "UNAUTHENTICATED",
        "INVALID API KEY",
        "API KEY NOT VALID",
        "PERMISSION_DENIED",
        "403",
    )

    if any(
        marker in error_text
        for marker in authentication_markers
    ):
        return "authentication"

    model_unavailable_markers = (
        "404",
        "NOT_FOUND",
        "NO LONGER AVAILABLE",
        "MODEL NOT FOUND",
        "NOT SUPPORTED FOR GENERATECONTENT",
        "DEPRECATED",
        "SHUT DOWN",
    )

    if any(
        marker in error_text
        for marker in model_unavailable_markers
    ):
        return "model_unavailable"

    quota_markers = (
        "429",
        "RESOURCE_EXHAUSTED",
        "QUOTA EXCEEDED",
        "RATE LIMIT",
    )

    if any(
        marker in error_text
        for marker in quota_markers
    ):
        return "quota"

    retryable_markers = (
        "500",
        "INTERNAL",
        "502",
        "BAD_GATEWAY",
        "503",
        "UNAVAILABLE",
        "504",
        "DEADLINE_EXCEEDED",
        "TIMEOUT",
        "TIMED OUT",
        "CONNECTION",
        "TEMPORARILY",
        "HIGH DEMAND",
        "SERVER ERROR",
        "EMPTY RESPONSE",
    )

    if any(
        marker in error_text
        for marker in retryable_markers
    ):
        return "retryable"

    return "fatal"


def extract_retry_delay(
    error: Exception,
) -> Optional[float]:
    """
    Extract a suggested retry delay from Gemini error text.

    Supported examples:
        Please retry in 25.9s
        'retryDelay': '54s'
        retry after 12 seconds
    """

    error_text = str(
        error or ""
    )

    patterns = (
        r"retry\s+in\s+([\d.]+)\s*s",
        r"retryDelay['\"]?\s*:\s*['\"]([\d.]+)s",
        r"retry\s+after\s+([\d.]+)\s*seconds?",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            error_text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        try:
            return float(
                match.group(1)
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

    return None


def calculate_retry_delay(
    error: Exception,
    attempt: int,
) -> float:
    """
    Use Gemini's suggested delay when available.

    Otherwise, apply a gradually increasing delay.
    """

    suggested_delay = extract_retry_delay(
        error
    )

    if suggested_delay is not None:
        return min(
            suggested_delay + 1.0,
            90.0,
        )

    return min(
        GEMINI_RETRY_DELAY_S * attempt,
        60.0,
    )


# ---------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------

def call_gemini(
    prompt: str,
) -> str:
    """
    Send the prompt to Gemini.

    Default model order:
        1. gemini-3.5-flash
        2. gemini-2.5-flash

    Temporary errors are retried before moving to the fallback model.
    Quota and model-availability errors move directly to the fallback.
    """

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

    client = genai.Client(
        api_key=api_key
    )

    # dict.fromkeys removes duplicates while preserving order.
    models_to_try = list(
        dict.fromkeys(
            [
                GEMINI_MODEL,
                *GEMINI_FALLBACK_MODELS,
            ]
        )
    )

    if not models_to_try:
        raise LLMClientError(
            "No Gemini model has been configured."
        )

    print(
        "DEBUG: Gemini model order = "
        + " -> ".join(models_to_try)
    )

    recent_errors: list[str] = []

    for model_name in models_to_try:
        for attempt in range(
            1,
            GEMINI_MAX_ATTEMPTS + 1,
        ):
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
                        "temperature": GEMINI_TEMPERATURE,
                        "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
                    },
                )

                response_text = (
                    response.text or ""
                ).strip()

                if not response_text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                print(
                    f"DEBUG: Success with Gemini model: {model_name}"
                )

                return response_text

            except Exception as error:
                error_type = classify_gemini_error(
                    error
                )

                error_summary = (
                    f"{model_name}, attempt {attempt}: "
                    f"{type(error).__name__}: {error}"
                )

                recent_errors.append(
                    error_summary
                )

                print(
                    f"DEBUG: Gemini model {model_name} failed "
                    f"({error_type}) on attempt {attempt}: {error}"
                )

                # Invalid credentials will affect every model.
                if error_type == "authentication":
                    raise LLMClientError(
                        "Gemini authentication failed. "
                        "Check GEMINI_API_KEY in the environment settings. "
                        f"Original error: {error}"
                    ) from error

                # An unavailable/retired model should not be retried.
                # Quota may be different for the fallback model.
                if error_type in {
                    "model_unavailable",
                    "quota",
                    "fatal",
                }:
                    print(
                        "DEBUG: Moving to the next configured "
                        "Gemini model."
                    )
                    break

                # Retry temporary server and network failures.
                if (
                    error_type == "retryable"
                    and attempt < GEMINI_MAX_ATTEMPTS
                ):
                    delay = calculate_retry_delay(
                        error,
                        attempt,
                    )

                    print(
                        f"DEBUG: Retrying Gemini model {model_name} "
                        f"in {delay:.1f}s"
                    )

                    time.sleep(delay)
                    continue

                # Attempts exhausted for this model.
                break

    condensed_errors = " | ".join(
        recent_errors[-6:]
    )

    raise LLMClientError(
        "Gemini could not complete the analysis using "
        "any configured model. "
        f"Models attempted: {', '.join(models_to_try)}. "
        f"Recent errors: {condensed_errors}"
    )