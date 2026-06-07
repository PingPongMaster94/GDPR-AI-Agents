import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.compliance_service import (
    assess_policy_text as assess_hybrid,
)
from src.compliance_service_llm_only import (
    assess_policy_text as assess_llm_only,
)
from src.llm_client import LLMClientError


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "GDPR Compliance Checker API",
        "status": "available",
    }), 200


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "llm_provider": os.getenv(
            "LLM_PROVIDER",
            "gemini",
        ),
    }), 200


@app.route(
    "/api/check-compliance",
    methods=["POST"],
)
def check_compliance():
    try:
        data = request.get_json(
            force=True,
            silent=False,
        )

        if not isinstance(data, dict):
            return jsonify({
                "error": "Invalid request.",
                "detail": (
                    "The request body must be a JSON object."
                ),
            }), 400

        policy_text = str(
            data.get("policy_text", "")
        ).strip()

        mode = str(
            data.get("mode", "hybrid")
        ).strip().lower()

        if not policy_text:
            return jsonify({
                "error": "No policy text provided.",
                "detail": (
                    "Upload a document or paste a privacy "
                    "policy before running the assessment."
                ),
            }), 400

        if mode not in {
            "hybrid",
            "llm_only",
        }:
            return jsonify({
                "error": "Invalid analysis mode.",
                "detail": (
                    "Mode must be either 'hybrid' "
                    "or 'llm_only'."
                ),
            }), 400

        if mode == "llm_only":
            result = assess_llm_only(policy_text)
        else:
            result = assess_hybrid(policy_text)

        result["selected_mode"] = mode

        return jsonify(result), 200

    except LLMClientError as error:
        app.logger.exception(
            "The LLM provider could not complete the request."
        )

        return jsonify({
            "error": "Analysis temporarily unavailable.",
            "detail": str(error),
            "retryable": True,
        }), 503

    except ValueError as error:
        app.logger.exception(
            "The analysis response was invalid."
        )

        return jsonify({
            "error": "The analysis could not be completed.",
            "detail": str(error),
            "retryable": True,
        }), 502

    except FileNotFoundError as error:
        app.logger.exception(
            "A required backend file was not found."
        )

        return jsonify({
            "error": "Backend configuration error.",
            "detail": str(error),
            "retryable": False,
        }), 500

    except Exception as error:
        app.logger.exception(
            "Unexpected compliance analysis error."
        )

        return jsonify({
            "error": "Compliance check failed.",
            "detail": str(error),
            "retryable": True,
        }), 500


if __name__ == "__main__":
    local_port = int(
        os.getenv("LOCAL_PORT", "5001")
    )

    app.run(
        host="0.0.0.0",
        port=local_port,
    )