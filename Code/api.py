import time

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.compliance_service import assess_policy_text as assess_hybrid
from src.compliance_service_llm_only import assess_policy_text as assess_llm_only

app = Flask(__name__)
CORS(app)


@app.route("/api/check-compliance", methods=["POST"])
def check_compliance():
    try:
        data = request.get_json(force=True)

        policy_text = data.get("policy_text", "")
        mode = data.get("mode", "hybrid")

        start = time.perf_counter()

        if mode == "llm_only":
            result = assess_llm_only(policy_text)
        else:
            result = assess_hybrid(policy_text)

        processing_time = round(time.perf_counter() - start, 2)

        result["selected_mode"] = mode
        result["processing_time_seconds"] = processing_time

        print(f"Processing time: {processing_time}s")

        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({
            "error": "Compliance check failed.",
            "detail": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)