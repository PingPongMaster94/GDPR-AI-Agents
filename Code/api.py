from flask import Flask, request, jsonify
from flask_cors import CORS

from src.compliance_service import assess_policy_text

app = Flask(__name__)
CORS(app)


@app.route("/api/check-compliance", methods=["POST"])
def check_compliance():
    try:
        data = request.get_json(force=True)
        policy_text = data.get("policy_text", "")

        result = assess_policy_text(policy_text)

        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({
            "error": "Compliance check failed.",
            "detail": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)