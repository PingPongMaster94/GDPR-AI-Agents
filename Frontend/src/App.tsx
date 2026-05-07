import { useState } from "react";
import "./App.css";

function App() {
  const [policyText, setPolicyText] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function runCheck() {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:5000/api/check-compliance", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ policy_text: policyText }),
      });

      if (!response.ok) {
        throw new Error("Compliance check failed");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Something went wrong while analysing the policy.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="hero">
        <p className="tag">AI-powered privacy policy analysis</p>
        <h1>GDPR Compliance Checker</h1>
        <p className="subtitle">
          Upload or paste a privacy policy to assess its alignment with GDPR requirements.
        </p>
      </section>

      <section className="card">
        <p className="step">STEP 01</p>
        <h2>Submit a Privacy Policy</h2>

        <textarea
          value={policyText}
          onChange={(e) => setPolicyText(e.target.value)}
          placeholder="Paste the full privacy policy text here..."
        />

        <p className="word-count">
          {policyText.trim().split(/\s+/).filter(Boolean).length} words entered
        </p>

        <button onClick={runCheck} disabled={loading || !policyText.trim()}>
          {loading ? "Analysing..." : "Run Compliance Check"}
        </button>

        {error && <p className="error">{error}</p>}
      </section>

      {result && (
        <section className="card results">
          <p className="step">STEP 02 — RESULTS</p>
          <h2>Compliance Check Outcome</h2>

          <div className="summary-box">
            <div>
              <p className="label">Overall Status</p>
              <h3>{result.overall_status}</h3>
              <p>{result.summary}</p>
            </div>

            <div className="score">
              <p className="label">Compliance Score</p>
              <strong>{result.combined_score_percent}%</strong>
            </div>
          </div>

          <h3>Detected Issues</h3>
          {result.issues?.length ? (
            result.issues.map((issue: any, index: number) => (
              <div className="issue" key={index}>
                <strong>{issue.title}</strong>
                <p>{issue.description}</p>
              </div>
            ))
          ) : (
            <p>No major issues detected.</p>
          )}

          <h3>Recommendations</h3>
          {result.recommendations?.map((rec: any, index: number) => (
            <div className="recommendation" key={index}>
              {typeof rec === "string" ? rec : rec.text}
            </div>
          ))}
        </section>
      )}
    </main>
  );
}

export default App;