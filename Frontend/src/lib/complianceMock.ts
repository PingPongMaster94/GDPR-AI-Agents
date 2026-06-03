export type ComplianceStatus = "Compliant" | "Partially Compliant" | "Non-Compliant";

export interface ComplianceIssue {
  severity: "high" | "medium" | "low";
  title: string;
  description: string;
}

export interface ComplianceSection {
  name: string;
  status: "present" | "weak" | "missing";
  note: string;
}

export interface ComplianceResult {
  status: ComplianceStatus;
  score: number;
  summary: string;
  issues: ComplianceIssue[];
  sections: ComplianceSection[];
  recommendations: string[];
  analysedAt: string;
  wordCount: number;
}

const API_URL =
  import.meta.env.VITE_API_URL ||
  "https://gdpr-ai-agents.onrender.com/api/check-compliance";

export async function runComplianceCheck(
  input: string,
  mode: "llm_only" | "hybrid" = "llm_only"
): Promise<ComplianceResult> {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  body: JSON.stringify({
  policy_text: input,
  mode,
}),
  });

  if (!response.ok) {
    throw new Error("Compliance check failed");
  }

  const data = await response.json();

  return {
    status: normaliseStatus(data.overall_status || data.combined_label),
    score: data.combined_score_percent ?? Math.round((data.combined_score || 0) * 100),
    summary: data.summary || "Compliance analysis completed.",
    issues: (data.issues || []).map((issue: any) => ({
      severity: normaliseSeverity(issue.severity),
      title: issue.title || "Potential compliance issue",
      description: cleanAssessment(issue.description || ""),
    })),
   sections: data.sections || buildSectionsFromParagraphs(data.paragraph_results || []),
    recommendations: (data.recommendations || []).map((r: any) =>
      typeof r === "string" ? r : r.text
    ),
    analysedAt: new Date().toISOString(),
    wordCount: data.word_count || input.trim().split(/\s+/).filter(Boolean).length,
  };
}

function normaliseStatus(status: string): ComplianceStatus {
  const s = String(status || "").toLowerCase();

  if (s.includes("partial")) return "Partially Compliant";
  if (s.includes("non")) return "Non-Compliant";
  if (s.includes("compliant")) return "Compliant";

  return "Non-Compliant";
}

function normaliseSeverity(severity: string): "high" | "medium" | "low" {
  const s = String(severity || "").toLowerCase();

  if (s === "high") return "high";
  if (s === "low") return "low";

  return "medium";
}

function buildSectionsFromParagraphs(paragraphs: any[]): ComplianceSection[] {
  return paragraphs.map((p) => ({
    name: `GDPR Article ${p.best_article_number}`,
    status:
      p.combined_score >= 0.55
        ? "present"
        : p.combined_score >= 0.3
        ? "weak"
        : "missing",
    note: p.llm_verdict
      ? `${p.llm_verdict}: ${cleanAssessment(p.llm_assessment)}`
      : `Assessment linked to GDPR Article ${p.best_article_number}.`,
  }));
}

function cleanAssessment(text: string): string {
  return String(text || "")
    .replace(/\x1b\[[0-9;]*[A-Za-z]/g, "")
    .replace(/-{5,}/g, "")
    .trim();
}