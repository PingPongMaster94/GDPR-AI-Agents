export type AnalysisMode = "llm_only" | "hybrid";

export type ComplianceStatus =
  | "Compliant"
  | "Partially Compliant"
  | "Non-Compliant";

export type SectionStatus =
  | "present"
  | "weak"
  | "missing";

export type IssueSeverity =
  | "high"
  | "medium"
  | "low";

export interface ComplianceSection {
  name: string;
  status: SectionStatus;
  note: string;
}

export interface ComplianceIssue {
  title: string;
  severity: IssueSeverity;
  description: string;
}

export interface AnalysisMethod {
  title: string;
  description: string;
  limitations: string;
}

export interface PolicyStrength {
  title: string;
  evidence: string;
  gdprRelevance: string;
}

export interface ComplianceResult {
  status: ComplianceStatus;
  score: number;
  summary: string;
  analysedAt: string;
  wordCount: number;
  paragraphCount: number;
  selectedMode: AnalysisMode;
  apiMode: string;
  sections: ComplianceSection[];
  issues: ComplianceIssue[];
  recommendations: string[];
  analysisMethod: AnalysisMethod | null;
  strengths: PolicyStrength[];
  policyWasTruncated: boolean;
}

interface BackendResponse {
  overall_status?: string;
  combined_label?: string;
  combined_score_percent?: number;
  combined_score?: number;
  summary?: string;
  word_count?: number;
  paragraph_count?: number;
  selected_mode?: string;
  api_mode?: string;
  sections?: unknown[];
  issues?: unknown[];
  recommendations?: unknown[];
  analysis_method?: {
    title?: string;
    description?: string;
    limitations?: string;
  };
  strengths?: unknown[];
  policy_was_truncated?: boolean;
  error?: string;
  detail?: string;
  retryable?: boolean;
}

const API_URL =
  import.meta.env.VITE_API_URL ||
  "http://127.0.0.1:5001/api/check-compliance";

function normaliseStatus(value: unknown): ComplianceStatus {
  const status = String(value || "").trim();

  if (status === "Compliant") {
    return "Compliant";
  }

  if (status === "Non-Compliant") {
    return "Non-Compliant";
  }

  return "Partially Compliant";
}

function normaliseSectionStatus(
  value: unknown,
): SectionStatus {
  const status = String(value || "")
    .trim()
    .toLowerCase();

  if (status === "present") {
    return "present";
  }

  if (status === "weak") {
    return "weak";
  }

  return "missing";
}

function normaliseSeverity(
  value: unknown,
): IssueSeverity {
  const severity = String(value || "")
    .trim()
    .toLowerCase();

  if (severity === "high") {
    return "high";
  }

  if (severity === "low") {
    return "low";
  }

  return "medium";
}

function normaliseSections(
  sections: unknown,
): ComplianceSection[] {
  if (!Array.isArray(sections)) {
    return [];
  }

  return sections
    .filter(
      (
        item,
      ): item is Record<string, unknown> =>
        Boolean(item) &&
        typeof item === "object",
    )
    .map((item) => ({
      name: String(
        item.name ||
          "General GDPR requirement",
      ),
      status: normaliseSectionStatus(
        item.status,
      ),
      note: String(item.note || ""),
    }));
}

function normaliseIssues(
  issues: unknown,
): ComplianceIssue[] {
  if (!Array.isArray(issues)) {
    return [];
  }

  return issues
    .filter(
      (
        item,
      ): item is Record<string, unknown> =>
        Boolean(item) &&
        typeof item === "object",
    )
    .map((item) => ({
      title: String(
        item.title ||
          item.policy_section ||
          "GDPR compliance concern",
      ),
      severity: normaliseSeverity(
        item.severity,
      ),
      description: String(
        item.description ||
          item.why ||
          "",
      ),
    }));
}

function normaliseRecommendations(
  recommendations: unknown,
): string[] {
  if (!Array.isArray(recommendations)) {
    return [];
  }

  return recommendations
    .map((item) => {
      if (typeof item === "string") {
        return item.trim();
      }

      if (
        item &&
        typeof item === "object"
      ) {
        const record =
          item as Record<string, unknown>;

        return String(
          record.text ||
            record.recommendation ||
            "",
        ).trim();
      }

      return "";
    })
    .filter(Boolean);
}

function normaliseStrengths(
  strengths: unknown,
): PolicyStrength[] {
  if (!Array.isArray(strengths)) {
    return [];
  }

  return strengths
    .filter(
      (
        item,
      ): item is Record<string, unknown> =>
        Boolean(item) &&
        typeof item === "object",
    )
    .map((item) => ({
      title: String(
        item.title ||
          "Positive GDPR practice",
      ),
      evidence: String(
        item.evidence || "",
      ),
      gdprRelevance: String(
        item.gdpr_relevance ||
          item.gdprRelevance ||
          item.relevance ||
          "",
      ),
    }))
    .filter(
      (item) =>
        item.title.trim().length > 0 &&
        item.evidence.trim().length > 0,
    );
}

function calculateScore(
  response: BackendResponse,
): number {
  if (
    typeof response.combined_score_percent ===
    "number"
  ) {
    return Math.max(
      0,
      Math.min(
        100,
        Math.round(
          response.combined_score_percent,
        ),
      ),
    );
  }

  if (
    typeof response.combined_score ===
    "number"
  ) {
    return Math.max(
      0,
      Math.min(
        100,
        Math.round(
          response.combined_score * 100,
        ),
      ),
    );
  }

  throw new Error(
    "The backend did not return a valid compliance score.",
  );
}

export async function runComplianceCheck(
  policyText: string,
  mode: AnalysisMode = "llm_only",
): Promise<ComplianceResult> {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      policy_text: policyText,
      mode,
    }),
  });

  let payload: BackendResponse;

  try {
    payload =
      (await response.json()) as BackendResponse;
  } catch {
    throw new Error(
      "The backend returned an invalid response.",
    );
  }

  if (!response.ok) {
    const message =
      payload.detail ||
      payload.error ||
      "The compliance analysis could not be completed.";

    throw new Error(message);
  }

  const selectedMode: AnalysisMode =
    payload.selected_mode === "hybrid"
      ? "hybrid"
      : payload.selected_mode ===
          "llm_only"
        ? "llm_only"
        : mode;

  const analysisMethod =
    payload.analysis_method
      ? {
          title: String(
            payload.analysis_method.title ||
              "LLM-only whole-policy review",
          ),
          description: String(
            payload.analysis_method
              .description || "",
          ),
          limitations: String(
            payload.analysis_method
              .limitations || "",
          ),
        }
      : null;

  return {
    status: normaliseStatus(
      payload.overall_status ||
        payload.combined_label,
    ),
    score: calculateScore(payload),
    summary: String(
      payload.summary ||
        "The privacy policy assessment was completed.",
    ),
    analysedAt: new Date().toISOString(),
    wordCount: Number(
      payload.word_count || 0,
    ),
    paragraphCount: Number(
      payload.paragraph_count || 0,
    ),
    selectedMode,
    apiMode: String(
      payload.api_mode || selectedMode,
    ),
    sections: normaliseSections(
      payload.sections,
    ),
    issues: normaliseIssues(
      payload.issues,
    ),
    recommendations:
      normaliseRecommendations(
        payload.recommendations,
      ),
    analysisMethod,
    strengths: normaliseStrengths(
      payload.strengths,
    ),
    policyWasTruncated: Boolean(
      payload.policy_was_truncated,
    ),
  };
}