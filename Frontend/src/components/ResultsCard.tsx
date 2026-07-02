import { useState } from "react";
import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Download,
  FileSearch,
  Info,
  Lightbulb,
  ListChecks,
  SearchCheck,
  ShieldCheck,
  Sparkles,
  XCircle,
} from "lucide-react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import type {
  ComplianceResult,
  IssueSeverity,
} from "@/lib/complianceMock";
import { cn } from "@/lib/utils";

import {
  exportReportAsDOCX,
  exportReportAsPDF,
  exportReportAsTXT,
} from "@/lib/exportReport";

const statusConfig = {
  Compliant: {
    icon: CheckCircle2,
    badge:
      "bg-success/10 text-success border-success/30",
    label: "Compliant",
  },
  "Partially Compliant": {
    icon: AlertTriangle,
    badge:
      "bg-warning/15 text-warning-foreground border-warning/40",
    label: "Partially Compliant",
  },
  "Non-Compliant": {
    icon: XCircle,
    badge:
      "bg-destructive/10 text-destructive border-destructive/30",
    label: "Non-Compliant",
  },
} as const;

const coverageConfig = {
  present: {
    dot: "bg-success",
    badge:
      "border-success/40 text-success",
    panel: "border-success/20",
    label: "Present",
  },
  weak: {
    dot: "bg-warning",
    badge:
      "border-warning/50 text-warning-foreground",
    panel: "border-warning/25",
    label: "Weak",
  },
  missing: {
    dot: "bg-destructive",
    badge:
      "border-destructive/40 text-destructive",
    panel: "border-destructive/20",
    label: "Missing",
  },
} as const;

const gapConfig: Record<
  IssueSeverity,
  {
    panel: string;
    badge: string;
    label: string;
  }
> = {
  high: {
    panel:
      "border-destructive/40 bg-destructive/5",
    badge:
      "border-destructive/40 text-destructive",
    label: "Missing",
  },
  medium: {
    panel:
      "border-warning/50 bg-warning/10",
    badge:
      "border-warning/50 text-warning-foreground",
    label: "Weak",
  },
  low: {
    panel:
      "border-accent/40 bg-accent/5",
    badge:
      "border-accent/40 text-accent",
    label: "Review",
  },
};

interface ResultsCardProps {
  result: ComplianceResult;
}

interface ParsedCoverageNote {
  description: string;
  reference: string;
  evidence: string;
  assessment: string;
}

function parseCoverageNote(
  note: string,
): ParsedCoverageNote {
  const lines = String(note || "")
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean);

  const referenceLine = lines.find(
    (line) =>
      line
        .toLowerCase()
        .startsWith("gdpr reference:"),
  );

  const evidenceLine = lines.find(
    (line) =>
      line
        .toLowerCase()
        .startsWith("evidence:"),
  );

  const assessmentLine = lines.find(
    (line) =>
      line
        .toLowerCase()
        .startsWith("assessment:"),
  );

  const description =
    lines.find(
      (line) =>
        !line
          .toLowerCase()
          .startsWith("gdpr reference:") &&
        !line
          .toLowerCase()
          .startsWith("evidence:") &&
        !line
          .toLowerCase()
          .startsWith("assessment:"),
    ) || "";

  return {
    description,
    reference: referenceLine
      ? referenceLine.replace(
          /^GDPR reference:\s*/i,
          "",
        )
      : "",
    evidence: evidenceLine
      ? evidenceLine.replace(
          /^Evidence:\s*/i,
          "",
        )
      : "",
    assessment: assessmentLine
      ? assessmentLine.replace(
          /^Assessment:\s*/i,
          "",
        )
      : "",
  };
}

function extractArticle(
  description: string,
): string {
  const match = description.match(
    /GDPR article to address:\s*(.*?)(?:\s+Why:|\n\nWhy:|$)/i,
  );

  return match?.[1]?.trim() || "";
}

function extractWhy(
  description: string,
): string {
  const match = description.match(
    /Why:\s*(.*)$/is,
  );

  return (
    match?.[1]?.trim() ||
    description
  );
}

export const ResultsCard = ({
  result,
}: ResultsCardProps) => {
  const [
    expandedSections,
    setExpandedSections,
  ] = useState<Set<string>>(
    new Set(),
  );

  const cfg =
    statusConfig[result.status];

  const StatusIcon = cfg.icon;

  const isHybrid =
    result.selectedMode === "hybrid";

  const presentCount =
    result.sections.filter(
      (section) =>
        section.status === "present",
    ).length;

  const weakCount =
    result.sections.filter(
      (section) =>
        section.status === "weak",
    ).length;

  const missingCount =
    result.sections.filter(
      (section) =>
        section.status === "missing",
    ).length;

  const toggleSection = (
    sectionName: string,
  ) => {
    setExpandedSections(
      (current) => {
        const updated =
          new Set(current);

        if (
          updated.has(sectionName)
        ) {
          updated.delete(
            sectionName,
          );
        } else {
          updated.add(
            sectionName,
          );
        }

        return updated;
      },
    );
  };

  return (
    <Card className="animate-in border-border/60 bg-gradient-card shadow-elevated fade-in slide-in-from-bottom-4 duration-500">
      <CardHeader className="space-y-2">
        <div className="flex items-center gap-2 text-accent">
          <FileSearch className="h-5 w-5" />

          <span className="text-xs font-medium uppercase tracking-widest">
            Step 02 — Results
          </span>
        </div>

        <CardTitle className="text-2xl md:text-3xl">
          Compliance Check Outcome
        </CardTitle>

        <CardDescription className="text-base">
          Analysis generated on{" "}
          {new Date(
            result.analysedAt,
          ).toLocaleString(
            undefined,
            {
              dateStyle: "long",
              timeStyle: "short",
            },
          )}{" "}
          •{" "}
          {result.wordCount.toLocaleString()}{" "}
          words analysed
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-10">
        {/* Overall result */}
        <section className="grid gap-6 rounded-xl border border-border/60 bg-secondary/30 p-6 md:grid-cols-[1fr_auto] md:items-center">
          <div className="flex items-start gap-4">
            <div
              className={cn(
                "flex h-12 w-12 shrink-0 items-center justify-center rounded-full border",
                cfg.badge,
              )}
            >
              <StatusIcon className="h-6 w-6" />
            </div>

            <div>
              <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
                Overall status
              </p>

              <p className="mt-1 text-2xl font-semibold text-foreground">
                {cfg.label}
              </p>

              <p className="mt-3 max-w-prose text-sm leading-relaxed text-muted-foreground">
                {result.summary}
              </p>
            </div>
          </div>

          <div className="md:min-w-[220px]">
            <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground md:text-right">
              Compliance score
            </p>

            <div className="mt-2 flex items-baseline gap-1 md:justify-end">
              <span className="text-5xl font-bold text-primary tabular-nums">
                {result.score}
              </span>

              <span className="text-xl text-muted-foreground">
                %
              </span>
            </div>

            <Progress
              value={result.score}
              className="mt-3 h-2"
            />
          </div>
        </section>

        {/* LLM-only method */}
        {!isHybrid &&
          result.analysisMethod && (
            <section className="rounded-xl border border-primary/20 bg-primary/5 p-5">
              <div className="flex items-start gap-4">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <BrainCircuit className="h-5 w-5" />
                </div>

                <div>
                  <p className="text-xs font-medium uppercase tracking-widest text-primary">
                    Analysis method
                  </p>

                  <h3 className="mt-1 text-lg font-semibold text-foreground">
                    {
                      result
                        .analysisMethod
                        .title
                    }
                  </h3>

                  <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                    {
                      result
                        .analysisMethod
                        .description
                    }
                  </p>

                  {result
                    .analysisMethod
                    .limitations && (
                    <div className="mt-3 flex items-start gap-2 rounded-lg border border-border/50 bg-background/70 p-3">
                      <Info className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />

                      <p className="text-xs leading-relaxed text-muted-foreground">
                        {
                          result
                            .analysisMethod
                            .limitations
                        }
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </section>
          )}

        {/* Truncation warning */}
        {result.policyWasTruncated && (
          <section className="rounded-xl border border-warning/40 bg-warning/10 p-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-warning" />

              <div>
                <p className="font-medium text-foreground">
                  Document analysis limit reached
                </p>

                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                  The submitted policy exceeded the
                  current analysis limit. The model
                  assessed the available policy text
                  and was instructed not to treat the
                  artificial cutoff as a compliance
                  failure.
                </p>
              </div>
            </div>
          </section>
        )}

        {/* LLM-only strengths */}
        {!isHybrid &&
          result.strengths.length >
            0 && (
            <section>
              <div className="mb-4 flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-success" />

                <h3 className="text-xl font-semibold">
                  Policy Strengths
                </h3>

                <Badge
                  variant="secondary"
                  className="ml-1"
                >
                  {
                    result
                      .strengths
                      .length
                  }
                </Badge>
              </div>

              <p className="mb-5 text-sm leading-relaxed text-muted-foreground">
                These are the main GDPR transparency
                and information areas that the language
                model identified as being clearly or
                positively addressed in the privacy notice.
              </p>

              <div className="grid gap-4 sm:grid-cols-2">
                {result.strengths.map(
                  (
                    strength,
                    index,
                  ) => (
                    <article
                      key={`${strength.title}-${index}`}
                      className="rounded-xl border border-success/25 bg-success/5 p-5"
                    >
                      <div className="flex items-start gap-3">
                        <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-success" />

                        <div>
                          <h4 className="font-medium text-foreground">
                            {
                              strength.title
                            }
                          </h4>

                          <div className="mt-3 space-y-3 text-sm leading-relaxed text-muted-foreground">
                            <div>
                              <p className="text-xs font-semibold uppercase tracking-wider text-foreground">
                                Evidence
                              </p>

                              <p className="mt-1">
                                {
                                  strength.evidence
                                }
                              </p>
                            </div>

                            {strength.gdprRelevance && (
                              <div>
                                <p className="text-xs font-semibold uppercase tracking-wider text-foreground">
                                  GDPR relevance
                                </p>

                                <p className="mt-1">
                                  {
                                    strength
                                      .gdprRelevance
                                  }
                                </p>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </article>
                  ),
                )}
              </div>
            </section>
          )}

        {/* Hybrid requirement coverage */}
        {isHybrid && (
          <section>
            <div className="mb-4 flex flex-wrap items-center gap-2">
              <ListChecks className="h-5 w-5 text-primary" />

              <h3 className="text-xl font-semibold">
                GDPR Requirement Coverage
              </h3>

              <div className="ml-auto flex flex-wrap gap-2 text-xs">
                <Badge
                  variant="outline"
                  className="border-success/40 text-success"
                >
                  {presentCount} Present
                </Badge>

                <Badge
                  variant="outline"
                  className="border-warning/50 text-warning-foreground"
                >
                  {weakCount} Weak
                </Badge>

                <Badge
                  variant="outline"
                  className="border-destructive/40 text-destructive"
                >
                  {missingCount} Missing
                </Badge>
              </div>
            </div>

            <p className="mb-5 text-sm leading-relaxed text-muted-foreground">
              This section shows whether key
              GDPR disclosure areas were found
              in the policy. It provides
              traceability evidence and does
              not independently determine the
              final compliance score.
            </p>

            <div className="grid gap-4 sm:grid-cols-2">
              {result.sections.map(
                (section) => {
                  const coverage =
                    coverageConfig[
                      section.status
                    ];

                  const expanded =
                    expandedSections.has(
                      section.name,
                    );

                  const parsedNote =
                    parseCoverageNote(
                      section.note,
                    );

                  const hasDetails =
                    Boolean(
                      parsedNote.evidence ||
                        parsedNote.assessment,
                    );

                  return (
                    <article
                      key={
                        section.name
                      }
                      className={cn(
                        "flex min-h-[200px] flex-col rounded-xl border bg-background p-5 transition-all",
                        coverage.panel,
                        expanded &&
                          "sm:col-span-2",
                      )}
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex min-w-0 items-start gap-3">
                          <span
                            className={cn(
                              "mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full",
                              coverage.dot,
                            )}
                            aria-hidden
                          />

                          <h4 className="font-medium leading-snug text-foreground">
                            {
                              section.name
                            }
                          </h4>
                        </div>

                        <Badge
                          variant="outline"
                          className={cn(
                            "shrink-0 text-[10px] uppercase tracking-wider",
                            coverage.badge,
                          )}
                        >
                          {
                            coverage.label
                          }
                        </Badge>
                      </div>

                      <div className="mt-4 flex-1 space-y-3 text-sm leading-relaxed text-muted-foreground">
                        {parsedNote.description && (
                          <p>
                            {
                              parsedNote.description
                            }
                          </p>
                        )}

                        {parsedNote.reference && (
                          <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                            GDPR reference:{" "}
                            {
                              parsedNote.reference
                            }
                          </p>
                        )}

                        {expanded &&
                          hasDetails && (
                            <div className="space-y-4 border-t border-border/60 pt-4">
                              {parsedNote.evidence && (
                                <div>
                                  <p className="text-xs font-semibold uppercase tracking-wider text-foreground">
                                    Evidence found
                                  </p>

                                  <p className="mt-1">
                                    {
                                      parsedNote
                                        .evidence
                                    }
                                  </p>
                                </div>
                              )}

                              {parsedNote.assessment && (
                                <div>
                                  <p className="text-xs font-semibold uppercase tracking-wider text-foreground">
                                    Assessment
                                  </p>

                                  <p className="mt-1">
                                    {
                                      parsedNote
                                        .assessment
                                    }
                                  </p>
                                </div>
                              )}
                            </div>
                          )}
                      </div>

                      {hasDetails && (
                        <button
                          type="button"
                          onClick={() =>
                            toggleSection(
                              section.name,
                            )
                          }
                          aria-expanded={
                            expanded
                          }
                          className="mt-4 inline-flex w-fit items-center gap-1 text-xs font-medium text-primary transition hover:opacity-75"
                        >
                          {expanded ? (
                            <>
                              Hide details
                              <ChevronUp className="h-4 w-4" />
                            </>
                          ) : (
                            <>
                              Show details
                              <ChevronDown className="h-4 w-4" />
                            </>
                          )}
                        </button>
                      )}
                    </article>
                  );
                },
              )}
            </div>
          </section>
        )}

        {/* Findings / gaps */}
        {result.issues.length > 0 ? (
          <section>
            <div className="mb-4 flex items-center gap-2">
              {isHybrid ? (
                <AlertTriangle className="h-5 w-5 text-warning" />
              ) : (
                <SearchCheck className="h-5 w-5 text-primary" />
              )}

              <h3 className="text-xl font-semibold">
                {isHybrid
                  ? "Compliance Gaps"
                  : "Key Findings"}
              </h3>

              <Badge
                variant="secondary"
                className="ml-1"
              >
                {
                  result.issues
                    .length
                }
              </Badge>
            </div>

            <p className="mb-5 text-sm leading-relaxed text-muted-foreground">
              {isHybrid
                ? "These gaps correspond directly to requirements classified as weak or missing in the coverage assessment above."
                : "These are the most important privacy-notice disclosure concerns identified during the holistic LLM review."}
            </p>

            <ul className="space-y-4">
              {result.issues.map(
                (
                  issue,
                  index,
                ) => {
                  const gap =
                    gapConfig[
                      issue.severity
                    ];

                  const article =
                    extractArticle(
                      issue.description,
                    );

                  const why =
                    extractWhy(
                      issue.description,
                    );

                  return (
                    <li
                      key={`${issue.title}-${index}`}
                      className={cn(
                        "rounded-xl border px-5 py-5",
                        isHybrid
                          ? gap.panel
                          : "border-border/70 bg-background",
                      )}
                    >
                      <div className="flex items-start justify-between gap-4">
                        <p className="font-medium text-foreground">
                          {
                            issue.title
                          }
                        </p>

                        {isHybrid && (
                          <Badge
                            variant="outline"
                            className={cn(
                              "shrink-0 text-[10px] uppercase tracking-wider",
                              gap.badge,
                            )}
                          >
                            {
                              gap.label
                            }
                          </Badge>
                        )}
                      </div>

                      {article && (
                        <p className="mt-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                          GDPR reference:{" "}
                          {article}
                        </p>
                      )}

                      <div className="mt-4 text-sm leading-relaxed text-muted-foreground">
                        <p>
                          <span className="font-medium text-foreground">
                            Why:{" "}
                          </span>

                          {why}
                        </p>
                      </div>
                    </li>
                  );
                },
              )}
            </ul>
          </section>
        ) : (
          <section className="rounded-xl border border-success/30 bg-success/5 p-5">
            <div className="flex items-start gap-3">
              <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-success" />

              <div>
                <h3 className="font-medium text-foreground">
                  No major compliance gaps detected
                </h3>

                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                  The assessment did not
                  identify significant GDPR
                  issues requiring corrective
                  action.
                </p>
              </div>
            </div>
          </section>
        )}

        {/* Recommended actions */}
        {result.recommendations.length >
          0 && (
          <section>
            <div className="mb-4 flex items-center gap-2">
              <Lightbulb className="h-5 w-5 text-accent" />

              <h3 className="text-xl font-semibold">
                Recommended Actions
              </h3>
            </div>

            <p className="mb-5 text-sm leading-relaxed text-muted-foreground">
              {isHybrid
                ? "These actions address the weak and missing requirements identified above."
                : "These actions address the main privacy-notice disclosure issues identified during the LLM-only review."}
            </p>

            <ol className="space-y-3">
              {result.recommendations.map(
                (
                  recommendation,
                  index,
                ) => (
                  <li
                    key={`${recommendation}-${index}`}
                    className="flex gap-4 rounded-xl border border-border/60 bg-background p-5 text-sm leading-relaxed text-foreground"
                  >
                    <span className="shrink-0 font-semibold text-accent tabular-nums">
                      {String(
                        index + 1,
                      ).padStart(
                        2,
                        "0",
                      )}
                    </span>

                    <span>
                      {
                        recommendation
                      }
                    </span>
                  </li>
                ),
              )}
            </ol>
          </section>
        )}

        {/* Download report */}
                {/* Download report */}
        <section className="overflow-hidden rounded-2xl border border-primary/20 bg-gradient-to-br from-primary/10 via-secondary/40 to-accent/10 p-6 shadow-sm">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-start gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl border border-primary/20 bg-background/80 text-primary shadow-sm">
                <Download className="h-6 w-6" />
              </div>

              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-xl font-semibold text-foreground">
                    Download report
                  </h3>

                  <Badge
                    variant="outline"
                    className="border-primary/30 bg-background/60 text-xs text-primary"
                  >
                    Exportable
                  </Badge>
                </div>

                <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
                  Save the full compliance assessment, including the overall result,
                  requirement coverage, identified gaps, evidence and recommended actions.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 lg:min-w-[360px]">
             <button
  type="button"
  onClick={() => exportReportAsPDF(result)}
  className="group rounded-xl border border-red-300/60 bg-red-100 px-4 py-3 text-sm font-semibold text-red-700 shadow-sm transition hover:-translate-y-0.5 hover:bg-red-200 hover:text-red-800 hover:shadow-md"
>
  <span className="flex items-center justify-center gap-2">
    <Download className="h-4 w-4 transition group-hover:scale-110" />
    PDF
  </span>
</button>

              <button
                type="button"
                onClick={() => {
                  void exportReportAsDOCX(result);
                }}
                className="group rounded-xl border border-accent/40 bg-accent/10 px-4 py-3 text-sm font-semibold text-accent transition hover:-translate-y-0.5 hover:bg-accent/15 hover:shadow-md"
              >
                <span className="flex items-center justify-center gap-2">
                  <Download className="h-4 w-4 transition group-hover:scale-110" />
                  Word
                </span>
              </button>

              <button
                type="button"
                onClick={() => exportReportAsTXT(result)}
                className="group rounded-xl border border-border/70 bg-background/80 px-4 py-3 text-sm font-semibold text-foreground transition hover:-translate-y-0.5 hover:bg-secondary hover:shadow-md"
              >
                <span className="flex items-center justify-center gap-2">
                  <Download className="h-4 w-4 transition group-hover:scale-110" />
                  TXT
                </span>
              </button>
            </div>
          </div>

          <div className="mt-5 rounded-xl border border-border/50 bg-background/60 px-4 py-3">
            <p className="text-xs leading-relaxed text-muted-foreground">
              The exported report is intended for documentation and review purposes only.
              It does not constitute legal advice.
            </p>
          </div>
        </section>
      </CardContent>
    </Card>
  );
};