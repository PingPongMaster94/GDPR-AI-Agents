import { CheckCircle2, AlertTriangle, XCircle, FileSearch, Lightbulb, ListChecks } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import type { ComplianceResult } from "@/lib/complianceMock";
import { cn } from "@/lib/utils";

const statusConfig = {
  Compliant: {
    icon: CheckCircle2,
    badge: "bg-success/10 text-success border-success/30",
    bar: "bg-success",
    label: "Compliant",
  },
  "Partially Compliant": {
    icon: AlertTriangle,
    badge: "bg-warning/15 text-warning-foreground border-warning/40",
    bar: "bg-warning",
    label: "Partially Compliant",
  },
  "Non-Compliant": {
    icon: XCircle,
    badge: "bg-destructive/10 text-destructive border-destructive/30",
    bar: "bg-destructive",
    label: "Non-Compliant",
  },
} as const;

const sectionDot = {
  present: "bg-success",
  weak: "bg-warning",
  missing: "bg-destructive",
} as const;

const sectionLabel = {
  present: "Present",
  weak: "Weak",
  missing: "Missing",
} as const;

const severityColor = {
  high: "border-destructive/40 bg-destructive/5 text-destructive",
  medium: "border-warning/50 bg-warning/10 text-warning-foreground",
  low: "border-accent/40 bg-accent/5 text-accent",
} as const;

interface ResultsCardProps {
  result: ComplianceResult;
}

export const ResultsCard = ({ result }: ResultsCardProps) => {
  const cfg = statusConfig[result.status];
  const StatusIcon = cfg.icon;

  return (
    <Card className="border-border/60 shadow-elevated bg-gradient-card animate-in fade-in slide-in-from-bottom-4 duration-500">
      <CardHeader className="space-y-2">
        <div className="flex items-center gap-2 text-accent">
          <FileSearch className="h-5 w-5" />
          <span className="text-xs font-medium uppercase tracking-widest">Step 02 — Results</span>
        </div>
        <CardTitle className="text-2xl md:text-3xl">Compliance Check Outcome</CardTitle>
        <CardDescription className="text-base">
          Analysis generated on{" "}
          {new Date(result.analysedAt).toLocaleString(undefined, {
            dateStyle: "long",
            timeStyle: "short",
          })}{" "}
          • {result.wordCount.toLocaleString()} words analysed
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-8">
        {/* Status + Score */}
        <div className="grid gap-6 rounded-xl border border-border/60 bg-secondary/30 p-6 md:grid-cols-[1fr_auto] md:items-center">
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
              <p className="mt-1 font-serif text-2xl font-semibold text-foreground">
                {cfg.label}
              </p>
              <p className="mt-2 max-w-prose text-sm text-muted-foreground">{result.summary}</p>
            </div>
          </div>

          <div className="md:min-w-[220px]">
            <div className="flex items-baseline justify-between gap-2 md:justify-end">
              <span className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
                Compliance score
              </span>
            </div>
            <div className="mt-2 flex items-baseline gap-1 md:justify-end">
              <span className="font-serif text-5xl font-bold text-primary tabular-nums">
                {result.score}
              </span>
              <span className="text-xl text-muted-foreground">%</span>
            </div>
            <Progress value={result.score} className="mt-3 h-2" />
          </div>
        </div>

        {/* Sections grid */}
        <section>
          <div className="mb-4 flex items-center gap-2">
            <ListChecks className="h-5 w-5 text-primary" />
            <h3 className="font-serif text-xl font-semibold">Section coverage</h3>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {result.sections.map((s) => (
              <div
                key={s.name}
                className="flex items-start gap-3 rounded-lg border border-border/60 bg-background p-4"
              >
                <span
                  className={cn("mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full", sectionDot[s.status])}
                  aria-hidden
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <p className="font-medium text-foreground">{s.name}</p>
                    <Badge
                      variant="outline"
                      className={cn(
                        "text-[10px] uppercase tracking-wider",
                        s.status === "present" && "border-success/40 text-success",
                        s.status === "weak" && "border-warning/50 text-warning-foreground",
                        s.status === "missing" && "border-destructive/40 text-destructive",
                      )}
                    >
                      {sectionLabel[s.status]}
                    </Badge>
                  </div>
                  <p className="mt-1 text-sm text-muted-foreground">{s.note}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Issues */}
        {result.issues.length > 0 && (
          <section>
            <div className="mb-4 flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-warning" />
              <h3 className="font-serif text-xl font-semibold">Detected issues</h3>
              <Badge variant="secondary" className="ml-1">
                {result.issues.length}
              </Badge>
            </div>
            <ul className="space-y-3">
              {result.issues.map((issue, i) => (
                <li
                  key={i}
                  className={cn("rounded-lg border px-4 py-3", severityColor[issue.severity])}
                >
                  <div className="flex items-center justify-between gap-2">
                    <p className="font-medium text-foreground">{issue.title}</p>
                    <span className="text-[10px] font-semibold uppercase tracking-wider opacity-80">
                      {issue.severity}
                    </span>
                  </div>
                  <p className="mt-1 text-sm text-muted-foreground">{issue.description}</p>
                </li>
              ))}
            </ul>
          </section>
        )}

        {/* Recommendations */}
        <section>
          <div className="mb-4 flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-accent" />
            <h3 className="font-serif text-xl font-semibold">Recommendations</h3>
          </div>
          <ol className="space-y-2">
            {result.recommendations.map((r, i) => (
              <li
                key={i}
                className="flex gap-3 rounded-lg border border-border/60 bg-background p-4 text-sm leading-relaxed text-foreground"
              >
                <span className="font-serif font-semibold text-accent tabular-nums">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span>{r}</span>
              </li>
            ))}
          </ol>
        </section>
      </CardContent>
    </Card>
  );
};
