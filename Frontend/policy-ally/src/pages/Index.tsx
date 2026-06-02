import { useEffect, useRef, useState } from "react";
import { Loader2, ShieldCheck, ScrollText, GraduationCap } from "lucide-react";
import { UploadCard } from "@/components/UploadCard";
import { ResultsCard } from "@/components/ResultsCard";
import { runComplianceCheck, type ComplianceResult } from "@/lib/complianceMock";

const Index = () => {
const [isLoading, setIsLoading] = useState(false);
const [error, setError] = useState<string | null>(null);
const [result, setResult] = useState<ComplianceResult | null>(null);
const [analysisMode, setAnalysisMode] = useState<"llm_only" | "hybrid">("llm_only");
const resultsRef = useRef<HTMLDivElement>(null);

  const handleSubmit = async (input: string) => {
    if (!input || !input.trim()) {
      setError("Please upload a document or paste a privacy policy before running the check.");
      setResult(null);
      return;
    }
    setError(null);
    setIsLoading(true);
    setResult(null);
    try {
      const r = await runComplianceCheck(input, analysisMode);
      setResult(r);
    } catch (e) {
      setError("Something went wrong while analysing the policy. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (result && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [result]);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/60 bg-background/80 backdrop-blur-sm">
        <div className="container mx-auto flex h-16 items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <ShieldCheck className="h-4 w-4" />
            </div>
            <span className="font-serif text-lg font-semibold tracking-tight">
              GDPR Compliance Checker
            </span>
          </div>
          <div className="hidden items-center gap-2 text-xs text-muted-foreground sm:flex">
            <GraduationCap className="h-4 w-4" />
            <span>Thesis Project</span>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div
          className="absolute inset-0 -z-10 opacity-[0.04]"
          style={{
            backgroundImage:
              "radial-gradient(circle at 1px 1px, hsl(var(--primary)) 1px, transparent 0)",
            backgroundSize: "32px 32px",
          }}
          aria-hidden
        />
        <div className="container mx-auto px-4 py-16 md:py-24">
          <div className="mx-auto max-w-3xl text-center">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-border bg-secondary/60 px-4 py-1.5 text-xs font-medium text-secondary-foreground">
              <span className="h-1.5 w-1.5 rounded-full bg-accent" />
              AI-powered privacy policy analysis
            </div>
            <h1 className="font-serif text-4xl font-bold leading-tight tracking-tight text-foreground md:text-6xl">
              GDPR Compliance{" "}
              <span className="bg-gradient-hero bg-clip-text text-transparent">Checker</span>
            </h1>
            <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-muted-foreground md:text-xl">
              Upload or paste a privacy policy to assess its alignment with GDPR requirements
              using an AI-powered compliance analysis tool.
            </p>

            <div className="mx-auto mt-8 max-w-xl rounded-xl border border-border/70 bg-card/60 p-5 text-left shadow-sm">
              <div className="flex items-start gap-3">
                <ScrollText className="mt-0.5 h-5 w-5 shrink-0 text-accent" />
                <p className="text-sm leading-relaxed text-muted-foreground">
                  <span className="font-medium text-foreground">About this project. </span>
                  This tool was developed as part of a thesis project to support automated
                  privacy policy analysis and identify potential GDPR compliance issues.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main content */}
      <main className="container mx-auto px-4 pb-24">
        <div className="mx-auto max-w-3xl space-y-10">
  <div className="rounded-xl border border-border/70 bg-card/70 p-5 shadow-sm">
    <p className="mb-3 text-sm font-medium text-foreground">
      Analysis mode
    </p>

    <div className="grid grid-cols-2 gap-3">
      <button
        type="button"
        onClick={() => setAnalysisMode("llm_only")}
        disabled={isLoading}
        className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
          analysisMode === "llm_only"
            ? "bg-primary text-primary-foreground"
            : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
        }`}
      >
        LLM-only
      </button>

      <button
        type="button"
        onClick={() => setAnalysisMode("hybrid")}
        disabled={isLoading}
        className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
          analysisMode === "hybrid"
            ? "bg-primary text-primary-foreground"
            : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
        }`}
      >
        Hybrid system
      </button>
    </div>

    <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
      {analysisMode === "llm_only"
        ? "LLM-only mode reviews the full policy as one document using the hosted model."
        : "Hybrid mode combines heuristic checks, semantic article matching, and targeted LLM review."}
    </p>
  </div>

  <UploadCard onSubmit={handleSubmit} isLoading={isLoading} error={error} />

          {/* Loading state */}
          {isLoading && (
            <div className="flex flex-col items-center justify-center gap-4 rounded-xl border border-border/60 bg-card p-12 text-center shadow-elegant">
              <div className="relative">
                <div className="absolute inset-0 animate-ping rounded-full bg-accent/20" />
                <div className="relative flex h-14 w-14 items-center justify-center rounded-full bg-accent/10 text-accent">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>
              </div>
              <div>
                <p className="font-serif text-xl font-semibold text-foreground">
                  Analysing privacy policy for GDPR compliance…
                </p>
                <p className="mt-2 text-sm text-muted-foreground">
                  The AI agent is reviewing the document against key GDPR articles.
                </p>
              </div>
            </div>
          )}

          {/* Results */}
          <div ref={resultsRef}>
            {result && !isLoading && <ResultsCard result={result} />}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/60 bg-secondary/30">
        <div className="container mx-auto px-4 py-8">
          <div className="flex flex-col items-center justify-between gap-3 text-sm text-muted-foreground sm:flex-row">
            <p>GDPR Compliance Checker – Thesis Project</p>
            <p className="text-xs">For academic demonstration purposes only.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
