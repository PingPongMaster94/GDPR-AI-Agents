import { useCallback, useRef, useState } from "react";
import { FileText, Upload, X, Loader2, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

const ACCEPTED = ".txt,.pdf,.doc,.docx";
const ACCEPTED_MIME = [
  "text/plain",
  "application/pdf",
  "application/msword",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
];

interface UploadCardProps {
  onSubmit: (input: string, fileName?: string) => void;
  isLoading: boolean;
  error: string | null;
}

export const UploadCard = ({ onSubmit, isLoading, error }: UploadCardProps) => {
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (f: File) => {
    setFile(f);
    // Best-effort: read .txt content into the textarea so the model has text.
    if (f.type === "text/plain" || f.name.endsWith(".txt")) {
      const content = await f.text();
      setText(content);
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  const handleSubmit = () => {
    const content = text.trim();
    if (!content && !file) return onSubmit("", undefined); // surfaces validation error in parent
    onSubmit(content || `[Binary document submitted: ${file?.name}]`, file?.name);
  };

  return (
    <Card className="border-border/60 shadow-elegant bg-gradient-card">
      <CardHeader className="space-y-2">
        <div className="flex items-center gap-2 text-accent">
          <ShieldCheck className="h-5 w-5" />
          <span className="text-xs font-medium uppercase tracking-widest">Step 01</span>
        </div>
        <CardTitle className="text-2xl md:text-3xl">Submit a Privacy Policy</CardTitle>
        <CardDescription className="text-base">
          You can either upload a document or paste the policy text directly.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Drop zone */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={cn(
            "group relative cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-smooth",
            dragOver
              ? "border-accent bg-accent/5"
              : "border-border hover:border-accent/60 hover:bg-secondary/40",
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED}
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
            }}
          />

          {file ? (
            <div className="flex items-center justify-between rounded-lg bg-secondary/60 p-4 text-left">
              <div className="flex items-center gap-3 min-w-0">
                <FileText className="h-5 w-5 shrink-0 text-primary" />
                <div className="min-w-0">
                  <p className="truncate font-medium text-foreground">{file.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={(e) => {
                  e.stopPropagation();
                  setFile(null);
                  if (inputRef.current) inputRef.current.value = "";
                }}
                aria-label="Remove file"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-secondary text-primary transition-smooth group-hover:scale-110 group-hover:bg-accent group-hover:text-accent-foreground">
                <Upload className="h-5 w-5" />
              </div>
              <div>
                <p className="font-medium text-foreground">
                  Drag & drop your policy file here, or click to browse
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Accepted formats: .txt, .pdf, .doc, .docx
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Divider */}
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <span className="w-full border-t border-border" />
          </div>
          <div className="relative flex justify-center text-xs uppercase tracking-widest">
            <span className="bg-card px-3 text-muted-foreground">Or paste text</span>
          </div>
        </div>

        {/* Textarea */}
        <div className="space-y-2">
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste the full privacy policy text here…"
            className="min-h-[180px] resize-y border-border/80 bg-background font-mono text-sm leading-relaxed"
          />
          <p className="text-xs text-muted-foreground">
            {text.trim() ? `${text.trim().split(/\s+/).length} words` : "No text entered yet."}
          </p>
        </div>

        {error && (
          <div
            role="alert"
            className="rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive"
          >
            {error}
          </div>
        )}

        <Button
          onClick={handleSubmit}
          disabled={isLoading}
          size="lg"
          className="w-full bg-primary text-primary-foreground hover:bg-primary/90 sm:w-auto sm:px-8"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Analysing…
            </>
          ) : (
            "Run Compliance Check"
          )}
        </Button>
      </CardContent>
    </Card>
  );
};
