import { useCallback, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  FileText,
  Loader2,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { extractFileText, isSupportedPolicyFile } from "@/lib/extractFileText";
import { cn } from "@/lib/utils";

const ACCEPTED = ".txt,.pdf,.docx";

interface UploadCardProps {
  onSubmit: (input: string, fileName?: string) => void;
  isLoading: boolean;
  error: string | null;
}

export const UploadCard = ({
  onSubmit,
  isLoading,
  error,
}: UploadCardProps) => {
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  const clearFile = () => {
    setFile(null);
    setText("");
    setFileError(null);

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const handleFile = useCallback(async (selectedFile: File) => {
    setFileError(null);

    if (!isSupportedPolicyFile(selectedFile)) {
      setFile(null);
      setText("");

      if (inputRef.current) {
        inputRef.current.value = "";
      }

      setFileError(
        "Unsupported file type. Please upload a TXT, PDF, or DOCX file.",
      );

      return;
    }

    setIsExtracting(true);
    setFile(selectedFile);
    setText("");

    try {
      const extractedText = await extractFileText(selectedFile);
      const cleanedText = extractedText.replace(/\s+/g, " ").trim();

      if (!cleanedText || cleanedText.split(/\s+/).length < 30) {
        setFile(null);
        setText("");

        if (inputRef.current) {
          inputRef.current.value = "";
        }

        setFileError(
          "Could not extract enough readable text from this file. If this is a scanned PDF, please copy and paste the policy text manually.",
        );

        return;
      }

      setText(extractedText);
    } catch {
      setFile(null);
      setText("");

      if (inputRef.current) {
        inputRef.current.value = "";
      }

      setFileError(
        "Could not read this file. Please try another PDF/DOCX/TXT file or paste the policy text manually.",
      );
    } finally {
      setIsExtracting(false);
    }
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);

      const selectedFile = e.dataTransfer.files?.[0];

      if (selectedFile) {
        void handleFile(selectedFile);
      }
    },
    [handleFile],
  );

  const handleSubmit = () => {
    const content = text.trim();

    if (!content) {
      onSubmit("", undefined);
      return;
    }

    onSubmit(content, file?.name);
  };

  return (
    <Card className="border-border/60 bg-gradient-card shadow-elegant">
      <CardHeader className="space-y-2">
        <div className="flex items-center gap-2 text-accent">
          <ShieldCheck className="h-5 w-5" />

          <span className="text-xs font-medium uppercase tracking-widest">
            Step 01
          </span>
        </div>

        <CardTitle className="text-2xl md:text-3xl">
          Submit a Privacy Policy
        </CardTitle>

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
          onClick={() => {
            if (!isExtracting && !isLoading) {
              inputRef.current?.click();
            }
          }}
          className={cn(
            "group relative cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-smooth",
            dragOver
              ? "border-accent bg-accent/5"
              : "border-border hover:border-accent/60 hover:bg-secondary/40",
            (isExtracting || isLoading) && "cursor-not-allowed opacity-80",
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED}
            className="hidden"
            disabled={isExtracting || isLoading}
            onChange={(e) => {
              const selectedFile = e.target.files?.[0];

              if (selectedFile) {
                void handleFile(selectedFile);
              }
            }}
          />

          {file ? (
            <div className="flex items-center justify-between rounded-lg bg-secondary/60 p-4 text-left">
              <div className="flex min-w-0 items-center gap-3">
                {isExtracting ? (
                  <Loader2 className="h-5 w-5 shrink-0 animate-spin text-primary" />
                ) : (
                  <FileText className="h-5 w-5 shrink-0 text-primary" />
                )}

                <div className="min-w-0">
                  <p className="truncate font-medium text-foreground">
                    {file.name}
                  </p>

                  <p className="text-xs text-muted-foreground">
                    {isExtracting
                      ? "Extracting readable text..."
                      : `${(file.size / 1024).toFixed(1)} KB • ${wordCount.toLocaleString()} words extracted`}
                  </p>
                </div>
              </div>

              <Button
                variant="ghost"
                size="icon"
                disabled={isExtracting || isLoading}
                onClick={(e) => {
                  e.stopPropagation();
                  clearFile();
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
                  Accepted formats: .txt, .pdf, .docx
                </p>

                <p className="mt-1 text-xs text-muted-foreground">
                  Scanned PDFs are not supported unless selectable text is present.
                </p>
              </div>
            </div>
          )}
        </div>

        {(fileError || error) && (
          <div
            role="alert"
            className="rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive"
          >
            <div className="flex items-start gap-2">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{fileError || error}</span>
            </div>
          </div>
        )}

        {file && !isExtracting && text.trim() && (
          <div className="rounded-lg border border-success/30 bg-success/5 px-4 py-3 text-sm text-success">
            <div className="flex items-start gap-2">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
              <span>
                Text extracted successfully. You can review or edit it below before
                running the compliance check.
              </span>
            </div>
          </div>
        )}

        {/* Divider */}
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <span className="w-full border-t border-border" />
          </div>

          <div className="relative flex justify-center text-xs uppercase tracking-widest">
            <span className="bg-card px-3 text-muted-foreground">
              Or paste text
            </span>
          </div>
        </div>

        {/* Textarea */}
        <div className="space-y-2">
          <Textarea
            value={text}
            onChange={(e) => {
              setText(e.target.value);

              if (fileError) {
                setFileError(null);
              }
            }}
            placeholder="Paste the full privacy policy text here…"
            className="min-h-[180px] resize-y border-border/80 bg-background font-mono text-sm leading-relaxed"
          />

          <p className="text-xs text-muted-foreground">
            {wordCount
              ? `${wordCount.toLocaleString()} words`
              : "No text entered yet."}
          </p>
        </div>

        <Button
          onClick={handleSubmit}
          disabled={isLoading || isExtracting}
          size="lg"
          className="w-full bg-primary text-primary-foreground hover:bg-primary/90 sm:w-auto sm:px-8"
        >
          {isExtracting ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Extracting text…
            </>
          ) : isLoading ? (
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