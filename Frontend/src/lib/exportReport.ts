import jsPDF from "jspdf";
import {
  Document,
  HeadingLevel,
  Packer,
  Paragraph,
  TextRun,
} from "docx";
import { saveAs } from "file-saver";

import type { ComplianceResult } from "@/lib/complianceMock";


function formatMode(result: ComplianceResult): string {
  return result.selectedMode === "hybrid"
    ? "Hybrid"
    : "LLM-only";
}


function formatDate(value: string): string {
  return new Date(value).toLocaleString(undefined, {
    dateStyle: "long",
    timeStyle: "short",
  });
}


function buildReportText(result: ComplianceResult): string {
  const lines: string[] = [];

  lines.push("GDPR Compliance Checker Report");
  lines.push("================================");
  lines.push("");
  lines.push(`Generated at: ${formatDate(result.analysedAt)}`);
  lines.push(`Analysis mode: ${formatMode(result)}`);
  lines.push(`API mode: ${result.apiMode}`);
  lines.push(`Words analysed: ${result.wordCount}`);
  lines.push("");
  lines.push("Overall Assessment");
  lines.push("------------------");
  lines.push(`Status: ${result.status}`);
  lines.push(`Compliance score: ${result.score}%`);
  lines.push(`Summary: ${result.summary}`);
  lines.push("");

  if (result.analysisMethod) {
    lines.push("Analysis Method");
    lines.push("---------------");
    lines.push(result.analysisMethod.title);
    lines.push(result.analysisMethod.description);
    if (result.analysisMethod.limitations) {
      lines.push(`Limitations: ${result.analysisMethod.limitations}`);
    }
    lines.push("");
  }

  if (result.strengths.length > 0) {
    lines.push("Policy Strengths");
    lines.push("----------------");
    result.strengths.forEach((strength, index) => {
      lines.push(`${index + 1}. ${strength.title}`);
      lines.push(`Evidence: ${strength.evidence}`);
      if (strength.gdprRelevance) {
        lines.push(`GDPR relevance: ${strength.gdprRelevance}`);
      }
      lines.push("");
    });
  }

  if (result.sections.length > 0) {
    lines.push("GDPR Requirement Coverage");
    lines.push("-------------------------");
    result.sections.forEach((section, index) => {
      lines.push(`${index + 1}. ${section.name}`);
      lines.push(`Status: ${section.status}`);
      if (section.note) {
        lines.push(`Details: ${section.note}`);
      }
      lines.push("");
    });
  }

  if (result.issues.length > 0) {
    lines.push(
      result.selectedMode === "hybrid"
        ? "Compliance Gaps"
        : "Key Findings",
    );
    lines.push("----------------");
    result.issues.forEach((issue, index) => {
      lines.push(`${index + 1}. ${issue.title}`);
      lines.push(`Description: ${issue.description}`);
      lines.push("");
    });
  }

  if (result.recommendations.length > 0) {
    lines.push("Recommended Actions");
    lines.push("-------------------");
    result.recommendations.forEach((recommendation, index) => {
      lines.push(`${index + 1}. ${recommendation}`);
    });
    lines.push("");
  }

  lines.push("Disclaimer");
  lines.push("----------");
  lines.push(
    "This report was generated automatically by a prototype academic tool. It does not constitute legal advice and should be reviewed by a qualified professional before being used for compliance decisions.",
  );

  return lines.join("\n");
}


function downloadBlob(
  content: BlobPart,
  filename: string,
  type: string,
): void {
  const blob = new Blob([content], { type });
  saveAs(blob, filename);
}


function makeFilename(
  result: ComplianceResult,
  extension: string,
): string {
  const mode = result.selectedMode === "hybrid"
    ? "hybrid"
    : "llm-only";

  const date = new Date()
    .toISOString()
    .slice(0, 10);

  return `gdpr-compliance-report-${mode}-${date}.${extension}`;
}


export function exportReportAsTXT(result: ComplianceResult): void {
  const text = buildReportText(result);

  downloadBlob(
    text,
    makeFilename(result, "txt"),
    "text/plain;charset=utf-8",
  );
}


export function exportReportAsPDF(result: ComplianceResult): void {
  const doc = new jsPDF({
    orientation: "portrait",
    unit: "pt",
    format: "a4",
  });

  const marginX = 48;
  const marginTop = 56;
  const lineHeight = 15;
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const maxWidth = pageWidth - marginX * 2;

  let y = marginTop;

  const addPageIfNeeded = (extraHeight = lineHeight) => {
    if (y + extraHeight > pageHeight - marginTop) {
      doc.addPage();
      y = marginTop;
    }
  };

  const addText = (
    text: string,
    options?: {
      fontSize?: number;
      bold?: boolean;
      spacingAfter?: number;
    },
  ) => {
    const fontSize = options?.fontSize || 10;
    const spacingAfter = options?.spacingAfter ?? 6;

    doc.setFont("helvetica", options?.bold ? "bold" : "normal");
    doc.setFontSize(fontSize);

    const wrapped = doc.splitTextToSize(text || " ", maxWidth);

    wrapped.forEach((line: string) => {
      addPageIfNeeded(lineHeight);
      doc.text(line, marginX, y);
      y += lineHeight;
    });

    y += spacingAfter;
  };

  const addHeading = (text: string) => {
    y += 8;
    addText(text, {
      fontSize: 14,
      bold: true,
      spacingAfter: 8,
    });
  };

  addText("GDPR Compliance Checker Report", {
    fontSize: 18,
    bold: true,
    spacingAfter: 14,
  });

  addText(`Generated at: ${formatDate(result.analysedAt)}`);
  addText(`Analysis mode: ${formatMode(result)}`);
  addText(`Words analysed: ${result.wordCount}`);

  addHeading("Overall Assessment");
  addText(`Status: ${result.status}`, { bold: true });
  addText(`Compliance score: ${result.score}%`, { bold: true });
  addText(result.summary);

  if (result.analysisMethod) {
    addHeading("Analysis Method");
    addText(result.analysisMethod.title, { bold: true });
    addText(result.analysisMethod.description);

    if (result.analysisMethod.limitations) {
      addText(`Limitations: ${result.analysisMethod.limitations}`);
    }
  }

  if (result.strengths.length > 0) {
    addHeading("Policy Strengths");

    result.strengths.forEach((strength, index) => {
      addText(`${index + 1}. ${strength.title}`, { bold: true });
      addText(`Evidence: ${strength.evidence}`);

      if (strength.gdprRelevance) {
        addText(`GDPR relevance: ${strength.gdprRelevance}`);
      }
    });
  }

  if (result.sections.length > 0) {
    addHeading("GDPR Requirement Coverage");

    result.sections.forEach((section, index) => {
      addText(`${index + 1}. ${section.name}`, { bold: true });
      addText(`Status: ${section.status}`);

      if (section.note) {
        addText(section.note);
      }
    });
  }

  if (result.issues.length > 0) {
    addHeading(
      result.selectedMode === "hybrid"
        ? "Compliance Gaps"
        : "Key Findings",
    );

    result.issues.forEach((issue, index) => {
      addText(`${index + 1}. ${issue.title}`, { bold: true });
      addText(issue.description);
    });
  }

  if (result.recommendations.length > 0) {
    addHeading("Recommended Actions");

    result.recommendations.forEach((recommendation, index) => {
      addText(`${index + 1}. ${recommendation}`);
    });
  }

  addHeading("Disclaimer");
  addText(
    "This report was generated automatically by a prototype academic tool. It does not constitute legal advice and should be reviewed by a qualified professional before being used for compliance decisions.",
  );

  doc.save(makeFilename(result, "pdf"));
}


export async function exportReportAsDOCX(
  result: ComplianceResult,
): Promise<void> {
  const children: Paragraph[] = [];

  const heading = (text: string) => {
    children.push(
      new Paragraph({
        text,
        heading: HeadingLevel.HEADING_1,
      }),
    );
  };

  const subheading = (text: string) => {
    children.push(
      new Paragraph({
        text,
        heading: HeadingLevel.HEADING_2,
      }),
    );
  };

  const paragraph = (text: string, bold = false) => {
    children.push(
      new Paragraph({
        children: [
          new TextRun({
            text,
            bold,
          }),
        ],
      }),
    );
  };

  heading("GDPR Compliance Checker Report");

  paragraph(`Generated at: ${formatDate(result.analysedAt)}`);
  paragraph(`Analysis mode: ${formatMode(result)}`);
  paragraph(`API mode: ${result.apiMode}`);
  paragraph(`Words analysed: ${result.wordCount}`);

  subheading("Overall Assessment");
  paragraph(`Status: ${result.status}`, true);
  paragraph(`Compliance score: ${result.score}%`, true);
  paragraph(result.summary);

  if (result.analysisMethod) {
    subheading("Analysis Method");
    paragraph(result.analysisMethod.title, true);
    paragraph(result.analysisMethod.description);

    if (result.analysisMethod.limitations) {
      paragraph(`Limitations: ${result.analysisMethod.limitations}`);
    }
  }

  if (result.strengths.length > 0) {
    subheading("Policy Strengths");

    result.strengths.forEach((strength, index) => {
      paragraph(`${index + 1}. ${strength.title}`, true);
      paragraph(`Evidence: ${strength.evidence}`);

      if (strength.gdprRelevance) {
        paragraph(`GDPR relevance: ${strength.gdprRelevance}`);
      }
    });
  }

  if (result.sections.length > 0) {
    subheading("GDPR Requirement Coverage");

    result.sections.forEach((section, index) => {
      paragraph(`${index + 1}. ${section.name}`, true);
      paragraph(`Status: ${section.status}`);

      if (section.note) {
        paragraph(`Details: ${section.note}`);
      }
    });
  }

  if (result.issues.length > 0) {
    subheading(
      result.selectedMode === "hybrid"
        ? "Compliance Gaps"
        : "Key Findings",
    );

    result.issues.forEach((issue, index) => {
      paragraph(`${index + 1}. ${issue.title}`, true);
      paragraph(issue.description);
    });
  }

  if (result.recommendations.length > 0) {
    subheading("Recommended Actions");

    result.recommendations.forEach((recommendation, index) => {
      paragraph(`${index + 1}. ${recommendation}`);
    });
  }

  subheading("Disclaimer");
  paragraph(
    "This report was generated automatically by a prototype academic tool. It does not constitute legal advice and should be reviewed by a qualified professional before being used for compliance decisions.",
  );

  const document = new Document({
    sections: [
      {
        children,
      },
    ],
  });

  const blob = await Packer.toBlob(document);

  saveAs(
    blob,
    makeFilename(result, "docx"),
  );
}