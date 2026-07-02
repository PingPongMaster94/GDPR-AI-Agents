import * as pdfjsLib from "pdfjs-dist";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import mammoth from "mammoth";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

const SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx"];

function getFileExtension(file: File): string {
  const fileName = file.name.toLowerCase();
  const dotIndex = fileName.lastIndexOf(".");

  return dotIndex >= 0 ? fileName.slice(dotIndex) : "";
}

async function extractTextFromPdf(file: File): Promise<string> {
  const arrayBuffer = await file.arrayBuffer();

  const pdf = await pdfjsLib.getDocument({
    data: arrayBuffer,
  }).promise;

  const pages: string[] = [];

  for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
    const page = await pdf.getPage(pageNumber);
    const content = await page.getTextContent();

    const pageText = content.items
      .map((item) => {
        if ("str" in item && typeof item.str === "string") {
          return item.str;
        }

        return "";
      })
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();

    if (pageText) {
      pages.push(pageText);
    }
  }

  return pages.join("\n\n").trim();
}

async function extractTextFromDocx(file: File): Promise<string> {
  const arrayBuffer = await file.arrayBuffer();

  const result = await mammoth.extractRawText({
    arrayBuffer,
  });

  return result.value.trim();
}

export function isSupportedPolicyFile(file: File): boolean {
  const extension = getFileExtension(file);

  return SUPPORTED_EXTENSIONS.includes(extension);
}

export async function extractFileText(file: File): Promise<string> {
  const extension = getFileExtension(file);

  if (extension === ".txt") {
    return (await file.text()).trim();
  }

  if (extension === ".pdf") {
    return extractTextFromPdf(file);
  }

  if (extension === ".docx") {
    return extractTextFromDocx(file);
  }

  throw new Error(
    "Unsupported file type. Please upload a TXT, PDF, or DOCX file.",
  );
}