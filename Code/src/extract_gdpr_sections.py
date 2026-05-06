import re
import pandas as pd

# === CONFIG ===
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "raw_law" / "Reference Law - GDPR.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "reference_law_articles.csv"

def extract_sections(text: str):
    text = re.sub(r'\r', '', text)
    text = re.sub(r'\n{2,}', '\n\n', text)

    # Identify where real articles begin
    chapter_match = re.search(r'\bCHAPTER\s+I\b', text, re.IGNORECASE)
    chapter_start = chapter_match.start() if chapter_match else len(text)

    recitals_block = text[:chapter_start]
    articles_block = text[chapter_start:]

    sections = []

    # -------------------------------
    # Extract Recitals (Before CHAPTER I)
    # -------------------------------
    recitals = re.split(r'\((\d{1,3})\)\s*\n', recitals_block)
    for i in range(1, len(recitals), 2):
        num = recitals[i]
        body = recitals[i + 1].strip() if i + 1 < len(recitals) else ""
        if body:
            sections.append({
                "section_type": "Recital",
                "number": num,
                "title": f"Recital {num}",
                "text": body
            })

    # -------------------------------
    # Extract Articles (After CHAPTER I)
    # -------------------------------
    # Matches headers like "Article 1" or "Article 2 – Title"
    article_header_pattern = re.compile(r'(Article\s+\d+[A-Z]?(?:\s*[-–]\s*[^\n]*)?)\n', re.IGNORECASE)

    article_matches = list(article_header_pattern.finditer(articles_block))
    for i, match in enumerate(article_matches):
        header_start = match.start()
        header_end = match.end()
        next_start = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(articles_block)

        header = match.group(1).strip()
        body = articles_block[header_end:next_start].strip()

        num, title = "", ""
        header_match = re.match(r'Article\s+(\d+[A-Z]?)\s*[-–]?\s*(.*)', header)
        if header_match:
            num = header_match.group(1)
            title = header_match.group(2).strip()
        sections.append({
            "section_type": "Article",
            "number": num,
            "title": title,
            "text": body
        })

    return sections

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    sections = extract_sections(text)
    df = pd.DataFrame(sections)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    recitals = len(df[df["section_type"] == "Recital"])
    articles = len(df[df["section_type"] == "Article"])
    print(f"✅ Extracted {recitals} Recitals and {articles} Articles → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()