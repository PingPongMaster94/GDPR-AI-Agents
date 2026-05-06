GDPR Compliance Checker

Automated GDPR compliance evaluation pipeline developed for a Data Science thesis project.

The system combines:

rule-based GDPR heuristics
semantic GDPR article retrieval
local LLM reasoning
score fusion into a final compliance assessment
PROJECT STRUCTURE

project_root/

src/
extract_gdpr_sections.py
build_dataset.py
semantic_linker.py
gdpr_agent.py
combine_scores_LLM.py
cleaning_utils.py
data_loader.py

data/
raw_law/
raw_policies/
processed/
annotated/

requirements.txt
README.txt

All Python scripts are expected to live inside "src".

All paths should be resolved relative to the project root using:

PROJECT_ROOT = Path(file).resolve().parents[1]

Do NOT use absolute paths like /Users/...

PIPELINE RUN ORDER

Run all commands from the project root:

python src/extract_gdpr_sections.py
python src/build_dataset.py
python src/semantic_linker.py
python src/combine_scores_LLM.py

Each step depends on the previous one.

PIPELINE STEPS
Extract GDPR Articles

Command:
python src/extract_gdpr_sections.py

Input:
data/raw_law/Reference Law - GDPR.txt

Output:
data/reference_law_articles.csv

This extracts GDPR recitals and articles into a structured dataset.

Build Policy Dataset

Command:
python src/build_dataset.py

Input:
data/raw_policies/*.txt

Output:
data/processed/paragraphs.csv

This cleans raw policies and splits them into paragraphs.

Semantic Linking

Command:
python src/semantic_linker.py

Inputs:
data/processed/paragraphs.csv
data/reference_law_articles.csv

Output:
data/annotated/paragraphs_with_articles.csv

Each paragraph is linked to the most relevant GDPR article.

Hybrid Compliance Scoring

Command:
python src/combine_scores_LLM.py

Inputs:
data/annotated/paragraphs_with_articles.csv
data/reference_law_articles.csv

Output:
data/annotated/combined_compliance_scores.csv

Final score combines:

Heuristic score = 0.25
Semantic score = 0.25
LLM verdict = 0.50

LLM BACKEND

Default backend: Ollama
Default model: phi3:mini

Install models:

ollama pull phi3:mini
ollama pull mistral

Run normally:

python src/combine_scores_LLM.py

Run with another model:

OLLAMA_MODEL=mistral python src/combine_scores_LLM.py

If using backend switch:

LLM_BACKEND=ollama OLLAMA_MODEL=phi3:mini python src/combine_scores_LLM.py

TESTING CONTROLS

Limit number of policies:

LIMIT_SOURCES=5 python src/combine_scores_LLM.py

Limit number of rows:

LIMIT_ROWS=100 python src/combine_scores_LLM.py

Optional run naming (if enabled in code):

RUN_NAME=phi3_test LIMIT_SOURCES=5 OLLAMA_MODEL=phi3:mini python src/combine_scores_LLM.py

RUN_NAME=mistral_test LIMIT_SOURCES=5 OLLAMA_MODEL=mistral python src/combine_scores_LLM.py

REQUIREMENTS

Install with:

pip install -r requirements.txt

Core requirements:

pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
regex>=2023.10.3
tqdm>=4.66.0
python-dateutil>=2.8.2
joblib>=1.3.0

Optional (for HuggingFace models):

torch
transformers
accelerate
safetensors

TROUBLESHOOTING

FileNotFoundError:

Usually means:

pipeline steps were run out of order
wrong working directory
missing intermediate files

Check these exist:

data/reference_law_articles.csv
data/processed/paragraphs.csv
data/annotated/paragraphs_with_articles.csv

Ollama errors:

Make sure:

Ollama is installed
Ollama is running
model is downloaded

Example:

ollama pull phi3:mini

NOTES

This is a hybrid system, not a pure ML model.

The final compliance score is based on:

rule-based checks
semantic similarity
LLM reasoning

The LLM acts as an auditing component, not the sole decision-maker.



Project layers:

🟢 Layer 1 — Data Collection
Webscrap.ipynb

👉 produces:

data/raw_policies/*.txt
🔵 Layer 2 — Core Pipeline (production)
extract_gdpr_sections.py
build_dataset.py
semantic_linker.py
combine_scores_LLM.py

👉 produces final output

🟡 Layer 3 — Exploration / Experiments
Dataset exploring.ipynb

👉 used for:

testing ideas
validating models
visuals