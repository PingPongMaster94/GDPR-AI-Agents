# 🛡️ GDPR Compliance Checker

Automated GDPR compliance evaluation pipeline developed as part of a Data Science thesis.

This project combines **rule-based logic, semantic retrieval, and LLM reasoning** to assess whether privacy policies align with GDPR requirements.

---

## 🚀 Overview

The system evaluates privacy policies through a **hybrid pipeline**:

- 🧠 Heuristic analysis (GDPR checklist)
- 🔎 Semantic matching to GDPR articles
- 🤖 LLM-based compliance reasoning
- 📊 Final weighted compliance score

This approach ensures **interpretability + robustness**, avoiding reliance on a single black-box model.

---

## 🏗️ Project Structure


project_root/
│
├── src/
│ ├── extract_gdpr_sections.py
│ ├── build_dataset.py
│ ├── semantic_linker.py
│ ├── gdpr_agent.py
│ ├── combine_scores_LLM.py
│ ├── cleaning_utils.py
│ └── data_loader.py
│
├── data/
│ ├── raw_law/
│ ├── raw_policies/
│ ├── processed/
│ └── annotated/
│
├── requirements.txt
└── README.md


---

## ⚙️ Pipeline

Run everything from the project root:

```bash
python src/extract_gdpr_sections.py
python src/build_dataset.py
python src/semantic_linker.py
python src/combine_scores_LLM.py
🔄 Pipeline Steps
1. Extract GDPR Articles
Input: data/raw_law/Reference Law - GDPR.txt
Output: data/reference_law_articles.csv

Parses GDPR into structured articles + recitals.

2. Build Policy Dataset
Input: data/raw_policies/*.txt
Output: data/processed/paragraphs.csv

Cleans and splits policies into paragraph-level data.

3. Semantic Linking
Input:
paragraphs.csv
reference_law_articles.csv
Output: paragraphs_with_articles.csv

Maps each paragraph to the most relevant GDPR article.

4. Hybrid Compliance Scoring
Output: combined_compliance_scores.csv

Final score combines:

Component	Weight
Heuristic (rules)	0.25
Semantic similarity	0.25
LLM verdict	0.50
🤖 LLM Integration

Uses local LLMs via Ollama
.

Default model
phi3:mini
Install models
ollama pull phi3:mini
ollama pull mistral
Run with different model
OLLAMA_MODEL=mistral python src/combine_scores_LLM.py
🧪 Testing & Experimentation

Run smaller subsets:

LIMIT_SOURCES=5 python src/combine_scores_LLM.py
LIMIT_ROWS=100 python src/combine_scores_LLM.py

Optional run naming:

RUN_NAME=test_phi3 python src/combine_scores_LLM.py
📦 Installation

Install dependencies:

pip install -r requirements.txt
Core dependencies
pandas
numpy
scikit-learn
matplotlib
regex
tqdm
python-dateutil
joblib
Optional (for HuggingFace models)
torch
transformers
accelerate
safetensors
⚠️ Important Notes
Always run scripts from the project root
Do not use absolute paths (/Users/...)
The pipeline is sequential — each step depends on the previous one
🧩 System Design Philosophy

This is not a single model — it’s a hybrid compliance system:

Heuristics → deterministic, interpretable
Semantic layer → contextual grounding
LLM → reasoning + explanation

The final output is:

explainable
auditable
closer to real-world compliance workflows
🛠️ Troubleshooting
FileNotFoundError

Make sure these exist:

data/reference_law_articles.csv
data/processed/paragraphs.csv
data/annotated/paragraphs_with_articles.csv
Ollama issues

Check:

ollama list

If missing:

ollama pull phi3:mini
📌 Future Improvements
Web interface integration (React / Flask)
Multi-model evaluation (benchmarking LLMs)
Fine-tuned compliance classifier
Document-type detection layer (policy vs non-policy)
📄 License

Academic use only (Thesis Project)

👤 Author

David Silva
MSc Data Science — ISCTE
