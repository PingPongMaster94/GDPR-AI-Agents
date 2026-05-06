# GDPR-AI-Agents

This repository contains the implementation code and supporting study database for a Master's thesis on automated GDPR compliance assessment using AI agents, semantic retrieval, and local LLM reasoning.

The repository has two main components:

- `Code/` — implementation of the GDPR compliance checking pipeline
- `SLR/` — database of studies used for the Systematic Literature Review (SLR)

---

## Repository Structure

```text
GDPR-AI-Agents/
│
├── Code/
│   ├── src/
│   ├── data/
│   ├── notebooks/
│   └── requirements.txt
│
├── SLR/
│   ├── 1. Contextual/
│   ├── 2. Supporting/
│   └── 3. Core/
│
├── README.md
└── .gitignore
Code Component

The Code/ folder contains the automated GDPR compliance evaluation pipeline.

Core Python Files
File	Purpose
extract_gdpr_sections.py	Extracts GDPR articles and recitals from the raw GDPR reference text.
build_dataset.py	Loads raw privacy policies, cleans them, and splits them into paragraph-level records.
semantic_linker.py	Links each policy paragraph to the most relevant GDPR article using TF-IDF cosine similarity.
gdpr_agent.py	Implements the rule-based GDPR heuristic checklist and scoring logic.
combine_scores_LLM.py	Combines heuristic score, semantic score, and LLM verdict into the final compliance score.
cleaning_utils.py	Provides text cleaning and preprocessing utilities.
data_loader.py	Provides helper functions for loading local text datasets.
Pipeline Run Order

Run the pipeline from the repository root:

python Code/src/extract_gdpr_sections.py
python Code/src/build_dataset.py
python Code/src/semantic_linker.py
python Code/src/combine_scores_LLM.py

The expected flow is:

raw GDPR text
→ structured GDPR articles
→ raw privacy policies
→ paragraph-level dataset
→ semantic GDPR article linking
→ hybrid compliance scoring
→ final compliance results
Pipeline Outputs
Step	Output
GDPR extraction	Code/data/processed/reference_law_articles.csv
Dataset construction	Code/data/processed/paragraphs.csv
Semantic linking	Code/data/annotated/paragraphs_with_articles.csv
Final scoring	Code/data/results/compliance_results.csv
Hybrid Scoring Logic

The final compliance score combines:

Component	Weight
Heuristic GDPR checklist	0.25
Semantic article similarity	0.25
LLM compliance verdict	0.50

The LLM is used as an auditing component rather than as the only decision mechanism.

SLR Component

The SLR/ folder contains the study database used for the Systematic Literature Review phase of the thesis.

Studies are organised into three categories:

Folder	Meaning
1. Contextual/	Background and contextual studies.
2. Supporting/	Supporting studies relevant to automated compliance, GDPR, NLP, and AI governance.
3. Core/	Core studies most directly aligned with the thesis topic.

This folder is included to make the literature selection transparent and traceable.

Requirements

Install dependencies from inside the Code/ folder:

cd Code
pip install -r requirements.txt

Core dependencies include:

pandas
numpy
scikit-learn
matplotlib
regex
tqdm
python-dateutil
joblib

Ollama must be installed separately for local LLM execution.

Example:

ollama pull phi3:mini
ollama pull mistral
Notes

This repository supports a thesis project on transparent and explainable GDPR compliance automation. The implementation is designed as a hybrid system combining deterministic rules, semantic retrieval, and LLM-based reasoning.


## 3. Add both folders

From your local repo root:

```bash
cd "/Users/davidj.silva/Desktop/Data Science/Thesis/GDPR-AI-Agents"

Then:

git add Code SLR README.md .gitignore
git commit -m "Upload code and SLR study database"