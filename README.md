# GDPR-AI-Agents

This repository contains the implementation and supporting research material for a Master's thesis focused on automated GDPR compliance assessment using a hybrid AI-driven approach.

The system combines:

- Rule-based GDPR compliance checks
- Semantic retrieval of relevant GDPR articles
- Large Language Model (LLM) reasoning
- Explainable compliance recommendations

The objective is to support transparent and scalable privacy policy assessment while maintaining traceability between detected issues and GDPR requirements.

---

# Repository Structure

```text
GDPR-AI-Agents/
│
├── Code/
│   ├── api.py
│   ├── requirements.txt
│   ├── src/
│   └── data/
│
├── Frontend/
│   └── React + Vite web application
│
├── SLR/
│
├── README.md
└── .gitignore
```

---

# System Architecture

The project supports two analysis modes.

## LLM-Only Mode

The entire privacy policy is evaluated directly by an LLM.

Features:

- Whole-policy assessment
- GDPR compliance score
- Compliance findings
- Improvement recommendations

This mode provides faster assessments and serves as a baseline approach.

---

## Hybrid Mode

The hybrid mode combines multiple analysis layers:

1. Heuristic GDPR compliance checks
2. Semantic GDPR article matching
3. LLM-based compliance reasoning

The heuristic and semantic layers generate structured evidence that is supplied to the LLM to support explainable compliance assessments.

Features:

- Evidence-guided analysis
- GDPR article traceability
- Explainable recommendations
- Improved transparency of decision-making

---

# Backend

The backend is implemented using Flask.

Main entry point:

```bash
python api.py
```

The API exposes a GDPR compliance assessment endpoint:

```http
POST /api/check-compliance
```

Example request:

```json
{
  "policy_text": "Privacy policy text here",
  "mode": "llm_only"
}
```

or

```json
{
  "policy_text": "Privacy policy text here",
  "mode": "hybrid"
}
```

---

# Frontend

The frontend is implemented using:

- React
- TypeScript
- Vite

The interface allows users to:

- Submit privacy policies
- Select analysis mode
- View compliance scores
- Review findings and recommendations

---

# Data Pipeline

The repository also contains the original GDPR processing pipeline used during development and experimentation.

Core components include:

| File | Purpose |
|--------|---------|
| extract_gdpr_sections.py | Extracts GDPR articles from source documents |
| build_dataset.py | Builds the privacy policy dataset |
| semantic_linker.py | Links policy content to GDPR articles |
| gdpr_agent.py | Implements heuristic GDPR checks |
| combine_scores_LLM.py | Combines heuristic, semantic, and LLM outputs |

Generated outputs include:

- GDPR article database
- Processed privacy policy dataset
- GDPR article mappings
- Compliance assessment results

---

# Requirements

Install backend dependencies:

```bash
cd Code
pip install -r requirements.txt
```

Key libraries include:

- pandas
- numpy
- scikit-learn
- Flask
- google-genai
- matplotlib

---

# Systematic Literature Review (SLR)

The `SLR/` directory contains the study database used during the thesis literature review.

Studies are organised into three categories:

| Folder | Description |
|----------|------------|
| 1. Contextual | Background and contextual studies |
| 2. Supporting | Studies related to GDPR, NLP, compliance automation, and AI governance |
| 3. Core | Studies most directly aligned with the thesis research question |

The inclusion of the SLR database promotes transparency and traceability of the literature review process.

---

# Thesis Focus

This project investigates how hybrid AI architectures can support GDPR compliance assessment through:

- Explainable compliance reasoning
- Semantic retrieval of legal requirements
- Automated privacy policy analysis
- Human-readable compliance recommendations

The system is designed as a research prototype and educational tool rather than a substitute for legal advice.
