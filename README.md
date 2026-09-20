# GDPR-AI-Agents

This repository contains the implementation and supporting research material for a Master's thesis focused on preliminary GDPR privacy policy assessment using a hybrid evidence-guided AI approach.

The project investigates how automated text analysis and Large Language Model (LLM) reasoning can support first-pass review of privacy policies, while maintaining traceability between identified disclosure issues and relevant GDPR requirements.

The system is designed as a research prototype and decision-support tool. It is not intended to replace legal advice, certify GDPR compliance, or assess the full GDPR compliance program of an organization.

---

## Thesis Focus

This project focuses on preliminary assessment of privacy policy transparency and information duties under the GDPR.

The system supports the review of privacy policies by helping users identify:

- Potentially missing disclosures
- Weak or vague explanations
- Relevant GDPR article associations
- Requirement-level coverage issues
- Suggested improvements for further professional review

The framework is intentionally limited to what can be inferred from the submitted privacy policy text. It does not verify whether an organization actually follows the practices described in the policy.

---

## System Overview

The prototype supports two assessment modes:

1. **LLM-only mode**
2. **Hybrid mode**

Both modes receive privacy policy text as input and return a structured assessment. The main difference is that the Hybrid mode generates intermediate evidence before the final assessment.

---

## LLM-only Mode

In the LLM-only mode, the full privacy policy is assessed directly by the model as a first-pass review.

This mode provides:

- Overall assessment status
- Preliminary privacy policy assessment score
- Summary of the assessment
- Strengths
- Findings
- Recommendations

This mode is designed to provide a faster and simpler assessment. It is useful for obtaining an initial overview of the policy, but it provides less intermediate traceability than the Hybrid mode.

---

## Hybrid Mode

The Hybrid mode combines several analysis layers before producing the final assessment:

1. Heuristic evidence extraction
2. Semantic GDPR article matching
3. Requirement-level coverage review
4. Evidence-guided LLM assessment

The heuristic and semantic layers generate structured evidence that is supplied to the model. This allows the final assessment to be grounded in intermediate signals such as policy excerpts, GDPR article matches, and requirement coverage indicators.

The Hybrid mode classifies key privacy policy requirements as:

- `PRESENT`
- `WEAK`
- `MISSING`

This makes the output more explainable and easier to inspect than a direct model-only response.

---

## Main Features

The prototype includes:

- Privacy policy text submission
- Document upload support
- LLM-only assessment mode
- Hybrid evidence-guided assessment mode
- Requirement-level coverage review
- GDPR article matching
- Structured JSON responses
- User-facing web interface
- Assessment report export
- Functional backend API
- Supporting material for the thesis literature review

---

## Repository Structure

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
│   └── Supporting material for the Systematic Literature Review
│
├── README.md
└── .gitignore
