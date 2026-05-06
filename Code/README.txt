GDPR Compliance Agents — Run Order & Requirements

Exact run order (from project root):

1) python src/build_dataset.py
2) python src/extract_gdpr_sections.py
3) python src/semantic_linker.py
4) python src/combine_scores_LLM.py

Notes:
- The pipeline uses a local LLM via Ollama.
- Default model: phi3:mini (you can switch to mistral by setting OLLAMA_MODEL=mistral).
- Outputs are written under data/processed and data/annotated.

Requirements (install with: pip install -r requirements.txt):
- pandas
- numpy
- scikit-learn
- matplotlib
- regex
- tqdm (optional)
- python-dateutil (optional)

Ollama models (install separately via Ollama):
- ollama pull phi3:mini
- or: ollama pull mistral
