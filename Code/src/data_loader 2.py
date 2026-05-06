from pathlib import Path
import pandas as pd

def load_texts(folder: Path, label: str):
    files = list(folder.glob("*.txt"))
    return [{"file_name": f.name, "text": f.read_text(encoding="utf-8"), "label": label} for f in files]

def load_datasets(consent_dir, reference_dir):
    data = []
    data += load_texts(consent_dir, "consent_form")
    data += load_texts(reference_dir, "reference_law")
    return pd.DataFrame(data)

def load_text_file(path):
    """Reads a plain .txt file and returns its content as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()