import re

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}",
    "phone": r"\b(?:\+?\d{1,3}[-. (]*)?\d{3,4}[-. )]?\d{3,4}[-. ]?\d{3,4}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "postal_code": r"\b\d{4,5}(?:[-\s]\d{3,4})?\b",
    "name": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # basic first+last name
}

class SensitiveDataAgent:
    def detect(self, text):
        results = {}
        for label, pattern in PII_PATTERNS.items():
            found = re.findall(pattern, text)
            if found:
                results[label] = list(set(found))
        return results

class PolicyAuditAgent:
    def __init__(self):
        self.rules = [
    # Core GDPR principles
    "lawful", "legitimate", "consent", "withdraw",
    "purpose", "storage", "retention", "controller", "processor",
    "rights", "access", "erasure", "rectification", "portability",
    "transfer", "supervisory authority", "complaint",
    "data protection", "privacy notice", "profiling",
]

    def evaluate(self, text):
        return {r: 1.0 if r in text else 0.0 for r in self.rules}
