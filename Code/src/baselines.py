from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

def train_baseline_model(df):
    """
    Train a baseline TF-IDF + Logistic Regression classifier.
    Saves the model under <project_root>/models/compliance_model.pkl
    """

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42
    )

    # TF-IDF vectorisation
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    # Evaluate on test set
    print(classification_report(y_test, clf.predict(X_test_vec)))

    # ✅ Always resolve path relative to project root
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "compliance_model.pkl"

    # Save model
    joblib.dump((vectorizer, clf), model_path)
    print(f"✅ Model saved successfully at: {model_path}")

    return vectorizer, clf
