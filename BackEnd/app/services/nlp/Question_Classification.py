#Run: python -m app.services.nlp.Question_Classification
from __future__ import annotations
 
import os
from pathlib import Path
 
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
 
from app.services.nlp.Feature_Extraction import build_tfidf_vectorizer
 
 
# Training
 
def train_and_save(
    data_path: str | Path = "data/nlp/raw/dataset.csv",
    model_dir: str | Path = "data/nlp/trained/models",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    

    # Train a TF-IDF → LogisticRegression pipeline on the CSV data and save it to *model_dir/nlp_pipeline.joblib*.
 
    # Returns:
    # dict with keys accuracy, pipeline_path.


    data_path = Path(data_path)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
 
    df = pd.read_csv(data_path)
    if "question" not in df.columns or "topic" not in df.columns:
        raise ValueError("CSV must contain 'question' and 'topic' columns.")
 
    df = df.dropna(subset=["question", "topic"])
    X, y = df["question"].tolist(), df["topic"].tolist()
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
 
    pipeline = Pipeline(
        [
            ("tfidf", build_tfidf_vectorizer()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=5.0,
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )
 
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
 
    pipeline_path = model_dir / "nlp_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
 
    print(f"[NLP] Model saved → {pipeline_path}  |  test accuracy: {accuracy:.4f}")
    return {"accuracy": accuracy, "pipeline_path": str(pipeline_path)}
 
 
# Inference

 
def _load_pipeline(model_dir: Path = "data/nlp/trained/models") -> Pipeline:
    path = model_dir / "nlp_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run train_and_save() first."
        )
    return joblib.load(path)
 
 
def predict_topic(question: str, model_dir: str | Path = "data/nlp/trained/models") -> str:
    #question:  Raw question string.
    #model_dir: Directory where *nlp_pipeline.joblib* lives.

    # Predict the topic for a single classroom question.
     
    pipeline = _load_pipeline(Path(model_dir))
    #Predicted topic label (e.g."Mathematics").
    return pipeline.predict([question])[0]
 
 

# CLI entry-point
 
if __name__ == "__main__":
    train_and_save()
 
