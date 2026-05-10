"""
Question Classification Pipeline
==================================
Sentence Transformers + Calibrated Logistic Regression + Hybrid Keyword Routing
pipeline for classifying classroom questions into topic categories.

Supports training from a CSV dataset and inference with in-memory caching.

Train::

    python -m app.services.nlp.Question_Classification

Usage::

    from app.services.nlp.Question_Classification import predict_topic_with_confidence
    topic, conf = predict_topic_with_confidence("What is a semaphore?")
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.services.nlp.Feature_Extraction import build_feature_extractor
from app.services.nlp.Text_Preprocessing import clean_text
from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Module-level pipeline cache ──────────────────────────────────────
_cached_pipeline: Optional[Pipeline] = None
_cached_model_dir: Optional[str] = None

# ── Keyword Routing ──────────────────────────────────────────────────
KEYWORDS = {
    "Computer Networks": ["tcp", "udp", "routing", "ip", "subnet", "network", "osi", "protocol", "router", "switch", "handshake"],
    "Computer Organization and Architecture": ["cache", "pipeline", "register", "cpu", "instruction", "architecture", "memory", "processor", "alu"],
    "Operating System": ["semaphore", "deadlock", "paging", "thread", "process", "scheduler", "mutex", "kernel", "os", "virtual memory"],
    "Theory of Computation": ["dfa", "nfa", "automata", "grammar", "turing", "regular expression", "context free", "language"],
    "Programming and Data Structure": ["stack", "queue", "linked list", "binary tree", "array", "pointer", "recursion", "graph", "sort", "tree"],
    "Mathematics": ["matrix", "vector", "linear transformation", "derivative", "integral", "probability", "statistics", "calculus", "algebra", "eigen"],
    "Digital Logic": ["truth table", "flip flop", "mux", "demux", "logic gate", "boolean", "circuit", "k-map", "multiplexer", "latch"]
}

def get_keyword_boosts(text: str, classes: list) -> dict:
    """Calculates keyword-based probability boosts for each class."""
    text_lower = text.lower()
    boosts = {c: 0.0 for c in classes}
    
    for cls, words in KEYWORDS.items():
        if cls in boosts:
            for w in words:
                # Match full words/phrases to avoid partial matches like 'ip' in 'pipeline'
                if re.search(r'\b' + re.escape(w) + r'\b', text_lower):
                    boosts[cls] += 0.15  # Boost per keyword found
    return boosts

# ── Training ─────────────────────────────────────────────────────────

def train_and_save(
    data_path: Union[str, Path] = settings.NLP_DATASET_PATH,
    model_dir: Union[str, Path] = settings.NLP_MODEL_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    global _cached_pipeline, _cached_model_dir

    data_path = Path(data_path)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if "question" not in df.columns or "topic" not in df.columns:
        raise ValueError("CSV must contain 'question' and 'topic' columns.")

    df = df.dropna(subset=["question", "topic"])
    import numpy as np
    X, y = df["question"].tolist(), np.array(df["topic"].tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Base estimator: Logistic Regression
    base_lr = LogisticRegression(
        max_iter=1000,
        C=5.0,
        solver="lbfgs",
        random_state=random_state,
        class_weight="balanced"
    )
    
    # Calibrated Classifier
    calibrated_clf = CalibratedClassifierCV(base_lr, method='sigmoid', cv=5)

    pipeline = Pipeline(
        [
            ("embedder", build_feature_extractor()),
            ("clf", calibrated_clf),
        ]
    )

    logger.info("[NLP] Training Semantic Pipeline...")
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)

    pipeline_path = model_dir / "nlp_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)

    _cached_pipeline = pipeline
    _cached_model_dir = str(model_dir)

    logger.info("[NLP] Model saved -> %s  |  test accuracy: %.4f", pipeline_path, accuracy)
    print(f"[NLP] Model saved -> {pipeline_path}  |  test accuracy: {accuracy:.4f}")
    return {"accuracy": accuracy, "pipeline_path": str(pipeline_path)}

# ── Pipeline loading (with caching) ──────────────────────────────────

def _load_pipeline(model_dir: Union[str, Path] = settings.NLP_MODEL_DIR) -> Pipeline:
    global _cached_pipeline, _cached_model_dir
    import warnings
    import sklearn

    model_dir_str = str(Path(model_dir))

    if _cached_pipeline is not None and _cached_model_dir == model_dir_str:
        return _cached_pipeline

    path = Path(model_dir) / "nlp_pipeline.joblib"

    if path.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pipeline = joblib.load(path)
            except Exception as exc:
                logger.warning("[NLP] Failed to load cached model: %s", exc)
                pipeline = None

        if pipeline is not None:
            try:
                # Quick test
                pipeline.predict(["test question"])
                _cached_pipeline = pipeline
                _cached_model_dir = model_dir_str
                logger.info("[NLP] Pipeline loaded from %s (sklearn %s)", path, sklearn.__version__)
                return _cached_pipeline
            except Exception as exc:
                logger.warning(
                    "[NLP] Cached model incompatible (sklearn %s): %s — rebuilding …",
                    sklearn.__version__, exc,
                )

    # Auto-rebuild
    logger.info("[NLP] Rebuilding model from dataset …")
    data_path = settings.NLP_DATASET_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"No trained model at {path} and no dataset at {data_path}. "
            "Cannot build NLP pipeline."
        )

    result = train_and_save(data_path=data_path, model_dir=model_dir)
    logger.info("[NLP] Model rebuilt — accuracy: %.4f", result["accuracy"])
    return _cached_pipeline

def clear_cache() -> None:
    global _cached_pipeline, _cached_model_dir
    _cached_pipeline = None
    _cached_model_dir = None

# ── Inference ────────────────────────────────────────────────────────

def predict_topic(
    question: str,
    model_dir: Union[str, Path] = settings.NLP_MODEL_DIR,
) -> str:
    topic, _ = predict_topic_with_confidence(question, model_dir)
    return topic

def predict_topic_with_confidence(
    question: str,
    model_dir: Union[str, Path] = settings.NLP_MODEL_DIR,
) -> Tuple[str, float]:
    
    pipeline = _load_pipeline(model_dir)
    
    cleaned_for_kw = clean_text(question)
    
    classes = list(pipeline.classes_)
    
    # Base Semantic Probabilities
    base_probs = pipeline.predict_proba([question])[0]
    probs_dict = dict(zip(classes, base_probs))
    
    # Keyword Boosts
    boosts = get_keyword_boosts(cleaned_for_kw, classes)
    
    # Hybrid Top-K Fusion
    for cls in probs_dict:
        probs_dict[cls] += boosts.get(cls, 0.0)
        
    # Re-normalize
    total_prob = sum(probs_dict.values())
    if total_prob > 0:
        for cls in probs_dict:
            probs_dict[cls] /= total_prob
            
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    top_1_cls, top_1_prob = sorted_probs[0]
    
    # Uncertainty Handling
    if top_1_prob < 0.30:
        final_topic = "Uncertain"
        final_confidence = top_1_prob
    else:
        final_topic = top_1_cls
        final_confidence = top_1_prob
    
    print("\n" + "="*50)
    print("NLP DEBUG INFO (HYBRID V2)")
    print("="*50)
    print(f"Original text:\n\"{question}\"\n")
    print(f"Cleaned text:\n\"{cleaned_for_kw}\"\n")
    print(f"Predicted topic:\n{final_topic}\n")
    print(f"Confidence:\n{final_confidence:.4f}\n")
    
    print("Top candidates (Hybrid):")
    for cls, prob in sorted_probs[:3]:
        print(f"  {cls}: {prob:.4f} (Boost: {boosts.get(cls, 0.0):.2f})")
    print("="*50 + "\n")
    
    return final_topic, float(final_confidence)

# ── CLI entry-point ──────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = train_and_save()
    print(f"\nAccuracy: {result['accuracy']:.4f}")

    demo_questions = [
        "What is tcp handshake",
        "What is semaphore",
        "What is linear transformation",
        "What is finite automata",
        "What is linked list",
        "What is truth table"
    ]
    print("\n--- Demo Predictions ---")
    for q in demo_questions:
        topic, conf = predict_topic_with_confidence(q)
        print(f"  Q: {q}")
        print(f"  -> {topic} ({conf:.0%})\n")
