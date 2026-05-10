from sklearn.base import BaseEstimator, TransformerMixin
import logging
import os

logger = logging.getLogger(__name__)

class SentenceTransformerExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts semantic text embeddings using sentence-transformers.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model_ = None

    def fit(self, X, y=None):
        if self.model_ is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"[NLP] Loading SentenceTransformer: {self.model_name}")
                self.model_ = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        return self

    def transform(self, X):
        if self.model_ is None:
            self.fit(X)
        # Return numpy array of embeddings
        return self.model_.encode(X, show_progress_bar=False)

def build_feature_extractor(model_name="all-MiniLM-L6-v2") -> SentenceTransformerExtractor:
    """
    Return a scikit-learn compatible transformer for semantic embeddings.
    """
    return SentenceTransformerExtractor(model_name=model_name)