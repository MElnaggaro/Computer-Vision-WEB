from sklearn.feature_extraction.text import TfidfVectorizer
 
from app.services.nlp.Text_Preprocessing import clean_text
 
 
def build_tfidf_vectorizer(
    max_features: int = 10_000,
    ngram_range: tuple = (1, 2),
    sublinear_tf: bool = True,
) -> TfidfVectorizer:

    #Return a TfidfVectorizer configured to use clean_text as its analyser.
 
        #max_features:  Vocabulary capacity.
        #ngram_range:   Unigram + bigram by default.
        #sublinear_tf:  Apply log(1 + tf) scaling for better signal on long texts.
 

    return TfidfVectorizer(
        analyzer="word",
        preprocessor=clean_text,
        tokenizer=str.split,
        token_pattern=None,
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
    )