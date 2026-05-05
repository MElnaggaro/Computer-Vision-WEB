#Run with:  python -m pytest tests/test_nlp.py -v
import pytest
 
from app.services.nlp.Text_Preprocessing import clean_text
from app.services.nlp.Question_Classification import train_and_save, predict_topic
 
# Fixtures
 
@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """Train the pipeline once per test session into a temp directory."""
    model_dir = tmp_path_factory.mktemp("models")
    result = train_and_save(model_dir=model_dir)
    return {"model_dir": model_dir, "accuracy": result["accuracy"]}
 
 

# Test 1 — clean_text preprocessing contract
 
class TestCleanText:
    def test_lowercases_and_strips_punctuation(self):
        raw = "What is the TIME complexity of QuickSort?!"
        cleaned = clean_text(raw)
        assert cleaned == cleaned.lower(), "Output must be lowercase"
        for ch in "?!.,":
            assert ch not in cleaned, f"Punctuation '{ch}' must be removed"
 
    def test_removes_stopwords(self):
        raw = "what is the best way to sort a list"
        cleaned = clean_text(raw)
        stopwords_present = {"what", "is", "the", "to", "a"} & set(cleaned.split())
        assert not stopwords_present, f"Stopwords still present: {stopwords_present}"
 
    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""
 
 

# Test 2 — end-to-end classification accuracy & predict_topic contract

class TestClassification:
    def test_training_accuracy_above_threshold(self, trained_model):
        #Pipeline must reach at least 85 % test accuracy on the held-out set.
        assert trained_model["accuracy"] >= 0.85, (
            f"Accuracy {trained_model['accuracy']:.4f} is below the 0.85 threshold"
        )
 
    @pytest.mark.parametrize(
        "question, expected_topic",
        [
            (
                "What is the probability of flipping a coin and getting heads three times in a row?",
                "Mathematics",
            ),
            (
                "What is the function of the ALU in a CPU architecture?",
                "Computer Organization and Architecture",
            ),
            (
                "State the difference between a recursive and recursively enumerable language.",
                "Theory of Computation",
            ),
            (
                "How does the sliding window protocol work in flow control?",
                "Computer Networks",
            ),
            (
                "What is a semaphore and how is it used in process synchronization?",
                "Operating System",
            ),
            
        ],

    )
    def test_predict_topic_returns_correct_label(
        self, trained_model, question, expected_topic
    ):
        predicted = predict_topic(question, model_dir=trained_model["model_dir"])
        assert predicted == expected_topic, (
            f"Question: '{question}'\n"
            f"  Expected : {expected_topic}\n"
            f"  Got      : {predicted}"
        )
 
    def test_predict_topic_returns_string(self, trained_model):
        result = predict_topic(
            "Explain virtual memory management.",
            model_dir=trained_model["model_dir"],
        )
        assert isinstance(result, str) and len(result) > 0
 