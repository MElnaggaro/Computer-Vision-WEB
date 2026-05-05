from app.services.speech.speech_to_text import speech_to_text
from app.services.nlp.Question_Classification import predict_topic


def run_voice_nlp():
    print("🎤 Speak now...")

    text = speech_to_text()

    if text is None:
        print("❌ No speech detected")
        return

    print("📝 Text:", text)

    topic = predict_topic(text)

    print("🎯 Topic:", topic)


if __name__ == "__main__":
    run_voice_nlp()