#speech recognition module

import speech_recognition as sr

def speech_to_text():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak now...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        try:
            audio = recognizer.listen(source, timeout=5)
        except sr.WaitTimeoutError:
            return None  # no speech detected

    try:
        text = recognizer.recognize_google(audio)
        return text.lower().strip()

    except sr.UnknownValueError:
        return None  # unclear audio

    except sr.RequestError:
        return None  # API issue


# test
if __name__ == "__main__":
    result = speech_to_text()

    if result is None:
        print("No valid speech detected")
    else:
        print("Result:", result)