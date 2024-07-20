import os
import speech_recognition as sr
import whisper
import warnings
import time
import sys
from ollama import Client
import contextlib

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress stdout temporarily


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


wake_word = 'jarvis'
ollama_client = Client()
model_name = 'gemma:2b'

r = sr.Recognizer()
tiny_model_path = os.path.expanduser('~/.cache/whisper/tiny.pt')
base_model_path = os.path.expanduser('~/.cache/whisper/base.pt')

# Load models silently
with suppress_stdout():
    if not os.path.exists(tiny_model_path):
        print("Downloading Whisper tiny model...")
        tiny_model = whisper.load_model("tiny",
            download_root=os.path.expanduser('~/.cache/whisper/'))
    else:
        tiny_model = whisper.load_model(tiny_model_path)

    if not os.path.exists(base_model_path):
        print("Downloading Whisper base model...")
        base_model = whisper.load_model("base", download_root=os.path.expanduser('~/.cache/whisper/'))
    else:
        base_model = whisper.load_model(base_model_path)

listening_for_wake_word = True
source = sr.Microphone()

if sys.platform != 'darwin':
    import pyttsx3
    engine = pyttsx3.init()

def speak(text):
    if sys.platform == 'darwin':
        ALLOWED_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_$:+-/ ')
        clean_text = ''.join(c for c in text if c in ALLOWED_CHARS)
        os.system(f"say '{clean_text}'")
    else:
        engine.say(text)
        engine.runAndWait()

def listen_for_wake_word(audio):
    global listening_for_wake_word
    with open('wake_detect.wav', 'wb') as f:
        f.write(audio.get_wav_data())
    with suppress_stdout():
        result = tiny_model.transcribe('wake_detect.wav')
    text_input = result['text']
    if wake_word in text_input.lower().strip():
        print('Wake word detected. Please speak your prompt to Gemma.')
        speak('Listening')
        listening_for_wake_word = False

def prompt_gemma(audio):
    global listening_for_wake_word
    try:
        with open('prompt.wav', 'wb') as f:
            f.write(audio.get_wav_data())
        with suppress_stdout():
            result = base_model.transcribe('prompt.wav')
        prompt_text = result['text']
        if len(prompt_text.strip()) == 0:
            print('Empty prompt. Please speak again.')
            speak('Empty prompt. Please speak again.')
            listening_for_wake_word = True
        else:
            print('User:', prompt_text)
            response = ollama_client.chat(model=model_name, messages=[
                {
                    'role': 'user',
                    'content': prompt_text
                }
            ])
            output = response['message']['content']
            print('Gemma:', output)
            speak(output)
            print('\nSay', wake_word, 'to wake me up. \n')
            listening_for_wake_word = True
    except Exception as e:
        print('Prompt error: ', e)

def callback(recognizer, audio):
    global listening_for_wake_word
    if listening_for_wake_word:
        listen_for_wake_word(audio)
    else:
        prompt_gemma(audio)

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print('\nSay', wake_word, 'to wake me up. \n')
    r.listen_in_background(source, callback)
    while True:
        time.sleep(1)

if __name__ == '__main__':
    start_listening()
