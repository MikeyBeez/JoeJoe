import os
import asyncio
import speech_recognition as sr
import whisper
import warnings
import time
import sys
from ollama import AsyncClient
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
ollama_client = AsyncClient()
model_name = 'gemma:2b'

r = sr.Recognizer()
print("Using CPU for Whisper models")

# Load models silently
with suppress_stdout():
    tiny_model = whisper.load_model("tiny")
    base_model = whisper.load_model("base")

listening_for_wake_word = True
source = sr.Microphone(sample_rate=16000)

def speak(text):
    ALLOWED_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_$:+-/ ')
    clean_text = ''.join(c for c in text if c in ALLOWED_CHARS)
    os.system(f"say '{clean_text}'")

async def transcribe_audio(audio_file, model):
    with suppress_stdout():
        result = model.transcribe(audio_file)
    return result['text']

async def listen_for_wake_word(audio):
    global listening_for_wake_word
    with open('wake_detect.wav', 'wb') as f:
        f.write(audio.get_wav_data())
    text_input = await transcribe_audio('wake_detect.wav', tiny_model)
    if wake_word in text_input.lower().strip():
        print('Wake word detected. Please speak your prompt to Gemma.')
        speak('Listening')
        listening_for_wake_word = False

async def prompt_gemma(audio):
    global listening_for_wake_word
    try:
        with open('prompt.wav', 'wb') as f:
            f.write(audio.get_wav_data())
        prompt_text = await transcribe_audio('prompt.wav', base_model)
        if len(prompt_text.strip()) == 0:
            print('Empty prompt. Please speak again.')
            speak('Empty prompt. Please speak again.')
            listening_for_wake_word = True
        else:
            print('User:', prompt_text)
            response = await ollama_client.chat(model=model_name, messages=[
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
    finally:
        # Clear unnecessary variables
        import gc
        gc.collect()

async def process_audio(recognizer, audio):
    global listening_for_wake_word
    if listening_for_wake_word:
        await listen_for_wake_word(audio)
    else:
        await prompt_gemma(audio)

def callback(recognizer, audio):
    asyncio.run(process_audio(recognizer, audio))

async def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print('\nSay', wake_word, 'to wake me up. \n')
    r.listen_in_background(source, callback)
    while True:
        await asyncio.sleep(1)

if __name__ == '__main__':
    asyncio.run(start_listening())
