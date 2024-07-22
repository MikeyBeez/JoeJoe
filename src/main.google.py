import os
import sys
import asyncio
import warnings
import google.generativeai as genai
import speech_recognition as sr
import whisper
import subprocess
import threading
from queue import Queue
import re

# Redirect stderr to /dev/null
stderr_fd = sys.stderr.fileno()
with open(os.devnull, 'w') as devnull:
    os.dup2(devnull.fileno(), stderr_fd)

# Suppress warnings
warnings.filterwarnings("ignore")

# Custom print function
def custom_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# Get API key from environment variable
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    custom_print("Error: GOOGLE_API_KEY not found in environment variables")
    sys.exit(1)

# Configure the Gemini API
genai.configure(api_key=api_key)

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')

# Initialize speech recognizer and Whisper model
recognizer = sr.Recognizer()
whisper_model = whisper.load_model("base")

def clean_text(text):
    # Remove asterisks and octothorps
    return re.sub(r'[*#]', '', text)

def speak(text):
    try:
        subprocess.run(["say", text], check=True, capture_output=True, text=True)
    except:
        pass

def queue_and_speak(speech_queue):
    while True:
        sentence = speech_queue.get()
        if sentence is None:
            break
        speak(sentence)
        speech_queue.task_done()

async def listen_for_input(timeout=10):
    with sr.Microphone() as source:
        try:
            audio = await asyncio.to_thread(recognizer.listen, source, timeout=timeout)
            with open("input.wav", "wb") as f:
                f.write(audio.get_wav_data())
            result = await asyncio.to_thread(whisper_model.transcribe, "input.wav")
            os.remove("input.wav")
            return result["text"].strip()
        except:
            return None

def extract_prompt(text, wake_phrases):
    text_lower = text.lower()
    for phrase in wake_phrases:
        if text_lower.startswith(phrase):
            return text[len(phrase):].strip()
    return None

async def main():
    wake_phrases = ["hey assistant", "okay assistant", "hi assistant"]
    custom_print(f"Say one of {wake_phrases} followed by your question, or 'quit' to exit.")

    speech_queue = Queue()
    speech_thread = threading.Thread(target=queue_and_speak, args=(speech_queue,))
    speech_thread.start()

    try:
        while True:
            text = await listen_for_input()
            if text:
                prompt = extract_prompt(text, wake_phrases)
                if prompt:
                    custom_print(f"You{prompt}")
                    try:
                        response = model.generate_content(prompt, stream=True)
                        
                        current_sentence = ""
                        for chunk in response:
                            if chunk.text:
                                current_sentence += chunk.text
                                sentences = current_sentence.split('\n')
                                
                                for sentence in sentences[:-1]:
                                    complete_sentence = sentence.strip()
                                    if complete_sentence:
                                        custom_print(complete_sentence)
                                        cleaned_sentence = clean_text(complete_sentence)
                                        speech_queue.put(cleaned_sentence)
                                
                                current_sentence = sentences[-1]
                                
                                if current_sentence.strip().endswith('.'):
                                    custom_print(current_sentence)
                                    cleaned_sentence = clean_text(current_sentence)
                                    speech_queue.put(cleaned_sentence)
                                    current_sentence = ""

                            await asyncio.sleep(0)

                        if current_sentence:
                            custom_print(current_sentence)
                            cleaned_sentence = clean_text(current_sentence)
                            speech_queue.put(cleaned_sentence)
                        
                    except Exception as e:
                        custom_print("Sorry, I couldn't generate a response.")
                        speech_queue.put("Sorry, I couldn't generate a response.")
                elif "quit" in text.lower():
                    custom_print("Goodbye!")
                    speech_queue.put("Goodbye!")
                    break
                else:
                    custom_print("Wake word not detected. Please start with a wake phrase.")
    except Exception as e:
        custom_print("An error occurred. Restarting the assistant.")
    finally:
        speech_queue.put(None)
        speech_thread.join()

if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            custom_print("\nProgram terminated by user.")
            break
        except Exception:
            custom_print("An unexpected error occurred. Restarting the assistant.")
