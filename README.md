# JoeJoe
Voice activated Assistant
# Voice Assistant with Whisper and Gemma

This project implements a voice assistant using OpenAI's Whisper for speech recognition and Ollama's Gemma model for natural language processing.

## Features

- Wake word detection ("Hey Assistant")
- Speech-to-text conversion using Whisper
- Natural language processing using Gemma:2b
- Text-to-speech output
- Non-blocking response output and simultaneous threaded speaking

## Requirements

- Python 3.9+
- Ollama (with Gemma:2b model)
- OpenAI Whisper
- SpeechRecognition
- PyAudio
- pyttsx3 (for non-macOS systems)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/MikeyBeez/JoeJoe.git
   cd JoeJoe
   ```

2. Create and activate a Conda environment:
   ```
   conda create -n voice_assistant python=3.9
   conda activate voice_assistant
   ```

3. Install required packages:
   ```
   conda install -c conda-forge speechrecognition
   conda install -c conda-forge pyttsx3
   pip install ollama openai-whisper pyaudio
   ```

4. Install Ollama (See Ollama.com for instructions) and pull the Gemma:2b model:
   ```
   ollama pull gemma:2b
   ```

## Usage

1. Activate the Conda environment:
   ```
   conda activate voice_assistant
   ```

2. Run the script:
   ```
   python main.py
   ```

3. Say "jarvis" to wake up the assistant, then speak your query.

## Note

- Ensure your microphone is properly set up and recognized by your system.
- The first run may take some time as it downloads the Whisper models.
- This project runs Whisper on CPU. For better performance, consider using a GPU if available.

## License

MIT

