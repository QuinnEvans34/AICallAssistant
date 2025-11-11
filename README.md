# CallAssist-LLM

A live-streaming Q&A assistant for solar sales calls, built with Python. It transcribes audio in real-time, detects questions dynamically, matches them to a knowledge base, and generates natural LLM-enhanced responses using Ollama's Mistral model.

## Features

- **Real-time ASR**: Continuous audio transcription using Faster-Whisper.
- **Question Detection**: Automatically identifies questions in live speech.
- **LLM Integration**: Generates contextual, natural responses via Ollama Mistral.
- **Modern UI**: Flutter-based interface with Flet for live updates.
- **Offline Operation**: Runs entirely locally with Ollama.
- **Context Awareness**: Maintains conversation history for coherent responses.

## Requirements

- Python 3.8+
- Ollama with Mistral model (`ollama pull mistral`)
- Microphone access

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd CallAssist-LLM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Ollama and pull Mistral:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull mistral
   ```

## Usage

1. Start Ollama server:
   ```bash
   ollama serve
   ```

2. Run the application:
   ```bash
   python main.py
   ```

3. In the UI:
   - Enter caller name (optional).
   - Click "Start Live" to begin listening.
   - Speak questions; responses appear in real-time.

## Architecture

- **stream_asr.py**: Handles audio streaming and transcription.
- **question_detector.py**: Processes transcripts for questions.
- **response_manager.py**: Matches questions and generates responses.
- **llm.py**: Interfaces with Ollama API.
- **ui_flet.py**: Modern UI with live updates.
- **matcher.py**: Fuzzy matching against Q&A database.

## Configuration

- Q&A data: `qa_script.json`
- Persona: `persona.txt`
- Adjust parameters in code for sensitivity/speed.

## License

[Add license if applicable]