import os
import requests
from dotenv import load_dotenv

def some_utility_function(param1, param2):
    return param1 + param2

def another_utility_function(data):
    return [item for item in data if item is not None]

def transcribe_audio_openai(audio_path, api_key=None, model="whisper-1", language=None):
    """
    Transcribe an audio file using OpenAI Whisper API.
    Returns the transcribed text, or None on failure.
    """
    load_dotenv()
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment or provided as argument.")
    
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": (os.path.basename(audio_path), open(audio_path, "rb"), "application/octet-stream")}
    data = {"model": model}
    if language:
        data["language"] = language
    
    try:
        response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        if response.status_code == 200:
            return response.json().get("text", None)
        else:
            print(f"OpenAI API error: {response.status_code} {response.text}")
            return None
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None