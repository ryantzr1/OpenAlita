import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
import base64

def transcribe_audio_openai(audio_path, api_key=None, api_url=None, model="gpt-4o-audio-preview"):
    """
    Transcribe an audio file using OpenAI GPT-4o Audio Model.
    Returns the transcribed text, or None on failure.
    """
    # Load API credentials
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = api_url or os.getenv("LLM_API_BASE", "https://oneapi.deepwisdom.ai/v1")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment or provided as argument.")

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Encode the audio file as base64
    try:
        with open(audio_path, "rb") as f:
            encoded_audio = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None

    # Call the API
    try:
        completion = client.chat.completions.create(
            model=model,
            modalities=["text"],  # Only want text output
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe what's in this recording."},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_audio,
                                "format": audio_path.split(".")[-1]  # e.g., mp3 or wav
                            }
                        }
                    ]
                }
            ]
        )

        # Extract transcript text
        transcript = completion.choices[0].message.content
        return transcript.strip() if transcript else None

    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None