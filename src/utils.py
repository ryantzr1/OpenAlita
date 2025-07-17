import os
import requests
from dotenv import load_dotenv
import base64
from openai import OpenAI

def transcribe_audio_openai(audio_path, api_key=None, api_url=None, model="gpt-4o-audio-preview"):
    """
    Transcribe an audio file using OpenAI GPT-4o Audio Model.
    Converts unsupported formats to mp3 automatically.
    Returns the transcribed text, or None on failure.
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = api_url or os.getenv("LLM_API_BASE", "https://oneapi.deepwisdom.ai/v1")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment or provided as argument.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    ext = audio_path.split(".")[-1].lower()

    # If not supported audio format, convert to mp3
    if ext not in {"mp3", "wav"}: # API only supports mp3 and wav currently
        import subprocess
        import tempfile
        import ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            converted_path = tmp_file.name
        try:
            ffmpeg.input(audio_path).output(converted_path, format="mp3").run(overwrite_output=True, quiet=True)
            audio_path_to_use = converted_path
            ext = "mp3"
        except ffmpeg.Error as e:
            print(f"Error converting {ext} to mp3: {e}")
            return None
    else:
        audio_path_to_use = audio_path

    try:
        with open(audio_path_to_use, "rb") as f:
            encoded_audio = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading audio file: {e}")
        if audio_path_to_use != audio_path:
            os.unlink(audio_path_to_use)
        return None

    try:
        completion = client.chat.completions.create(
            model=model,
            modalities=["text"],  # Only want text output
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe what's in this speech recording. If it's not a speech recording, analyze in detail what is in the audio."},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_audio,
                                "format": ext
                            }
                        }
                    ]
                }
            ]
        )

        transcript = completion.choices[0].message.content
        return transcript.strip() if transcript else None

    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

    finally:
        # Clean up the temporary mp3 file if created
        if audio_path_to_use != audio_path:
            try:
                os.unlink(audio_path_to_use)
            except Exception:
                pass
