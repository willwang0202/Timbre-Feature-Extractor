"""
Timbre Feature Extractor — Gradio Space
https://huggingface.co/spaces/WCA0202/Timbre-Feature-Extractor

Single endpoint: analyze_audio
  Input:  audio file (filepath, Gradio-managed temp file)
  Output: JSON string with acoustic features + inferred emotion

Called from the Timbre web frontend during onboarding when a client
uploads a reference track. No CSV reads/writes — fully ephemeral.
"""

import json
import gradio as gr

from download_models import ensure_models
from feature_extractor import extract_features_for_file
from emotion_mapper import infer_emotion

# Download models on startup (no-op if already present)
print("Checking Essentia models...")
ensure_models()
print("Models ready.")


def analyze_audio(audio_filepath: str) -> str:
    """
    Extract acoustic features from an uploaded audio file and infer emotion.

    Returns JSON:
    {
      "emotion": "melancholic",
      "features": {
        "bpm": 76.0,
        "valence": 4.12,
        "arousal": 3.88,
        "mood_happy": 0.08,
        "mood_sad": 0.72,
        "mood_aggressive": 0.04,
        "mood_relaxed": 0.61,
        "mood_party": 0.03,
        "danceability": 0.18
      }
    }

    On error, returns JSON: { "error": "<message>" }
    """
    if audio_filepath is None:
        return json.dumps({"error": "No audio file provided."})

    try:
        features = extract_features_for_file(audio_filepath)
        emotion  = infer_emotion(features)
        return json.dumps({"emotion": emotion, "features": features})
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {e}"})


demo = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(
        type="filepath",
        label="Upload audio track (MP3, WAV, AAC, FLAC, M4A)",
    ),
    outputs=gr.Textbox(label="Features JSON"),
    title="Timbre Feature Extractor",
    description=(
        "Upload an audio file to extract acoustic features (BPM, valence, arousal, "
        "mood scores) and infer the emotion label. Used internally by the Timbre brief engine."
    ),
    api_name="analyze_audio",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
