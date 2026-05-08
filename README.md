---
title: Timbre Feature Extractor
emoji: 🎵
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.25.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Timbre Feature Extractor

Extracts acoustic features from an uploaded audio file using Essentia, and infers an emotion label. Used by the Timbre brief engine when a client uploads a reference track during onboarding.

**Space**: `WCA0202/Timbre-Feature-Extractor`
**GitHub**: `willwang0202/Timbre-Feature-Extractor`

---

## API

Single endpoint exposed via the Gradio SSE pattern:

```
POST /call/analyze_audio
Body: { "data": [<audio file handle>] }
→ { "event_id": "abc123" }

GET /call/analyze_audio/{event_id}
→ SSE stream — read "event: complete" then the next "data:" line
```

### Response

```json
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
```

On error: `{ "error": "<message>" }`

### Feature definitions

| Feature | Range | Source |
|---------|-------|--------|
| `bpm` | 40–220 | PercivalBpmEstimator (44.1 kHz audio) |
| `valence` | 1–9 | DEAM model on MusiCNN embeddings (16 kHz) |
| `arousal` | 1–9 | DEAM model on MusiCNN embeddings (16 kHz) |
| `mood_happy/sad/aggressive/relaxed/party` | 0–1 | Classification heads on Discogs-EffNet embeddings |
| `danceability` | 0–1 | Danceability head on Discogs-EffNet embeddings |

---

## Models

Downloaded automatically on startup to `./models/` (~23 MB total). Source: `https://essentia.upf.edu/models`.

| File | Purpose |
|------|---------|
| `discogs-effnet-bs64-1.pb` | Backbone for mood + danceability heads |
| `msd-musicnn-1.pb` | Backbone for valence/arousal |
| `deam-msd-musicnn-2.pb` | Valence + arousal regression (DEAM dataset) |
| `mood_happy/sad/aggressive/relaxed/party-discogs-effnet-1.pb` | Mood classifiers |
| `danceability-discogs-effnet-1.pb` | Danceability classifier |

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio interface — `analyze_audio` endpoint, model warmup |
| `feature_extractor.py` | Essentia pipeline — lazy model loader, `extract_features_for_file()` |
| `emotion_mapper.py` | Rule-based emotion classifier using baked-in quantile thresholds |
| `download_models.py` | Downloads Essentia `.pb` models on startup; skips existing files |
| `requirements.txt` | `gradio==5.25.0`, `essentia-tensorflow`, `numpy` |

---

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py   # http://localhost:7860
```

Models are downloaded automatically on first run (~23 MB, one-time).

---

## Deploying

Push to both remotes to keep GitHub and HF Space in sync:

```bash
git push origin main   # → GitHub
git push hf main       # → HF Space
```
