"""
Timbre Feature Extractor — single-file extraction logic.

Isolated from the batch pipeline. No CSV reads/writes, no filesystem side
effects. Called by app.py for on-demand audio analysis.

Models expected in ./models/ (same filenames as the main repo):
  discogs-effnet-bs64-1.pb
  msd-musicnn-1.pb
  deam-msd-musicnn-2.pb
  mood_happy-discogs-effnet-1.pb
  mood_sad-discogs-effnet-1.pb
  mood_aggressive-discogs-effnet-1.pb
  mood_relaxed-discogs-effnet-1.pb
  mood_party-discogs-effnet-1.pb
  danceability-discogs-effnet-1.pb
"""

import os
import contextlib
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ESSENTIA_LOG_LEVEL"] = "silent"
warnings.filterwarnings("ignore")

MODELS_DIR = "./models"

# ── Silence native stderr (TF/Essentia C++ noise) ─────────────────────────────

@contextlib.contextmanager
def _silence_stderr():
    original_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(original_fd, 2)
        os.close(original_fd)


# ── Lazy model loader — loaded once on first call ─────────────────────────────

_models = None

def _load_models():
    global _models
    if _models is not None:
        return _models

    import essentia
    essentia.log.warningActive = False
    essentia.log.infoActive = False

    from essentia.standard import (
        MonoLoader,
        TensorflowPredictEffnetDiscogs,
        TensorflowPredictMusiCNN,
        TensorflowPredict2D,
        PercivalBpmEstimator,
    )

    def pb(name):
        return os.path.join(MODELS_DIR, name)

    with _silence_stderr():
        _models = {
            "loader":  MonoLoader,
            "effnet":  TensorflowPredictEffnetDiscogs(
                           graphFilename=pb("discogs-effnet-bs64-1.pb"),
                           output="PartitionedCall:1"),
            "musicnn": TensorflowPredictMusiCNN(
                           graphFilename=pb("msd-musicnn-1.pb"),
                           output="model/dense/BiasAdd"),
            "deam":    TensorflowPredict2D(
                           graphFilename=pb("deam-msd-musicnn-2.pb"),
                           output="model/Identity"),
            "happy":   TensorflowPredict2D(
                           graphFilename=pb("mood_happy-discogs-effnet-1.pb"),
                           input="model/Placeholder", output="model/Softmax"),
            "sad":     TensorflowPredict2D(
                           graphFilename=pb("mood_sad-discogs-effnet-1.pb"),
                           input="model/Placeholder", output="model/Softmax"),
            "aggressive": TensorflowPredict2D(
                           graphFilename=pb("mood_aggressive-discogs-effnet-1.pb"),
                           input="model/Placeholder", output="model/Softmax"),
            "relaxed": TensorflowPredict2D(
                           graphFilename=pb("mood_relaxed-discogs-effnet-1.pb"),
                           input="model/Placeholder", output="model/Softmax"),
            "party":   TensorflowPredict2D(
                           graphFilename=pb("mood_party-discogs-effnet-1.pb"),
                           input="model/Placeholder", output="model/Softmax"),
            "dance":   TensorflowPredict2D(
                           graphFilename=pb("danceability-discogs-effnet-1.pb"),
                           input="model/Placeholder", output="model/Softmax"),
            "bpm":     PercivalBpmEstimator(),
        }
    return _models


# ── Public API ────────────────────────────────────────────────────────────────

def extract_features_for_file(filepath: str) -> dict:
    """
    Run Essentia feature extraction on a single audio file.

    Returns a plain dict with the same keys as song_features.csv
    (minus filename/title/language/country_code — those are metadata,
    not acoustic features).

    Raises ValueError if the file cannot be loaded or processed.
    """
    import numpy as np

    if not os.path.exists(filepath):
        raise ValueError(f"File not found: {filepath}")

    m = _load_models()

    try:
        loader = m["loader"]
        audio_16k = loader(filename=filepath, sampleRate=16000,  resampleQuality=4)()
        audio_44k = loader(filename=filepath, sampleRate=44100, resampleQuality=4)()
    except Exception as e:
        raise ValueError(f"Could not load audio file: {e}") from e

    try:
        with _silence_stderr():
            emb_effnet  = m["effnet"](audio_16k)
            emb_musicnn = m["musicnn"](audio_16k)

            deam_preds      = m["deam"](emb_musicnn)
            valence         = float(np.mean(deam_preds[:, 0]))
            arousal         = float(np.mean(deam_preds[:, 1]))

            mood_happy      = float(np.mean(m["happy"](emb_effnet)[:, 0]))
            mood_sad        = float(np.mean(m["sad"](emb_effnet)[:, 1]))
            mood_aggressive = float(np.mean(m["aggressive"](emb_effnet)[:, 0]))
            mood_relaxed    = float(np.mean(m["relaxed"](emb_effnet)[:, 1]))
            mood_party      = float(np.mean(m["party"](emb_effnet)[:, 1]))
            danceability    = float(np.mean(m["dance"](emb_effnet)[:, 0]))
            bpm             = float(m["bpm"](audio_44k))
    except Exception as e:
        raise ValueError(f"Feature extraction failed: {e}") from e

    return {
        "bpm":             round(bpm, 2),
        "valence":         round(valence, 4),
        "arousal":         round(arousal, 4),
        "mood_happy":      round(mood_happy, 4),
        "mood_sad":        round(mood_sad, 4),
        "mood_aggressive": round(mood_aggressive, 4),
        "mood_relaxed":    round(mood_relaxed, 4),
        "mood_party":      round(mood_party, 4),
        "danceability":    round(danceability, 4),
    }
