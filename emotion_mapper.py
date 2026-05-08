"""
Timbre Emotion Mapper — single-row emotion classification.

Isolated from get_song_emotions.py. Uses fixed thresholds derived from
the full song_features.csv dataset (p30/p50/p70/p90 quantiles).
These are baked in so the Space needs no CSV at runtime.

Emotion taxonomy (must stay in sync with recommend_v2.py MOOD_PROFILES):
  party, euphoric, romantic_passionate, triumphant,
  angry, epic_dark, anxious,
  relaxed, romantic_tender, hopeful,
  sad, melancholic, lonely, nostalgic, dark_ambient, focused
"""

# ── Fixed quantile thresholds from song_features.csv (9,983 songs) ────────────
# Recompute these if the dataset is significantly expanded.
V_HIGH = 5.80   # valence p70
V_MID  = 5.20   # valence p50
V_LOW  = 4.55   # valence p30
A_HIGH = 5.50   # arousal p70
A_MID  = 4.85   # arousal p50
A_LOW  = 4.20   # arousal p30
A_TOP  = 6.30   # arousal p90


def infer_emotion(features: dict) -> str:
    """
    Classify a single feature dict (output of extract_features_for_file)
    into one of 16 emotion labels.

    Input keys used: valence, arousal, mood_happy, mood_sad,
                     mood_aggressive, mood_relaxed, mood_party, danceability
    """
    scores = {
        "party":      features["mood_party"],
        "happy":      features["mood_happy"],
        "sad":        features["mood_sad"],
        "relaxed":    features["mood_relaxed"],
        "aggressive": features["mood_aggressive"],
    }

    # Low-confidence → neutral
    if max(scores.values()) < 0.4:
        return "focused"

    base    = max(scores, key=scores.get)
    valence = features["valence"]
    arousal = features["arousal"]
    sad_score   = features["mood_sad"]
    happy_score = features["mood_happy"]
    danceability = features["danceability"]

    if base == "party":
        if arousal > A_HIGH and valence > V_HIGH:
            return "euphoric"
        if arousal > A_HIGH:
            return "party"
        if valence > V_HIGH and happy_score > 0.5:
            return "euphoric"
        if valence > V_HIGH:
            return "hopeful"
        if happy_score > 0.4 and valence > V_MID:
            return "hopeful"
        if sad_score > 0.4:
            return "nostalgic"
        return "focused"

    elif base == "happy":
        if arousal > A_HIGH and valence > V_HIGH:
            return "euphoric"
        if arousal > A_HIGH:
            return "romantic_passionate"
        if valence > V_HIGH:
            return "hopeful"
        if arousal < A_LOW:
            return "romantic_tender"
        return "romantic_passionate"

    elif base == "sad":
        if valence < V_LOW and arousal < A_LOW:
            return "lonely"
        if arousal < A_LOW:
            return "melancholic"
        if arousal > A_HIGH:
            return "sad"
        if valence < V_LOW:
            return "lonely"
        return "sad"

    elif base == "relaxed":
        if sad_score > 0.7:
            if arousal < A_LOW and valence < V_LOW:
                return "dark_ambient"
            if arousal < A_LOW:
                return "melancholic"
            return "sad"
        if sad_score > 0.5:
            if valence < V_LOW:
                return "melancholic"
            return "nostalgic"
        if valence > V_HIGH:
            if happy_score > 0.4:
                return "relaxed"
            return "hopeful"
        if valence < V_LOW:
            if arousal < A_LOW and sad_score > 0.3:
                return "dark_ambient"
            if arousal < A_LOW:
                return "melancholic"
            return "nostalgic"
        if happy_score > 0.3 and arousal > A_MID:
            return "focused"
        if happy_score > 0.3:
            return "romantic_tender"
        if arousal < A_LOW:
            return "nostalgic"
        return "focused"

    elif base == "aggressive":
        if arousal > A_TOP:
            return "angry"
        if valence > V_HIGH:
            return "triumphant"
        if arousal > A_HIGH:
            return "epic_dark"
        if valence > V_MID:
            return "triumphant"
        return "anxious"

    return "focused"
