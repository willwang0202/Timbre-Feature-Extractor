"""
Download Essentia pretrained model weights.
Runs automatically on Space startup (called from app.py before demo.launch).
Safe to re-run — skips files that already exist.
"""
import os
import ssl
import urllib.request

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

MODELS_DIR = "./models"
BASE_URL = "https://essentia.upf.edu/models"

MODELS = {
    "msd-musicnn-1.pb":                  f"{BASE_URL}/feature-extractors/musicnn/msd-musicnn-1.pb",
    "deam-msd-musicnn-2.pb":             f"{BASE_URL}/classification-heads/deam/deam-msd-musicnn-2.pb",
    "discogs-effnet-bs64-1.pb":          f"{BASE_URL}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
    "mood_happy-discogs-effnet-1.pb":    f"{BASE_URL}/classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb",
    "mood_sad-discogs-effnet-1.pb":      f"{BASE_URL}/classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb",
    "mood_aggressive-discogs-effnet-1.pb": f"{BASE_URL}/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb",
    "mood_relaxed-discogs-effnet-1.pb":  f"{BASE_URL}/classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb",
    "mood_party-discogs-effnet-1.pb":    f"{BASE_URL}/classification-heads/mood_party/mood_party-discogs-effnet-1.pb",
    "danceability-discogs-effnet-1.pb":  f"{BASE_URL}/classification-heads/danceability/danceability-discogs-effnet-1.pb",
}


def ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODELS.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filename} already present")
            continue
        print(f"  ⬇  Downloading {filename}...")
        try:
            with urllib.request.urlopen(url, context=_ssl_ctx) as response, \
                 open(filepath, "wb") as out:
                out.write(response.read())
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✅ {filename} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ❌ Failed: {filename} — {e}")


if __name__ == "__main__":
    ensure_models()
