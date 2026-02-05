import base64
import binascii
import io
import os
import numpy as np
import librosa
import torch
import joblib
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# ================= CONFIG =================
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
XGB_MODEL_PATH = "ai_voice_detector_xgb_v2.pkl"
TARGET_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

API_KEY_NAME = "x-api-key"
VALID_API_KEYS = {
    "sk_test_123456789",
    # add prod keys here
}

# ================= FASTAPI =================
app = FastAPI(
    title="AI Voice Detection API",
    version="1.0",
    description="Detects whether a voice is human or AI-generated"
)

# ================= LOAD MODELS =================
print("🔹 Loading wav2vec2...")
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

wav2vec = Wav2Vec2Model.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True,
    use_safetensors=True
).to(DEVICE)
wav2vec.eval()

print("🔹 Loading XGBoost model...")
model = joblib.load(XGB_MODEL_PATH)

# ================= AUTH =================
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key missing")

    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return True

# ================= REQUEST SCHEMA =================
class InferenceRequest(BaseModel):
    language: str | None = "Unknown"
    audioFormat: str | None = "mp3"
    audioBase64: str

# ================= AUDIO UTILS =================
def load_audio_from_base64(b64_audio: str):
    try:
        b64_audio = "".join(b64_audio.split())
        audio_bytes = base64.b64decode(b64_audio, validate=True)

        # basic audio header check
        if not (
            audio_bytes.startswith(b"ID3") or
            audio_bytes[:2] == b"\xff\xfb" or
            audio_bytes.startswith(b"RIFF") or
            audio_bytes.startswith(b"OggS")
        ):
            return None

        buffer = io.BytesIO(audio_bytes)
        y, _ = librosa.load(buffer, sr=TARGET_SR, mono=True)

        if y is None or len(y) < TARGET_SR:
            return None

        y, _ = librosa.effects.trim(y, top_db=25)
        return y

    except (binascii.Error, ValueError, Exception):
        return None

def chunk_audio(y, chunk_sec=2):
    chunk_len = TARGET_SR * chunk_sec
    chunks = [
        y[i:i + chunk_len]
        for i in range(0, len(y) - chunk_len + 1, chunk_len)
    ]
    return chunks if chunks else [y]

# ================= EMBEDDINGS =================
@torch.no_grad()
def extract_embedding(y):
    embeddings = []

    for chunk in chunk_audio(y):
        inputs = processor(
            chunk,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True
        ).input_values.to(DEVICE)

        outputs = wav2vec(inputs)
        hidden = outputs.hidden_states[8]  # middle layer
        pooled = hidden.mean(dim=1)
        embeddings.append(pooled.cpu().numpy())

    return np.mean(np.vstack(embeddings), axis=0)

# ================= EXPLANATION =================
def generate_explanation(label: str):
    if label == "AI_GENERATED":
        return (
            "Unnatural spectral consistency and temporal regularity "
            "commonly observed in synthesized speech."
        )
    else:
        return (
            "Natural variability in pitch, timing, and articulation "
            "consistent with human speech."
        )

# ================= API ENDPOINT =================
@app.post("/api/voice-detection")
def infer_voice(
    req: InferenceRequest,
    authorized: bool = Depends(verify_api_key)
):
    y = load_audio_from_base64(req.audioBase64)

    if y is None:
        return {
            "status": "error",
            "message": "Invalid or malformed audio input"
        }

    emb = extract_embedding(y)
    probs = model.predict_proba(emb.reshape(1, -1))[0]

    if probs[1] > probs[0]:
        label = "AI_GENERATED"
        confidence = probs[1]
    else:
        label = "HUMAN"
        confidence = probs[0]

    return {
        "status": "success",
        "language": req.language,
        "classification": label,
        "confidenceScore": round(float(confidence), 4),
        "explanation": generate_explanation(label)
    }
