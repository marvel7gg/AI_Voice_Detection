import base64
import io
import asyncio
from collections import deque

import numpy as np
import torch
import librosa
import resampy
import joblib

from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# ================= CONFIG =================
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
XGB_MODEL_PATH = "ai_voice_detector_xgb_v2.pkl"
TARGET_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VALID_API_KEYS = {"rt_wzOfeOpwehzVvaTA"}
MAX_BATCH_SIZE = 12

# ================= GPU PERFORMANCE =================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ================= FASTAPI =================
app = FastAPI(
    title="AI Voice Detection API (GPU)",
    version="6.0-gpu",
    description="Tesla T4 optimized continuous batching engine"
)

# ================= LOAD MODELS =================
print("🚀 Loading wav2vec2 on", DEVICE)

processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

wav2vec = Wav2Vec2Model.from_pretrained(
    MODEL_NAME,
    output_hidden_states=True,
    use_safetensors=True
).to(DEVICE)

wav2vec.eval()

# ⭐ Safe FP16 (keeps prediction stable)
if DEVICE == "cuda":
    wav2vec = wav2vec.half()

print("🔹 Loading XGBoost model...")
model = joblib.load(XGB_MODEL_PATH)

# ================= AUTH =================
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key missing")
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

# ================= REQUEST MODEL =================
class InferenceRequest(BaseModel):
    language: str | None = "Unknown"
    audioFormat: str | None = "mp3"
    audioBase64: str

# ================= MP3-OPTIMIZED AUDIO LOADER =================
def load_audio_from_base64(b64_audio: str):
    try:
        b64_audio = "".join(b64_audio.split())
        audio_bytes = base64.b64decode(b64_audio, validate=True)
        buffer = io.BytesIO(audio_bytes)

        # ⭐ Fast MP3 decoding path
        y, sr = librosa.load(
            buffer,
            sr=None,
            mono=True,
            res_type="kaiser_fast"
        )

        if y is None or len(y) < TARGET_SR:
            return None

        if sr != TARGET_SR:
            y = resampy.resample(y, sr, TARGET_SR, filter="kaiser_fast")

        abs_y = np.abs(y)
        idx = np.where(abs_y > 0.01)[0]
        if len(idx) > 0:
            y = y[idx[0]:idx[-1]]

        return y.astype(np.float32, copy=False)

    except Exception as e:
        print("❌ GPU loader failed:", e)
        return None

# ================= CHUNK AUDIO =================
def chunk_audio(y, chunk_sec=2):
    chunk_len = TARGET_SR * chunk_sec
    chunks = [
        y[i:i + chunk_len]
        for i in range(0, len(y) - chunk_len + 1, chunk_len)
    ]
    return chunks if chunks else [y]

# ================= CONTINUOUS QUEUE =================
REQUEST_QUEUE = deque()

# ================= GPU FORWARD =================
@torch.inference_mode()
def batched_embedding_forward(list_of_audio_arrays):

    all_chunks = []
    mapping = []

    for req_id, y in enumerate(list_of_audio_arrays):
        chunks = chunk_audio(y)
        for c in chunks:
            all_chunks.append(c)
            mapping.append(req_id)

    inputs = processor(
        all_chunks,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    ).input_values.to(DEVICE)

    if DEVICE == "cuda":
        inputs = inputs.to(torch.float16)

    outputs = wav2vec(inputs)

    hidden = outputs.hidden_states[8]
    pooled = hidden.mean(dim=1).cpu().numpy()

    per_request = [[] for _ in list_of_audio_arrays]

    for emb, idx in zip(pooled, mapping):
        per_request[idx].append(emb)

    final_embeddings = [
        np.mean(np.vstack(x), axis=0) for x in per_request
    ]

    return final_embeddings

# ================= GPU WORKER =================
async def gpu_continuous_worker():
    print("🔥 GPU batching worker started")

    while True:

        if not REQUEST_QUEUE:
            await asyncio.sleep(0.001)
            continue

        batch_audio = []
        futures = []

        while REQUEST_QUEUE and len(batch_audio) < MAX_BATCH_SIZE:
            audio, fut = REQUEST_QUEUE.popleft()
            batch_audio.append(audio)
            futures.append(fut)

        try:
            embeddings = batched_embedding_forward(batch_audio)

            for fut, emb in zip(futures, embeddings):
                if not fut.done():
                    fut.set_result(emb)

        except Exception as e:
            print("❌ GPU batch failed:", e)

            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(gpu_continuous_worker())

# ================= ENQUEUE =================
async def enqueue_embedding(y):
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    REQUEST_QUEUE.append((y, future))
    return await future

# ================= EXPLANATION =================
def generate_explanation(label: str):
    if label == "AI_GENERATED":
        return "Unnatural spectral consistency typical of synthesized speech."
    else:
        return "Natural variability consistent with human speech."

# ================= API =================
@app.post("/api/voice-detection")
async def infer_voice(
    req: InferenceRequest,
    authorized: bool = Depends(verify_api_key)
):

    y = load_audio_from_base64(req.audioBase64)

    if y is None:
        return {
            "status": "error",
            "message": "Invalid or malformed audio input"
        }

    emb = await enqueue_embedding(y)

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
