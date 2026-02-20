# Inference Pipeline

Step 1 — Decode base64 audio
Step 2 — Resample to 16kHz
Step 3 — Trim silence
Step 4 — Split into 2-second chunks
Step 5 — Extract embeddings from Wav2Vec2 layer 8
Step 6 — Mean pooling
Step 7 — XGBoost classification

Output:
{
  classification,
  confidenceScore,
  explanation
}