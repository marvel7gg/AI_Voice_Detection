# 🎙️ AI Voice Detection API

**Wav2Vec2 + XGBoost based Speech Authenticity Detection**

This project detects whether a voice recording is **Human** or
**AI-Generated** using self-supervised speech embeddings and a
lightweight classifier.

------------------------------------------------------------------------

## 🚀 Overview

The pipeline extracts deep speech representations using **Wav2Vec2**,
pools temporal features, and classifies them using **XGBoost**.

Designed for: - 🎤 Real microphone recordings\
- 🤖 AI generated speech (TTS)\
- 📱 Re-recorded audio scenarios\
- 🌍 Multi-language speech

------------------------------------------------------------------------

## 🧠 Architecture

Audio Input\
→ Silence Trimming\
→ 2-Second Chunking\
→ Wav2Vec2 Feature Extraction\
→ Mean Pooling\
→ XGBoost Classifier\
→ Prediction + Confidence

------------------------------------------------------------------------

## ⚙️ Key Design Decisions

### ✅ Silence Trimming

Removes non-speech regions to reduce background bias.

### ✅ 2-Second Chunking

Long recordings are split into fixed windows so the model focuses on
speech characteristics rather than duration.

### ✅ Middle-Layer Pooling

Uses hidden_states\[8\] embeddings from Wav2Vec2 for robust phonetic
features.

### ❌ No Augmentation (Baseline Model)

This version uses clean preprocessing only: - No artificial noise - No
reverbs - No spectral manipulation

------------------------------------------------------------------------

## 📂 Project Structure

ai-voice-detector/ - app.py\
- ai_voice_detector_xgb_v2.pkl\
- trainer/\
- notebooks/

------------------------------------------------------------------------

## 🏋️ Training Pipeline

1.  Resample audio to 16kHz\
2.  Trim silence\
3.  Split into 2-second chunks\
4.  Extract embeddings using facebook/wav2vec2-xls-r-300m\
5.  Train XGBoost classifier

------------------------------------------------------------------------

## 🧪 Inference Example

    predict_file("audio.wav")

Output:

    {
      "prediction": "AI",
      "confidence": 0.92,
      "explanation": "Detected spectral and prosodic patterns..."
    }

------------------------------------------------------------------------

## 🔐 API Usage

Endpoint:

POST /api/voice-detection

Headers: x-api-key: YOUR_API_KEY

Request JSON:

{ "language": "English", "audioFormat": "mp3", "audioBase64":
"`<BASE64_AUDIO>`{=html}" }

------------------------------------------------------------------------

## ⚡ Performance

-   Inference Time: \~200--400 ms per clip\
-   Lightweight deployment with FastAPI\
-   Real-time capable

------------------------------------------------------------------------

## 🧩 Limitations

-   Replay attacks may reduce confidence\
-   Extremely low bitrate audio may degrade performance

------------------------------------------------------------------------

## 🛠️ Future Improvements

-   Replay-robust augmentation\
-   CQCC + SSL hybrid features\
-   CNN-based replay detector

------------------------------------------------------------------------

Generated on: 2026-02-20 03:29:16
