# Training Details

Feature Extractor:
facebook/wav2vec2-xls-r-300m

Pooling:
Mean pooling over time dimension

Classifier:
XGBoost

Dataset:
- ASVspoof
- CommonVoice (multiple languages)
- FoR rerecorded
- EdgeTTS
- ElevenLabs

Preprocessing:
- Silence trimming
- 2-second chunking
- 16kHz resampling