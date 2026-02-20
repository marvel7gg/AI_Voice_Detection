# System Architecture

The project uses a hybrid AI voice detection pipeline:

1. Audio Input (Base64 MP3)
2. Silence trimming + 2s chunking
3. Wav2Vec2 SSL feature extraction
4. Mean pooling of hidden representations
5. XGBoost classifier for Human vs AI decision
6. FastAPI service layer

Why this architecture?

- Wav2Vec2 captures speech characteristics.
- XGBoost provides fast inference and strong tabular learning.
- Chunking improves replay attack robustness.