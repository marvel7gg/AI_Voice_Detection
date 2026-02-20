# API Usage

Endpoint:
POST /api/voice-detection

Headers:
x-api-key: YOUR_KEY

Body:
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "..."
}

Response:
{
  "status": "success",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.92
}