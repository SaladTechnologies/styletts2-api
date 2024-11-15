# styletts2-api
Text-To-Speech Inference Server for StyleTTS2

## Usage

```
docker compose up

curl -X POST "http://localhost:4321/generate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}' \
     --output audio.mp3
```
