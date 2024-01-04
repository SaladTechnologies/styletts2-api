import time
from io import BytesIO
import base64
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
from __version__ import __version__
from styletts2 import tts
import logging
import magic
from pydub import AudioSegment
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)

host = os.getenv("HOST", "*")
port = os.getenv("PORT", "4321")
port = int(port)

warmup_text = "This is an inference API for StyleTTS2. It is now warming up..."

load_start = time.perf_counter()
model = tts.StyleTTS2()
model.inference(warmup_text)
logging.info(f"Model loaded in {time.perf_counter() - load_start} seconds.")

app = FastAPI()


def process_voice(voice_sample: str):
    audio_data = base64.b64decode(voice_sample)
    audio_buffer = BytesIO(audio_data)
    file_type = magic.from_buffer(audio_buffer.read(2048), mime=True)
    audio_buffer.seek(0)
    audio = AudioSegment.from_file(audio_buffer, format=file_type)
    wav_buffer = BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    return wav_buffer, file_type


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    output_sample_rate: Optional[int] = 24000
    alpha: Optional[float] = 0.3
    beta: Optional[float] = 0.7
    diffusion_steps: Optional[int] = 5
    embedding_scale: Optional[int] = 1
    output_format: Optional[str] = "wav"


@app.get("/hc")
def health_check():
    return {"status": "ok", "version": __version__}


@app.post("/generate")
def generate(request: TTSRequest):
    start = time.perf_counter()
    params = request.model_dump()
    output_format = params["output_format"]
    del params["output_format"]
    if "voice" in params and params["voice"] is not None:
        wav_buffer, file_type = process_voice(request.voice)
        params["target_voice_path"] = wav_buffer
    del params["voice"]
    wav_bytes = BytesIO()
    model.inference(
        **params,
        output_wav_file=wav_bytes,
    )
    logging.info(wav_bytes)
    logging.info(f"Generated audio in {time.perf_counter() - start} seconds.")
    return StreamingResponse(
        wav_bytes,
        media_type="audio/wav",
    )


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
