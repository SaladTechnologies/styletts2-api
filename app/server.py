import time
from io import BytesIO
import base64
import os
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
from __version__ import __version__
from styletts2 import tts
import logging
import magic
from pydub import AudioSegment
from fastapi.responses import StreamingResponse
import tempfile

log_level = os.getenv("LOG_LEVEL", "INFO")
log_level = log_level.upper()
if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    log_level = "INFO"
logging.basicConfig(level=getattr(logging, log_level))

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
    if file_type == "video/mp4":
        file_type = "m4a"
    audio_buffer.seek(0)
    audio = AudioSegment.from_file(audio_buffer, format=file_type)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio.export(f.name, format="wav")
        return f.name


def get_wav_length_from_bytesio(bytes_io):
    # Ensure the buffer's position is at the start
    bytes_io.seek(0)
    audio = AudioSegment.from_file(bytes_io, format="wav")

    # Calculate the duration in milliseconds, then convert to seconds
    duration_seconds = len(audio) / 1000.0
    return audio, duration_seconds


class TTSRequest(BaseModel):
    text: str = Field(..., title="Text to convert to speech")
    voice: Optional[str] = Field(
        None,
        title="Base64 encoded voice sample",
        description="If provided, the model will attempt to match the voice of the provided sample. 3-5s of sample audio is recommended.",
    )
    output_sample_rate: Optional[int] = 24000
    alpha: Optional[float] = Field(
        0.3,
        title="Alpha",
        description="`alpha` is the factor to determine much we use the style sampled based on the text instead of the reference. The higher the value of `alpha`, the more suitable the style it is to the text but less similar to the reference. `alpha` determines the timbre of the speaker.",
    )
    beta: Optional[float] = Field(
        0.7,
        title="Beta",
        description="`beta` is the factor to determine much we use the style sampled based on the text instead of the reference. The higher the value of `beta` the more suitable the style it is to the text but less similar to the reference. Using higher beta makes the synthesized speech more emotional, at the cost of lower similarity to the reference. `beta` determines the prosody of the speaker.",
    )
    diffusion_steps: Optional[int] = Field(
        5,
        title="Diffusion steps",
        description="Since the sampler is ancestral, the higher the steps, the more diverse the samples are, with the cost of slower synthesis speed.",
    )
    embedding_scale: Optional[float] = Field(
        1,
        title="Embedding scale",
        description="This is the classifier-free guidance scale. The higher the scale, the more conditional the style is to the input text and hence more emotional.",
    )
    output_format: Optional[str] = "mp3"


@app.get("/hc")
def health_check():
    return {"status": "ok", "version": __version__}


@app.post("/generate")
def generate(request: TTSRequest, background_tasks: BackgroundTasks):
    start = time.perf_counter()
    params = request.model_dump()
    output_format = params["output_format"]
    del params["output_format"]
    wav_buffer = None
    if "voice" in params and params["voice"] is not None:
        wav_buffer = process_voice(request.voice)
        params["target_voice_path"] = wav_buffer
    del params["voice"]
    wav_bytes = BytesIO()
    model.inference(
        **params,
        output_wav_file=wav_bytes,
    )
    inference_time = time.perf_counter() - start
    logging.info(f"Generated audio in {inference_time} seconds.")
    audio, duration_seconds = get_wav_length_from_bytesio(wav_bytes)
    if wav_buffer is not None:
        background_tasks.add_task(os.remove, wav_buffer)
    headers = {
        "x-inference-time": str(inference_time),
        "x-audio-length": str(duration_seconds),
        "x-realtime-factor": str(duration_seconds / inference_time),
    }
    return_bytes = BytesIO()
    audio.export(return_bytes, format=output_format)
    return_bytes.seek(0)

    return StreamingResponse(
        return_bytes,
        media_type=f"audio/{output_format}",
        headers=headers,
    )


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
