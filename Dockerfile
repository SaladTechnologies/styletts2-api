FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app
ENV TZ=etc/UTC DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
  git \
  build-essential \
  ffmpeg \
  libmagic1

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

RUN python -c "from styletts2 import tts; tts.StyleTTS2()"

COPY app/ .

CMD ["python", "server.py"]