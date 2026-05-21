FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data

ENV MODEL_REPO=ozai-03/yolov8m-drone-detection
ENV MODEL_VERSION=v1.0

EXPOSE 7860

COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
