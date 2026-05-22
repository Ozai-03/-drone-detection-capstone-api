---
title: Drone Detection API
emoji: 🚁
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Drone Detection API

A production REST API and interactive Gradio demo for real-time object detection in aerial drone footage, built as a capstone deployment project.

**Live demo:** [huggingface.co/spaces/ozai-03/drone-detection-api](https://huggingface.co/spaces/ozai-03/drone-detection-api)  
**Training repo:** [Ozai-03/drone-object-detection-capstone](https://github.com/Ozai-03/drone-object-detection-capstone)  
**Model weights:** [ozai-03/yolov8m-drone-detection](https://huggingface.co/ozai-03/yolov8m-drone-detection)

---

## Overview

Powered by **YOLOv8m** and **ByteTrack**, the model detects 9 object classes in aerial footage and assigns persistent IDs across frames — so a vehicle visible across 30 frames is counted once, not 30 times.

**Detectable classes:** `car` · `pedestrian` · `truck` · `bus` · `van` · `motor` · `bicycle` · `awning-tricycle` · `tricycle`

**Training data:**
- [VisDrone](http://aiskyeye.com/) — large-scale aerial drone imagery dataset
- [UAVDT](https://sites.google.com/view/grli-uavdt) — UAV benchmark for detection and tracking

---

## Architecture

```
gradio_app.py   — Gradio demo UI (port 7861)
app.py          — FastAPI REST API (port 7860)
inference.py    — YOLOv8m + ByteTrack inference engine
Dockerfile      — Container definition for HF Spaces
start.sh        — Starts both servers simultaneously
```

Model weights are stored on HF Hub and downloaded at startup — no large files in this repo.

---

## API Reference

### `POST /predict`

Upload a video and receive per-frame detection results with track IDs.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `conf` | float | `0.25` | Confidence threshold (0.0–1.0) |
| `max_frames` | int | `120` | Max seconds of video to process |

**Example:**

```bash
curl -X POST "https://huggingface.co/spaces/ozai-03/drone-detection-api/predict" \
  -F "video=@sample.mp4"
```

**Response:**

```json
{
  "model_version": "v1.0",
  "inference_time_ms": 4200,
  "total_frames_processed": 60,
  "frames": [
    {
      "frame_id": 0,
      "timestamp_ms": 0,
      "detections": [
        {"class": "car", "confidence": 0.87, "bbox": [120, 340, 280, 420], "track_id": 1}
      ]
    }
  ],
  "summary": {"car": 14, "pedestrian": 3},
  "avg_confidence": 0.762
}
```

### `GET /health`

```json
{"status": "ok", "model_version": "v1.0", "uptime_seconds": 312}
```

### `GET /metrics`

```json
{
  "total_requests": 42,
  "successful_requests": 40,
  "failed_requests": 2,
  "avg_inference_time_ms": 3850.5,
  "avg_confidence": 0.7413
}
```

---

## Deployment

This project is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/ozai-03/drone-detection-api) using a Docker container. Both the FastAPI server and Gradio demo run simultaneously via `start.sh`.

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `MODEL_REPO` | `ozai-03/yolov8m-drone-detection` | HF Hub model repo |
| `MODEL_VERSION` | `v1.0` | Version string returned in API responses |

---

## Credits

**Developed by:** Mathew Peguero  
**Mentors:** Obumneme Stanley Dukor & David Adama
