---
title: Drone Detection API
emoji: 🚁
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Drone Detection API

A production REST API and interactive demo for real-time object detection in aerial drone footage. Powered by **YOLOv8m** trained on the **VisDrone** and **UAVDT** datasets.

The Gradio demo is available at the Space's main URL. The FastAPI endpoints are accessible at the paths below.

---

## API Endpoints

### `POST /predict`

Upload a video file and receive per-frame detection results.

**Query parameters:**
| Parameter | Type | Default | Description |
|---|---|---|---|
| `conf` | float | `0.25` | Confidence threshold (0.0–1.0) |
| `max_frames` | int | `120` | Max frames to process (sampled at 1fps) |

**Example request (curl):**
```bash
curl -X POST "https://huggingface.co/spaces/ozai-03/drone-detection-api/predict" \
  -F "video=@sample.mp4"
```

**Example response:**
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
        {"class": "car", "confidence": 0.87, "bbox": [120, 340, 280, 420]}
      ]
    }
  ],
  "summary": {"car": 14, "pedestrian": 3},
  "avg_confidence": 0.762
}
```

---

### `GET /health`

Returns server status and uptime.

```json
{
  "status": "ok",
  "model_version": "v1.0",
  "uptime_seconds": 312
}
```

---

### `GET /metrics`

Returns aggregate statistics across all processed requests.

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

## Model

- **Architecture:** YOLOv8m (medium)
- **Training data:** VisDrone + UAVDT aerial drone datasets
- **Detectable classes:** car, pedestrian, truck, bus, van, motor, bicycle, awning-tricycle, tricycle
- **Weights:** [`ozai-03/yolov8m-drone-detection`](https://huggingface.co/ozai-03/yolov8m-drone-detection)

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_REPO` | `ozai-03/yolov8m-drone-detection` | HF Hub model repository |
| `MODEL_VERSION` | `v1.0` | Model version string returned in responses |
