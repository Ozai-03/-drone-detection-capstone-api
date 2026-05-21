import os
import time
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

MODEL_REPO = os.environ.get("MODEL_REPO", "ozai-03/yolov8m-drone-detection")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1.0")

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="best.pt",
        repo_type="model",
    )
    _model = YOLO(model_path)
    return _model


try:
    load_model()
except Exception as e:
    print(f"[WARNING] Model failed to load at startup: {e}")


def run_inference(video_path: str, conf: float = 0.25, max_frames: int = 120, progress_callback=None) -> dict:
    model = load_model()
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(fps)  # sample 1 frame per second

    frames_result = []
    summary: dict[str, int] = {}
    confidence_sum = 0.0
    detection_count = 0
    frame_idx = 0
    sampled = 0

    while sampled < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp_ms = int((frame_idx / fps) * 1000)
            results = model(frame, conf=conf, verbose=False)
            detections = []

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                detections.append({
                    "class": cls_name,
                    "confidence": round(conf_val, 4),
                    "bbox": [x1, y1, x2, y2],
                })

                summary[cls_name] = summary.get(cls_name, 0) + 1
                confidence_sum += conf_val
                detection_count += 1

            frames_result.append({
                "frame_id": sampled,
                "timestamp_ms": timestamp_ms,
                "detections": detections,
            })

            sampled += 1
            if progress_callback:
                progress_callback(sampled, max_frames)

        frame_idx += 1

    cap.release()

    elapsed_ms = int((time.time() - start_time) * 1000)
    avg_confidence = round(confidence_sum / detection_count, 4) if detection_count > 0 else 0.0

    return {
        "model_version": MODEL_VERSION,
        "inference_time_ms": elapsed_ms,
        "total_frames_processed": sampled,
        "frames": frames_result,
        "summary": summary,
        "avg_confidence": avg_confidence,
    }
