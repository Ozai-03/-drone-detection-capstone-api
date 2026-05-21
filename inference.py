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


def run_inference(video_path: str, conf: float = 0.25, max_frames: int = 120, progress_callback=None, annotate: bool = False) -> dict:
    model = load_model()
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        cap.release()
        raise ValueError("Unable to read video dimensions — file may not be a valid video")

    # int(fps) can be 0 for sub-1fps or malformed files
    frame_interval = max(1, int(fps))

    out_path = None
    writer = None
    if annotate:
        out_fd, out_path = tempfile.mkstemp(suffix="_annotated.mp4")
        os.close(out_fd)
        # 2 fps so each annotated frame is visible for ~0.5 s in the player
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (width, height))

    frames_result = []
    # seen_ids counts unique tracked objects per class across the whole video
    seen_ids: dict[str, set] = {}
    confidence_sum = 0.0
    detection_count = 0
    frame_idx = 0
    sampled = 0
    # hard cap: process at most max_frames seconds of video
    max_total_frames = max_frames * frame_interval

    while frame_idx < max_total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracker on every frame so IDs stay consistent across the whole clip
        results = model.track(frame, conf=conf, verbose=False, persist=True)

        # Only collect and report results at the 1-per-second sampling points
        if frame_idx % frame_interval == 0:
            timestamp_ms = int((frame_idx / fps) * 1000)
            detections = []

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf_val = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else "N/A"
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                print(f"[TRACK] frame={sampled} ts={timestamp_ms}ms | id={track_id} cls={cls_name} conf={conf_val:.2f}")

                detections.append({
                    "class": cls_name,
                    "confidence": round(conf_val, 4),
                    "bbox": [x1, y1, x2, y2],
                    "track_id": track_id,
                })

                if track_id != "N/A":
                    seen_ids.setdefault(cls_name, set()).add(track_id)
                confidence_sum += conf_val
                detection_count += 1

            if writer is not None:
                # plot() returns a BGR numpy array with boxes and labels already drawn
                writer.write(results[0].plot())

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
    if writer is not None:
        writer.release()

    # Reset tracker so IDs from this video don't carry into the next request
    if hasattr(model, "predictor") and model.predictor is not None:
        if hasattr(model.predictor, "trackers") and model.predictor.trackers:
            model.predictor.trackers[0].reset()

    # Count unique objects per class, fall back to detection count if tracker had no IDs
    summary = {cls: len(ids) for cls, ids in seen_ids.items()} if seen_ids else {}

    elapsed_ms = int((time.time() - start_time) * 1000)
    avg_confidence = round(confidence_sum / detection_count, 4) if detection_count > 0 else 0.0

    return {
        "model_version": MODEL_VERSION,
        "inference_time_ms": elapsed_ms,
        "total_frames_processed": sampled,
        "frames": frames_result,
        "summary": summary,
        "avg_confidence": avg_confidence,
        "output_video_path": out_path,
    }
