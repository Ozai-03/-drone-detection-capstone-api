import os
import json
import time
import uuid
import tempfile
import traceback
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

import gradio as gr
from gradio_app import demo as gradio_demo
from inference import run_inference, MODEL_VERSION

START_TIME = time.time()
LOG_PATH = "/data/request_log.jsonl"

# In-memory store mapping video tokens to temp file paths
_video_store: dict[str, str] = {}

app = FastAPI(title="Drone Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _append_log(entry: dict):
    try:
        os.makedirs("/data", exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[WARNING] Could not write to log: {e}")


@app.post("/predict")
async def predict(
    video: UploadFile = File(...),
    conf: float = Query(default=0.25, ge=0.0, le=1.0),
    max_frames: int = Query(default=1800, ge=1, le=3600),
):
    tmp_path = None
    status = "success"
    try:
        suffix = os.path.splitext(video.filename or "upload.mp4")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name

        result = run_inference(tmp_path, conf=conf, max_frames=max_frames, annotate=True)

        # Store annotated video and return a token
        video_token = str(uuid.uuid4())
        if result.get("output_video_path") and os.path.exists(result["output_video_path"]):
            _video_store[video_token] = result.pop("output_video_path")
            result["video_token"] = video_token

        _append_log({
            "timestamp": datetime.utcnow().isoformat(),
            "filename": video.filename,
            "total_frames_processed": result["total_frames_processed"],
            "inference_time_ms": result["inference_time_ms"],
            "avg_confidence": result["avg_confidence"],
            "status": status,
        })

        return JSONResponse(content=result)

    except ValueError as e:
        status = "error"
        _append_log({"timestamp": datetime.utcnow().isoformat(), "filename": video.filename, "status": status, "error": str(e)})
        return JSONResponse(status_code=422, content={"error": "invalid video", "detail": str(e)})

    except Exception as e:
        status = "error"
        tb = traceback.format_exc()
        print(f"[ERROR] Inference failed:\n{tb}")
        _append_log({"timestamp": datetime.utcnow().isoformat(), "filename": video.filename, "status": status, "error": str(e)})
        return JSONResponse(status_code=500, content={"error": "inference failed", "detail": str(e)})

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.get("/video/{token}")
async def get_video(token: str):
    path = _video_store.get(token)
    if not path or not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "video not found or expired"})

    def cleanup():
        try:
            os.remove(path)
            _video_store.pop(token, None)
        except Exception:
            pass

    return FileResponse(
        path,
        media_type="video/mp4",
        filename="annotated.mp4",
        background=cleanup
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "uptime_seconds": int(time.time() - START_TIME),
    }


@app.get("/metrics")
def metrics():
    total = successful = failed = 0
    inference_times = []
    confidences = []

    try:
        with open(LOG_PATH, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    total += 1
                    if entry.get("status") == "success":
                        successful += 1
                        if "inference_time_ms" in entry:
                            inference_times.append(entry["inference_time_ms"])
                        if "avg_confidence" in entry:
                            confidences.append(entry["avg_confidence"])
                    else:
                        failed += 1
                except Exception:
                    pass
    except FileNotFoundError:
        pass

    avg_inference = round(sum(inference_times) / len(inference_times), 2) if inference_times else 0.0
    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    return {
        "total_requests": total,
        "successful_requests": successful,
        "failed_requests": failed,
        "avg_inference_time_ms": avg_inference,
        "avg_confidence": avg_conf,
    }


# Mount Gradio at "/" AFTER all API routes are registered.
gr.mount_gradio_app(app, gradio_demo, path="/")