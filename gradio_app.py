import gradio as gr

from inference import run_inference


def detect(video_path, conf, max_frames, progress=gr.Progress()):
    if video_path is None:
        return [["No video uploaded", ""]], "Upload a video and click Run Detection.", "", None

    max_frames_int = int(max_frames)
    progress(0, desc="Starting inference...")

    def on_frame(sampled, total):
        progress(sampled / total, desc=f"Processing frame {sampled} of {total}...")

    try:
        result = run_inference(video_path, conf=float(conf), max_frames=max_frames_int, progress_callback=on_frame, annotate=True)
    except Exception as e:
        return [["Error", str(e)]], f"Inference failed: {e}", "", None

    summary = result.get("summary", {})
    if not summary:
        table = [["No detections", "0"]]
    else:
        table = [[cls, str(count)] for cls, count in sorted(summary.items(), key=lambda x: -x[1])]

    total_objects = sum(summary.values())
    avg_conf = result.get("avg_confidence", 0.0)
    inf_time = result.get("inference_time_ms", 0) / 1000

    stats = f"Total objects: {total_objects}  |  Avg confidence: {avg_conf:.2%}  |  Inference time: {inf_time:.2f}s"
    frames_note = f"Frames processed: {result['total_frames_processed']}"
    out_video = result.get("output_video_path")

    return table, stats, frames_note, out_video


ABOUT_TEXT = """
## Drone Object Detection — YOLOv8m

A real-time aerial object detection system trained on drone footage datasets. Upload a drone video and the model detects and tracks vehicles, pedestrians, and other objects across all frames.

The tracker assigns **persistent IDs across frames** and returns **unique object counts per class** — so a vehicle visible across 30 frames is counted once, not 30 times.

---

### Model

| | |
|---|---|
| **Architecture** | YOLOv8m (medium variant) |
| **Tracker** | ByteTrack |
| **Input** | Aerial / drone video |

### Training Data

- [VisDrone](http://aiskyeye.com/) — large-scale aerial drone imagery dataset
- [UAVDT](https://sites.google.com/view/grli-uavdt) — UAV benchmark for detection and tracking

### Detectable Classes

`car` · `pedestrian` · `truck` · `bus` · `van` · `motor` · `bicycle` · `awning-tricycle` · `tricycle`

### Use Cases

Traffic monitoring · Surveillance · Search-and-rescue

---

For training details, architecture notes, and dataset preprocessing steps visit the
[project repository](https://github.com/Ozai-03/drone-object-detection-capstone).

**Developed by:** Mathew Peguero
**Mentors:** Obumneme Stanley Dukor & David Adama
"""

with gr.Blocks(title="Drone Object Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# Drone Object Detection
YOLOv8m + ByteTrack · Trained on VisDrone & UAVDT · Detects 9 object classes in aerial footage
        """
    )

    with gr.Tab("Try It"):
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input Video")
                conf_slider = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence Threshold")
                max_frames_slider = gr.Slider(60, 300, value=120, step=30, label="Max Duration (seconds)")
                run_btn = gr.Button("Run Detection", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_video = gr.Video(label="Annotated Output", interactive=False)
                output_table = gr.Dataframe(
                    headers=["Class", "Count"],
                    label="Unique Objects Detected",
                    interactive=False,
                )
                stats_text = gr.Textbox(label="Stats", interactive=False)
                frames_text = gr.Textbox(label="Frames Processed", interactive=False)

        run_btn.click(
            fn=detect,
            inputs=[video_input, conf_slider, max_frames_slider],
            outputs=[output_table, stats_text, frames_text, output_video],
        )

    with gr.Tab("About"):
        gr.Markdown(ABOUT_TEXT)
