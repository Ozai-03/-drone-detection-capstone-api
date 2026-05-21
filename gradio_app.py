import gradio as gr

from inference import run_inference


def detect(video_path, conf, max_frames):
    if video_path is None:
        return [["No video uploaded", ""]], "Upload a video and click Run Detection.", ""

    try:
        result = run_inference(video_path, conf=float(conf), max_frames=int(max_frames))
    except Exception as e:
        return [["Error", str(e)]], f"Inference failed: {e}", ""

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

    return table, stats, frames_note


ABOUT_TEXT = """
## About This Model

**Model:** YOLOv8m (medium variant)

**Training data:**
- [VisDrone](http://aiskyeye.com/) — aerial drone imagery dataset
- [UAVDT](https://sites.google.com/view/grli-uavdt) — UAV benchmark for detection and tracking

**Classes detected:**
`car`, `pedestrian`, `truck`, `bus`, `van`, `motor`, `bicycle`, `awning-tricycle`, `tricycle`

**Use case:** Real-time object detection in aerial/drone footage for surveillance, traffic monitoring, and search-and-rescue scenarios.
"""

with gr.Blocks(title="Drone Object Detection") as demo:
    gr.Markdown("# Drone Object Detection\nYOLOv8m trained on VisDrone + UAVDT aerial datasets.")

    with gr.Tab("Try It"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Drone Video")
                conf_slider = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence Threshold")
                max_frames_slider = gr.Slider(5, 60, value=30, step=5, label="Max Frames")
                run_btn = gr.Button("Run Detection", variant="primary")

            with gr.Column():
                output_table = gr.Dataframe(
                    headers=["Class", "Count"],
                    label="Detection Summary",
                    interactive=False,
                )
                stats_text = gr.Textbox(label="Stats", interactive=False)
                frames_text = gr.Textbox(label="Frames", interactive=False)

        run_btn.click(
            fn=detect,
            inputs=[video_input, conf_slider, max_frames_slider],
            outputs=[output_table, stats_text, frames_text],
        )

    with gr.Tab("About"):
        gr.Markdown(ABOUT_TEXT)
