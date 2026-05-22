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
## Drone Object Detection Capstone

This tool is the final submission for a 2026 capstone project completed as part of the
**USF CTPE Machine Learning Engineering and AI Bootcamp**. It is designed to detect and
count objects captured from drone footage using a fine-tuned YOLOv8m model with
ByteTrack multi-object tracking.

The tracker assigns **persistent IDs across frames** and returns **unique object counts
per class** — so a vehicle visible across 30 frames is counted once, not 30 times.

---

### Model

| | |
|---|---|
| **Architecture** | YOLOv8m (medium variant) |
| **Tracker** | ByteTrack |
| **Input** | Aerial / drone video (1 frame sampled per second) |

### Training Datasets

- **VisDrone 2019** — large-scale drone imagery dataset captured across 14 cities in China,
covering a wide range of object densities, altitudes, and lighting conditions
- **UAVDT** (Unmanned Aerial Vehicle Detection and Tracking) — benchmark dataset for
vehicle detection and tracking from UAV perspectives in urban environments

### Detectable Classes

| ID | Class | Description |
|----|-------------|-------------------------------|
| 0 | person | Pedestrians and crowds |
| 1 | vehicle | Cars, vans, trucks, buses |
| 2 | two_wheeler | Bicycles, motorcycles, tricycles |

### Use Cases

Traffic monitoring · Crowd analysis · Search and rescue · Perimeter surveillance

---

For training details, architecture notes, and dataset preprocessing steps visit the
[project repository](https://github.com/Ozai-03/drone-object-detection-capstone).

**Developed by:** Mathew Peguero
**Mentors:** Obumneme Stanley Dukor & David Adama
**Program:** USF CTPE Machine Learning Engineering and AI Bootcamp · 2026
"""

with gr.Blocks(
    title="Drone Object Detection Capstone",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ),
    css="""
        /* Base background */
        body, .gradio-container { background-color: #0f172a !important; color: #e2e8f0 !important; }

        /* Panels and cards */
        .gr-panel, .gr-box, .gr-form, .gr-block, .block, .gap, .contain { background-color: #1e293b !important; border-color: #334155 !important; }

        /* Tabs */
        .tab-nav button { background-color: #1e293b !important; color: #94a3b8 !important; border-color: #334155 !important; }
        .tab-nav button.selected { background-color: #3b82f6 !important; color: #ffffff !important; border-color: #3b82f6 !important; }

        /* Primary button */
        .gr-button-primary, button.primary { background-color: #3b82f6 !important; border-color: #2563eb !important; color: #ffffff !important; font-weight: 600 !important; }
        .gr-button-primary:hover, button.primary:hover { background-color: #2563eb !important; }

        /* Secondary / clear button */
        button.secondary { background-color: #1e293b !important; border-color: #334155 !important; color: #94a3b8 !important; }
        button.secondary:hover { border-color: #3b82f6 !important; color: #3b82f6 !important; }

        /* Inputs, sliders, textboxes */
        input, textarea, .gr-input, .gr-textbox, select { background-color: #0f172a !important; border-color: #334155 !important; color: #e2e8f0 !important; }
        label, .gr-label, span { color: #94a3b8 !important; }

        /* Slider accent */
        input[type=range]::-webkit-slider-thumb { background-color: #3b82f6 !important; }
        input[type=range]::-webkit-slider-runnable-track { background-color: #334155 !important; }

        /* Dataframe / table */
        .gr-dataframe table { background-color: #1e293b !important; color: #e2e8f0 !important; border-color: #334155 !important; }
        .gr-dataframe th { background-color: #0f172a !important; color: #3b82f6 !important; font-weight: 600 !important; }
        .gr-dataframe td { border-color: #334155 !important; }

        /* Markdown text */
        .gr-markdown, .gr-markdown p, .gr-markdown li { color: #cbd5e1 !important; }
        .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 { color: #f1f5f9 !important; }
        .gr-markdown a { color: #3b82f6 !important; }
        .gr-markdown code { background-color: #0f172a !important; color: #7dd3fc !important; padding: 2px 6px; border-radius: 4px; }

        /* Footer */
        #footer { text-align: center; padding: 24px 0 8px 0; color: #475569 !important; font-size: 0.8rem; border-top: 1px solid #1e293b; margin-top: 24px; }
        #footer a { color: #3b82f6 !important; text-decoration: none; }
    """
) as demo:
    gr.Markdown(
        """
# Drone Object Detection Capstone
### YOLOv8m + ByteTrack · VisDrone 2019 & UAVDT · 3 Object Classes
        """
    )

    with gr.Tab("Try It"):
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input Video")
                conf_slider = gr.Slider(
                    0.1, 0.9, value=0.25, step=0.05,
                    label="Confidence Threshold",
                    info="Minimum confidence score for a detection to be counted (lower = more detections, higher = more precise)"
                )
                max_frames_slider = gr.Slider(
                    60, 300, value=120, step=30,
                    label="Max Frames Sampled",
                    info="Maximum number of frames sampled from the video at 1 frame per second"
                )
                run_btn = gr.Button("Run Detection", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", variant="secondary", size="sm")

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
        clear_btn.click(
            fn=lambda: (None, [], "", "", None),
            inputs=[],
            outputs=[video_input, output_table, stats_text, frames_text, output_video],
        )

    with gr.Tab("About"):
        gr.Markdown(ABOUT_TEXT)

    gr.HTML(
        """
        <div id="footer">
            Drone Object Detection Capstone &copy; 2026 &nbsp;·&nbsp;
            Mathew Peguero &nbsp;·&nbsp;
            USF CTPE Machine Learning Engineering and AI Bootcamp<br/>
            Mentors: Obumneme Stanley Dukor &amp; David Adama &nbsp;·&nbsp;
            <a href="https://github.com/Ozai-03/drone-object-detection-capstone" target="_blank">GitHub Repository</a>
        </div>
        """
    )
