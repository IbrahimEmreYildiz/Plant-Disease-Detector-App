import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from src.config import TRAIN_CONFIG, INFERENCE_CONFIG, SEVERITY_THRESHOLDS, RESULTS_DIR
from src.detector import DiseaseDetector
from src.severity_analyzer import SeverityAnalyzer
from src.visualizer import Visualizer
from src.report_generator import ReportGenerator

# Model Loading logic — prefer fine-tuned train-2, fallback to train, then base model
def load_model():
    candidates = [
        "best.pt",
        os.path.join("runs", "segment", "train-2", "weights", "best.pt"),
        os.path.join("runs", "segment", "train", "weights", "best.pt"),
    ]
    weights_path = None
    for path in candidates:
        if os.path.exists(path):
            weights_path = path
            print(f"Loading model weights from: {weights_path}")
            break
    if weights_path is None:
        weights_path = TRAIN_CONFIG['model']
        print(f"No custom weights found. Using base pretrained model: {weights_path}")
    return DiseaseDetector(model_path=weights_path, conf_thresh=INFERENCE_CONFIG['conf'], iou_thresh=INFERENCE_CONFIG['iou'])

try:
    detector = load_model()
except Exception as e:
    print(f"Error loading model: {e}")
    detector = None

analyzer = SeverityAnalyzer(SEVERITY_THRESHOLDS)
visualizer = Visualizer()

def process_image(image):
    if detector is None:
        return image, "Model not loaded properly."
    
    report_gen = ReportGenerator(RESULTS_DIR)
    
    # 1. Detect
    detections = detector.detect(image)
    
    # 2. Analyze Severity
    dets_with_severity = []
    for det in detections:
        sev_result = analyzer.analyze(det)
        det["severity"] = sev_result
        dets_with_severity.append(det)
        
    # 3. Visualize
    annotated_image = visualizer.draw(image, dets_with_severity)
    
    # 4. Generate Report
    report_gen.add_frame(0, dets_with_severity)
    report_path = report_gen.generate_text_report("image_report.txt")
    # Format quick summary for UI
    quick_summary = f"Total Detections: {len(dets_with_severity)}\n"
    for det in dets_with_severity:
        quick_summary += f"- {det['class_name']}: {det['severity']['level']} ({det['severity']['ratio']*100:.0f}%)\n"
    
    # Read the full report to display in the UI
    with open(report_path, "r", encoding="utf-8") as f:
        full_report_text = f.read()
        
    return annotated_image, quick_summary, full_report_text, report_path

def process_video(video_path):
    if detector is None:
        return None, "Model not loaded properly."
        
    report_gen = ReportGenerator(RESULTS_DIR)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_path = os.path.join(RESULTS_DIR, "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    stride = 5  # Process every 5th frame to speed up
    last_dets = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only run heavy inference every 'stride' frames
        if frame_idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect(frame_rgb)
            
            last_dets = []
            for det in detections:
                det["severity"] = analyzer.analyze(det)
                last_dets.append(det)
            
            report_gen.add_frame(frame_idx, last_dets)
        
        # Draw (using last known detections for skipped frames)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame = visualizer.draw(frame_rgb, last_dets)
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        
        frame_idx += 1
        
    cap.release()
    out.release()
    
    report_path = report_gen.generate_text_report("video_report.txt")
    
    with open(report_path, "r", encoding="utf-8") as f:
        full_report_text = f.read()
        
    quick_summary = f"Processed {frame_idx} frames.\nReport saved and ready for download."
    
    return out_path, quick_summary, full_report_text, report_path

# Custom CSS
custom_css = """
body, .gradio-container {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
    min-height: 100vh;
}
.gradio-container {
    font-family: 'Segoe UI', sans-serif !important;
}
/* Force ALL text to be white/light */
* {
    color: #e8f4ff !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    text-shadow: 0 0 20px rgba(100, 200, 255, 0.4);
}
/* Panels / blocks */
.gr-block, .gr-box, .block, .panel, .wrap, .gap, .form {
    background: rgba(255, 255, 255, 0.07) !important;
    border: 1px solid rgba(100, 180, 255, 0.2) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px) !important;
}
#header_area, #header_area .block, #header_area .gr-block, #header_area .gr-box {
    background: transparent !important;
    border: none !important;
    backdrop-filter: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}
#header_area * {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
/* Primary button */
button.primary, .primary {
    background: linear-gradient(90deg, #1a6fc4, #22c1c3) !important;
    border: none !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s;
}
button.primary:hover {
    opacity: 0.85 !important;
}
/* All other buttons */
button {
    color: #e8f4ff !important;
}
/* Tabs */
.tab-nav button {
    color: #90caf9 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.tab-nav button.selected {
    color: #ffffff !important;
    border-bottom: 2px solid #22c1c3 !important;
}
/* Textbox / textarea */
textarea, input[type="text"], input {
    background: rgba(15, 32, 39, 0.6) !important;
    color: #e8f4ff !important;
    border: 1px solid rgba(100, 180, 255, 0.3) !important;
    border-radius: 8px !important;
}
/* Accordion */
.accordion, details, summary {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(100,180,255,0.15) !important;
    border-radius: 8px !important;
    color: #e8f4ff !important;
}
/* File component */
.file-preview, .file-preview * {
    color: #e8f4ff !important;
    background: rgba(15, 32, 39, 0.5) !important;
}
/* Markdown prose */
.prose, .prose * {
    color: #cce8ff !important;
}
"""

# Gradio Interface
with gr.Blocks(title="🌿 Crop Disease Intelligence System", css=custom_css) as demo:
    with gr.Column(elem_id="header_area"):
        gr.Markdown("# 🌿 Crop Disease Intelligence System")
        gr.Markdown("Real-time plant disease detection and severity analysis using YOLO instance segmentation.")
    
    with gr.Tab("Image Inference"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="Upload Crop Image", height=350)
                image_button = gr.Button("Analyze Image", variant="primary")
                with gr.Accordion("View Full Detailed Report", open=False):
                    image_detailed_report = gr.Textbox(label="", lines=10)
                image_file = gr.File(label="Download Text Report")
            with gr.Column():
                image_output = gr.Image(type="numpy", label="Result", height=350)
                image_summary = gr.Textbox(label="Quick Summary", lines=3)
                
        image_button.click(
            process_image,
            inputs=[image_input],
            outputs=[image_output, image_summary, image_detailed_report, image_file]
        )
        
    with gr.Tab("Video Inference"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Crop Video", height=350)
                video_button = gr.Button("Analyze Video", variant="primary")
                with gr.Accordion("View Full Detailed Report", open=False):
                    video_detailed_report = gr.Textbox(label="", lines=10)
                video_file = gr.File(label="Download Text Report")
            with gr.Column():
                video_output = gr.Video(label="Result Video", height=350)
                video_summary = gr.Textbox(label="Quick Summary", lines=2)
                
        video_button.click(
            process_video,
            inputs=[video_input],
            outputs=[video_output, video_summary, video_detailed_report, video_file]
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)
