"""
BrailleVision — Gradio Web Application
Real-time physical Braille to English using camera-based AI.

Run: python app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import gradio as gr
from PIL import Image
from detector import process_image, generate_synthetic_braille
from tts import text_to_speech


# ─── Helper ──────────────────────────────────────────────────────────────────

def pil_to_bgr(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def gray_to_pil(gray):
    return Image.fromarray(gray)


# ─── Core Processing ─────────────────────────────────────────────────────────

def run_detection(input_image):
    """Main detection callback for uploaded image or webcam snapshot."""
    if input_image is None:
        return None, None, "⚠️ No image provided. Upload a photo or use the webcam.", None

    bgr = pil_to_bgr(input_image)
    result = process_image(bgr)

    text = result['text']
    annotated_pil = bgr_to_pil(result['annotated'])
    debug_pil = gray_to_pil(result['debug_binary'])

    dot_info = (
        f"🔵 Dots detected: {result['dot_count']}\n"
        f"📦 Braille cells: {result['cell_count']}\n"
        f"📊 Confidence: {result['confidence'].upper()}\n\n"
        f"📝 Decoded text:\n{text if text else '(nothing decoded — see tips below)'}"
    )

    if not text:
        dot_info += (
            "\n\n💡 Tips for better results:\n"
            "• Use strong raking light (lamp at 45°)\n"
            "• Keep camera parallel to the Braille page\n"
            "• Ensure even, shadow-creating light on raised dots\n"
            "• Try higher resolution / zoom in closer"
        )

    # TTS
    audio_path = None
    if text:
        audio_path = text_to_speech(text)

    return annotated_pil, debug_pil, dot_info, audio_path


def run_demo(demo_text):
    """Generate a synthetic Braille image and run detection on it."""
    if not demo_text.strip():
        demo_text = "hello"

    demo_text = demo_text.lower()[:20]  # limit length
    img_w = max(500, len(demo_text) * 110)
    gray_img = generate_synthetic_braille(demo_text, img_size=(img_w, 200))
    bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    result = process_image(bgr)
    annotated_pil = bgr_to_pil(result['annotated'])
    debug_pil = gray_to_pil(result['debug_binary'])

    dot_info = (
        f"[DEMO MODE — synthetic Braille for: '{demo_text}']\n\n"
        f"🔵 Dots detected: {result['dot_count']}\n"
        f"📦 Braille cells: {result['cell_count']}\n"
        f"📊 Confidence: {result['confidence'].upper()}\n\n"
        f"📝 Decoded text:\n{result['text'] if result['text'] else '(none)'}"
    )

    audio_path = None
    if result['text']:
        audio_path = text_to_speech(result['text'])

    return annotated_pil, debug_pil, dot_info, audio_path


# ─── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
body { font-family: 'Segoe UI', sans-serif; }

.gr-button-primary {
    background: #1a56db !important;
    color: white !important;
    font-size: 1.1rem !important;
    padding: 12px 28px !important;
    border-radius: 8px !important;
    font-weight: bold !important;
}

.gr-button-secondary {
    font-size: 1rem !important;
    padding: 10px 22px !important;
    border-radius: 8px !important;
}

.output-text {
    font-size: 1.2rem !important;
    line-height: 1.8 !important;
    font-family: 'Courier New', monospace !important;
    background: #f0f4ff !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

.title-row {
    text-align: center;
    margin-bottom: 10px;
}

#big-result textarea {
    font-size: 1.4rem !important;
    font-weight: bold !important;
    color: #1a56db !important;
    min-height: 80px !important;
}
"""

def build_ui():
    with gr.Blocks(
        title="BrailleVision — Physical Braille to English",
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
        )
    ) as demo:

        # ── Header ──
        gr.HTML("""
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <h1 style="font-size:2.4rem; color:#1a56db; margin:0;">
                👁️ BrailleVision
            </h1>
            <p style="font-size:1.15rem; color:#444; margin-top:6px;">
                Real-time Physical Braille → English Text + Speech
            </p>
            <p style="font-size:0.95rem; color:#888;">
                Point your camera at real embossed or handwritten Braille paper
            </p>
        </div>
        """)

        # ── How to use banner ──
        with gr.Accordion("📷 How to get best results", open=False):
            gr.HTML("""
            <div style="padding:10px 20px; font-size:1rem; line-height:2;">
                <b>1.</b> Place Braille paper under a bright lamp at a 45° angle — the raking light creates shadows on raised dots.<br>
                <b>2.</b> Hold your phone/webcam steady, camera parallel to the paper.<br>
                <b>3.</b> Zoom in so Braille cells are clearly visible (dots should be ≥ 15px wide in the image).<br>
                <b>4.</b> Avoid blurry or overexposed images — sharp contrast between dot shadows and paper is key.<br>
                <b>5.</b> Grade 1 Braille (standard alphabet) is fully supported.
            </div>
            """)

        with gr.Tabs():

            # ── Tab 1: Image Upload / Webcam ──
            with gr.TabItem("📸 Scan Braille", id="scan"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(
                            label="Upload image or use webcam",
                            sources=["upload", "webcam", "clipboard"],
                            type="pil",
                            height=320,
                            elem_id="input-image"
                        )
                        scan_btn = gr.Button(
                            "🔍 Detect & Read Braille",
                            variant="primary",
                            size="lg"
                        )
                        gr.HTML("""
                        <p style="font-size:0.9rem; color:#666; margin-top:8px;">
                            ♿ Supports: embossed paper, slate & stylus Braille, Braille writer output
                        </p>
                        """)

                    with gr.Column(scale=1):
                        annotated_out = gr.Image(
                            label="Detected dots (green circles)",
                            height=250
                        )
                        debug_out = gr.Image(
                            label="Preprocessed (dot detection view)",
                            height=150
                        )

                with gr.Row():
                    with gr.Column():
                        result_text = gr.Textbox(
                            label="📝 Decoded English Text",
                            lines=3,
                            interactive=False,
                            elem_id="big-result"
                        )
                        info_text = gr.Textbox(
                            label="ℹ️ Detection Details",
                            lines=6,
                            interactive=False
                        )
                    with gr.Column():
                        audio_out = gr.Audio(
                            label="🔊 Text-to-Speech Output",
                            autoplay=True
                        )
                        gr.HTML("""
                        <div style="background:#e8f4fd; border-radius:8px; padding:14px; margin-top:10px;">
                            <b>♿ Accessibility Features:</b><br>
                            • Text-to-speech auto-plays after scan<br>
                            • High-contrast detection overlay<br>
                            • Large text output panel<br>
                            • Works with uploaded photos or live webcam
                        </div>
                        """)

                scan_btn.click(
                    fn=run_detection,
                    inputs=[img_input],
                    outputs=[annotated_out, debug_out, info_text, audio_out]
                )

            # ── Tab 2: Demo Mode ──
            with gr.TabItem("🧪 Demo (Synthetic Braille)", id="demo"):
                gr.HTML("""
                <div style="background:#fff8e1; border-radius:8px; padding:14px; margin-bottom:14px;">
                    <b>No physical Braille paper?</b> Type any word below to generate a synthetic
                    Braille image and test the full detection pipeline instantly.
                </div>
                """)
                with gr.Row():
                    demo_input = gr.Textbox(
                        label="Enter text to convert to synthetic Braille",
                        value="hello",
                        placeholder="e.g. medicine, care, help",
                        scale=3
                    )
                    demo_btn = gr.Button("Generate & Detect", variant="primary", scale=1)

                with gr.Row():
                    demo_annotated = gr.Image(label="Detected synthetic Braille", height=220)
                    demo_debug = gr.Image(label="Preprocessed view", height=220)

                demo_info = gr.Textbox(label="Results", lines=7, interactive=False)
                demo_audio = gr.Audio(label="🔊 Speech Output", autoplay=True)

                demo_btn.click(
                    fn=run_demo,
                    inputs=[demo_input],
                    outputs=[demo_annotated, demo_debug, demo_info, demo_audio]
                )

            # ── Tab 3: About ──
            with gr.TabItem("ℹ️ About"):
                gr.HTML("""
                <div style="max-width:700px; margin:auto; line-height:1.9; font-size:1rem;">
                    <h2>BrailleVision 2026</h2>
                    <p>
                        BrailleVision is a real-time assistive technology system that converts
                        <b>physical embossed or handwritten Braille</b> into English text and speech
                        using computer vision — no Braille literacy required by the reader.
                    </p>
                    <h3>Who is this for?</h3>
                    <ul>
                        <li>Caregivers reading Braille medical notes or prescription labels</li>
                        <li>Teachers in inclusive classrooms</li>
                        <li>Volunteers assisting visually impaired individuals</li>
                        <li>Accessibility workers and social service professionals</li>
                    </ul>
                    <h3>Technical Pipeline</h3>
                    <pre style="background:#f4f4f4; padding:12px; border-radius:6px; font-size:0.9rem;">
Camera Image
  → CLAHE + Adaptive Threshold (OpenCV)
  → Circular Blob Detection (contour filtering)
  → Row/Column Clustering (gap-based segmentation)
  → Braille Cell Dot Pattern Extraction
  → Grade 1 Braille Lookup Table
  → English Text
  → gTTS Text-to-Speech (mp3)
                    </pre>
                    <h3>Stack</h3>
                    <p>Python · OpenCV · NumPy · SciPy · Gradio · gTTS</p>
                    <h3>Future Improvements</h3>
                    <ul>
                        <li>Deep learning dot detector (YOLOv8 fine-tuned on Braille)</li>
                        <li>Grade 2 Braille (contracted) support</li>
                        <li>Real-time video stream (frame-by-frame)</li>
                        <li>Mobile-first PWA with camera alignment guide</li>
                        <li>Multi-language Braille (Tamil, Hindi)</li>
                    </ul>
                </div>
                """)

        # ── Footer ──
        gr.HTML("""
        <div style="text-align:center; margin-top:20px; color:#aaa; font-size:0.85rem;">
            BrailleVision Hackathon 2026 · Built with OpenCV + Gradio · Open Source · Assistive Technology
        </div>
        """)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,       # creates public link for demo
        show_error=True,
        favicon_path=None
    )
