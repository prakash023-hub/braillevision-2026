# 👁️ BrailleVision — Real-time Physical Braille to English

> **BrailleVision Hackathon 2026 Submission**  
> Camera-based AI system that converts real physical embossed Braille into English text and speech — no Braille literacy required.

---

## 🎯 What It Does

BrailleVision takes a **camera image or webcam feed of real physical Braille paper** — embossed, handwritten (slate & stylus), or Braille writer output — and converts it into:

- ✅ English text (Grade 1 Braille)
- ✅ Text-to-speech audio output
- ✅ Annotated image showing detected dot positions

This is **not** a Unicode Braille converter. The system processes real camera images of raised dots using computer vision.

---

## 🌟 Why This Matters

Braille is the essential reading system for millions of visually impaired individuals. Yet caregivers, teachers, volunteers, and accessibility workers often cannot read physical Braille quickly.

**Use case examples:**
- A nurse reading a patient's Braille medication notes
- A teacher in an inclusive classroom understanding a student's Braille worksheet  
- A volunteer reading Braille labels at a library for visually impaired patrons
- A caregiver reading prescription bottle Braille instructions

BrailleVision bridges this gap — point a phone camera at Braille paper, get instant English text and audio.

---

## 🔧 Technical Pipeline

```
Camera Image
  ↓
CLAHE Contrast Enhancement (OpenCV)
  ↓
Adaptive Thresholding (handles uneven lighting)
  ↓
Gaussian Denoising
  ↓
Circular Contour Detection (area + circularity filters)
  ↓
Row Clustering (gap-based, gap threshold = 0.9 × median dot height)
  ↓
Braille Line Grouping (3 dot-rows per Braille character line)
  ↓
Cell Segmentation (x-gap based, threshold = 1.75 × median dot width)
  ↓
Dot Pattern Extraction (2×3 grid → dot positions 1–6)
  ↓
Grade 1 Braille Lookup Table (64 patterns → characters)
  ↓
Number / Capital Indicator Handling
  ↓
English Text Output
  ↓
gTTS Text-to-Speech (mp3)
```

---

## 🚀 Quick Start

### Requirements

```bash
pip install opencv-python numpy gradio gtts pillow scipy
```

Python 3.9+ required.

### Run the App

```bash
cd src
python app.py
```

Open http://localhost:7860 in your browser.

The app also generates a public Gradio share link automatically.

### Project Structure

```
braillevision/
├── src/
│   ├── app.py          # Gradio web application (main entry point)
│   ├── detector.py     # Core computer vision pipeline
│   ├── braille_map.py  # Grade 1 Braille lookup tables
│   └── tts.py          # Text-to-speech (gTTS + pyttsx3 fallback)
├── output/             # Demo output images
├── requirements.txt
└── README.md
```

---

## 📷 How to Get Best Results with Real Braille

1. **Raking light:** Place a lamp at a **45° angle** to the paper — this creates shadows on raised dots, making them visible to the camera
2. **Steady camera:** Hold phone/webcam parallel to the paper surface
3. **Zoom in:** Dots should be at least 15–20px wide in the captured image
4. **Avoid glare:** Diffuse light works better than direct flash
5. **Sharp focus:** Blurry images reduce detection accuracy

---

## 🧪 Demo Mode (No Physical Braille Needed)

The app includes a **synthetic Braille generator** — type any word and the system generates a realistic synthetic Braille dot image, then runs the full detection pipeline on it. Perfect for testing and demonstration.

---

## 📊 Performance Notes

| Condition | Accuracy |
|-----------|----------|
| Synthetic Braille (controlled) | ~85% |
| High-contrast real Braille photo | ~70–80% (estimated) |
| Low-light / blurry images | ~30–50% |

**Key factors affecting accuracy:**
- Lighting quality (raking light essential for real Braille)
- Camera resolution and focus
- Braille quality (deep embossing vs shallow)
- Paper-to-dot contrast ratio

---

## ♿ Accessibility Features

- **Auto-play TTS:** Audio output plays immediately after detection
- **High-contrast overlay:** Green circles mark detected dots on annotated image
- **Camera guidance tips:** Built-in instructions for optimal positioning
- **Large text output panel:** Decoded text displayed prominently
- **Web-based:** No installation for end users — works in any browser
- **Webcam support:** Live capture from any connected camera

---

## 🔬 How Physical Braille Detection Works

Unlike digital Braille (Unicode symbols ⠓⠑⠇⠇⠕), physical Braille consists of **raised dots** on paper. Under raking light, these dots cast **shadows** that create detectable dark regions in the image.

Our approach:
1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) normalizes uneven lighting across the image
2. **Adaptive thresholding** converts the image to binary, isolating dark shadow regions cast by raised dots
3. **Contour filtering** selects blobs that are approximately circular (circularity > 0.45) and the right size (20–2500 px²) — these are the dot shadows
4. **Gap-based clustering** groups detected dots into rows and cells using the statistical distribution of detected dot sizes as reference
5. **Pattern extraction** maps each 2×3 dot grid to a Braille dot number pattern (positions 1–6)
6. **Lookup table** converts patterns to characters with special handling for number indicators and capital indicators

---

## 🛣️ Future Improvements

- **Deep learning dot detector:** Fine-tune YOLOv8 on a labeled Braille dot dataset for robustness under difficult lighting
- **Grade 2 Braille:** Support for contracted Braille (significantly reduces character count)
- **Real-time video stream:** Frame-by-frame processing with temporal smoothing
- **Mobile PWA:** Camera alignment guide with real-time feedback (distance, angle, lighting quality indicator)
- **Multi-language Braille:** Tamil, Hindi, and other Indian language Braille standards
- **Perspective correction:** Auto-detect and correct camera tilt relative to Braille page
- **Confidence feedback:** Visual and audio cues when detection confidence is low

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Image processing | OpenCV 4.x |
| Numerical computation | NumPy |
| Web interface | Gradio |
| Text-to-speech | gTTS (online) / pyttsx3 (offline fallback) |
| Language | Python 3.9+ |
| Deployment | Local / Gradio Share Link / HuggingFace Spaces |

---

## 📜 What Was Built During the Hackathon

- ✅ Complete computer vision pipeline (no pre-trained model — pure classical CV)
- ✅ Grade 1 Braille lookup table (64 patterns, numbers, punctuation, indicators)
- ✅ Adaptive preprocessing for real-world lighting conditions
- ✅ Gradio web UI with webcam and image upload support
- ✅ TTS integration with offline fallback
- ✅ Synthetic Braille generator for testing
- ✅ Accessibility-focused UI design

**Open-source libraries used:** OpenCV, NumPy, Gradio, gTTS, Pillow — all standard libraries, no pre-built Braille recognition systems.

---

## 👤 About

Built for BrailleVision Hackathon 2026.  
Healthcare AI × Computer Vision × Assistive Technology.
