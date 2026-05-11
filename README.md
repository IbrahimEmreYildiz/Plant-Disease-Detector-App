# 🌿 Crop Disease Intelligence System

A high-performance plant disease detection and severity analysis system powered by **YOLO26l-seg** (Large) instance segmentation, fine-tuned on the **PlantSeg** dataset.

## 🚀 Key Features

- **Advanced AI Architecture:** Uses the latest YOLO26l-seg model for pixel-perfect disease mask extraction.
- **9 Specialized Classes:** Optimized for the most critical field crop diseases (Wheat, Apple, Grape, Zucchini, etc.).
- **Real-time Severity Analysis:** Calculates the percentage of leaf infection to provide "Low" to "Critical" severity scores.
- **Premium UI:** Custom **Blue Gradient Glassmorphism** interface for professional project demonstrations.
- **Optimized Video Processing:** **5x faster inference** using intelligent frame striding (processes every 5th frame and reuses detections).

## 🔬 Dataset & Training

- **Source:** [PlantSeg](https://zenodo.org/records/13958858) — largest in-the-wild plant disease segmentation dataset.
- **Filtering:** Optimized from 118 classes down to **9 high-performing classes** to maximize mAP.
- **Model:** YOLO26l-seg (Large model for maximum accuracy).
- **Results (train-2):** Peak **mAP50 (Box): 0.578**, **mAP50 (Mask): 0.558**.

### 📋 Supported Disease Classes
1. **Apple Scab** (Elma Karaleke)
2. **Citrus Canker** (Narenciye Kankeri)
3. **Grape Downy Mildew** (Üzüm Mildiö)
4. **Soybean Frog Eye Leaf Spot** (Soya Fasulyesi Kurbağa Gözü Lekesi)
5. **Wheat Head Scab** (Buğday Başak Yanıklığı)
6. **Wheat Powdery Mildew** (Buğday Külleme)
7. **Wheat Septoria Blotch** (Buğday Septoria Leke)
8. **Wheat Stripe Rust** (Buğday Sarı Pas)
9. **Zucchini Powdery Mildew** (Kabak Külleme)

## 📊 Severity Levels

| Level | Ratio | Color | Action |
|-------|-------|-------|--------|
| 🟢 **Low** | 0–20% | Green | Monitor |
| 🟠 **Moderate** | 20–50% | Orange | Consider treatment |
| 🔴 **High** | 50–80% | Red | Urgent action |
| 🟣 **Critical** | 80%+ | Purple | Severe damage |

## 🏗️ Project Architecture

```text
Input (Image/Video) → YOLO26l-seg → Mask Extraction → Severity Scoring → Visualizer → Output (Gradio UI)
```

## 🧠 Model Weights

To use the fine-tuned disease detection capabilities, you need to download the trained weights:

1.  **Download:** [Click here to download the `best.pt` model weights from Google Drive](https://drive.google.com/file/d/1uWVzIAQnqvM9a2-DP9Pf9A2JE402J66f/view?usp=drive_link)
2.  **Install:** Place the downloaded `best.pt` file directly into the **project root directory** (the same folder as `app.py`). 
3.  **Verification:** When you run `python app.py`, look for the message `Loading model weights from: best.pt` in the terminal. If you see "Using base pretrained model", the custom weights were not found.

## 🚀 Installation & Usage

### 1. Setup Environment
```bash
git clone https://github.com/IbrahimEmreYildiz/crop-disease-detector
cd crop-disease-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Application
```bash
python app.py
```
Open **`http://localhost:7860`** in your browser.

## 📁 Project Structure

- `app.py`: Main Gradio web application with custom blue theme.
- `train_finetune.py`: Script used for the 100-epoch specialized fine-tuning.
- `src/detector.py`: Core inference engine.
- `src/severity_analyzer.py`: Geometric analysis of disease masks.
- `src/config.py`: Centralized configuration for all UI and AI parameters.

## 🛠️ Tech Stack

- **AI:** YOLO26 (Ultralytics), PyTorch.
- **Image Processing:** OpenCV, Numpy.
- **Interface:** Gradio (Custom CSS).
- **Backend:** Python 3.10+.

## 📄 Citation
*Based on the PlantSeg dataset by Wei et al. (2024).*