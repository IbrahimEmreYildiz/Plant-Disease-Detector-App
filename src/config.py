import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Data Paths
RAW_DATA_DIR = DATA_DIR / "plantseg" / "plantsegv2"
YOLO_DATA_DIR = DATA_DIR / "plantseg_yolo"
DATA_YAML_PATH = YOLO_DATA_DIR / "data.yaml"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters optimized for high mAP50 (>80)
TRAIN_CONFIG = {
    "model": "yolo26l-seg.pt",  # Large model for maximum accuracy
    "data": str(DATA_YAML_PATH),
    "epochs": 150,              # More epochs for better convergence
    "batch": 8,                 # Fits in 8GB VRAM for Large model
    "imgsz": 640,
    "patience": 30,             # Increased patience so early stopping doesn't fire too soon
    "save_period": 5,           # Checkpoint every 5 epochs for resume support
    "optimizer": "AdamW",       # AdamW is more stable than SGD for small datasets
    "lr0": 0.001,               # Lower starting LR works better with AdamW
    "lrf": 0.01,                # Final LR = lr0 * lrf = 0.00001 (proper cosine decay)
    "warmup_epochs": 5.0,       # Warmup prevents early instability
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "close_mosaic": 15,         # Disable mosaic last 15 epochs for fine-tuning
    "mosaic": 1.0,              # Strong augmentation during main training
    "mixup": 0.15,              # Slightly stronger mixup
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "bgr": 0.0,
    "copy_paste": 0.15,         # Slightly stronger copy_paste for segmentation
    "overlap_mask": True,       # Better mask quality
    "device": 0                 # GPU
}

# Inference Config
INFERENCE_CONFIG = {
    "conf": 0.25,              # Confidence threshold
    "iou": 0.45                # NMS IOU threshold
}

# Severity Thresholds (Ratio = mask_area / bbox_area)
SEVERITY_THRESHOLDS = [
    {"level": "Low", "max_ratio": 0.20, "color": (0, 255, 0)},       # Green
    {"level": "Moderate", "max_ratio": 0.50, "color": (0, 165, 255)},# Orange
    {"level": "High", "max_ratio": 0.80, "color": (0, 0, 255)},      # Red
    {"level": "Critical", "max_ratio": 1.00, "color": (255, 0, 255)} # Purple
]
