from ultralytics import YOLO
from src.config import TRAIN_CONFIG
import sys

def main():
    print(f"Starting training with model: {TRAIN_CONFIG['model']}")
    print(f"Targeting high mAP50 (>80) with {TRAIN_CONFIG['epochs']} epochs and strong augmentations...")
    
    try:
        # Initialize YOLO model
        model = YOLO(TRAIN_CONFIG['model'])
        
        # Start training using configuration optimized for high mAP50
        results = model.train(
            data=TRAIN_CONFIG['data'],
            epochs=TRAIN_CONFIG['epochs'],
            batch=TRAIN_CONFIG['batch'],
            imgsz=TRAIN_CONFIG['imgsz'],
            patience=TRAIN_CONFIG['patience'],
            save_period=TRAIN_CONFIG['save_period'],
            optimizer=TRAIN_CONFIG['optimizer'],
            lr0=TRAIN_CONFIG['lr0'],
            lrf=TRAIN_CONFIG['lrf'],
            warmup_epochs=TRAIN_CONFIG['warmup_epochs'],
            warmup_momentum=TRAIN_CONFIG['warmup_momentum'],
            warmup_bias_lr=TRAIN_CONFIG['warmup_bias_lr'],
            close_mosaic=TRAIN_CONFIG['close_mosaic'],
            mosaic=TRAIN_CONFIG['mosaic'],
            mixup=TRAIN_CONFIG['mixup'],
            hsv_h=TRAIN_CONFIG['hsv_h'],
            hsv_s=TRAIN_CONFIG['hsv_s'],
            hsv_v=TRAIN_CONFIG['hsv_v'],
            degrees=TRAIN_CONFIG['degrees'],
            translate=TRAIN_CONFIG['translate'],
            scale=TRAIN_CONFIG['scale'],
            shear=TRAIN_CONFIG['shear'],
            perspective=TRAIN_CONFIG['perspective'],
            flipud=TRAIN_CONFIG['flipud'],
            fliplr=TRAIN_CONFIG['fliplr'],
            copy_paste=TRAIN_CONFIG['copy_paste'],
            overlap_mask=TRAIN_CONFIG['overlap_mask'],
            device=TRAIN_CONFIG['device'],
            val=True
        )
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
