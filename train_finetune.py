from ultralytics import YOLO
from src.config import DATA_YAML_PATH
import os
import glob

def find_best_checkpoint():
    """runs/segment altındaki en son train klasöründe best.pt'yi bulur."""
    # Önce 'train' klasörünü dene
    direct = os.path.join("runs", "segment", "train", "weights", "best.pt")
    if os.path.exists(direct):
        return direct

    # Numbered train-X klasörlerini tara ve en yenisini al
    pattern = os.path.join("runs", "segment", "train-*", "weights", "best.pt")
    candidates = glob.glob(pattern)
    if candidates:
        return max(candidates, key=os.path.getmtime)

    return None

def main():
    checkpoint = find_best_checkpoint()
    if not checkpoint:
        print("HATA: best.pt bulunamadi! Once python train.py calistirin.")
        return

    print(f"Fine-tune baslangic noktasi: {checkpoint}")
    print("Ayarlar: lr0=0.00005, mosaic=OFF, 50 epoch, gercek gorsellere fine-tune...")

    model = YOLO(checkpoint)  # best.pt yükle (yolo26l-seg.pt değil!)

    model.train(
        data=str(DATA_YAML_PATH),
        epochs=50,
        batch=8,
        imgsz=640,
        patience=15,
        save_period=5,

        # ÇOK DÜŞÜK learning rate — modeli bozmadan ince ayar
        optimizer="AdamW",
        lr0=0.00005,        # 20x daha düşük
        lrf=0.1,            # Final LR = 0.000005

        # Mosaic KAPALI — gerçek görsel dağılımına alış
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,

        # Hafif augmentation — model zaten iyi, ezberletme
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        flipud=0.3,
        fliplr=0.5,
        copy_paste=0.0,
        overlap_mask=True,

        device=0,
        val=True,
        name="finetune",    # runs/segment/finetune/ olarak kaydeder
    )

    print("\nFine-tune tamamlandi!")
    print("Sonuclar: runs/segment/finetune/weights/best.pt")

if __name__ == "__main__":
    main()
