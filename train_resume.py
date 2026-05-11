from ultralytics import YOLO
import os
import glob

def find_latest_checkpoint():
    """runs/segment altindaki en son egitim klasorunde last.pt'yi bulur."""
    # Once 'train' klasorunu kontrol et (yeni egitim)
    direct = os.path.join("runs", "segment", "train", "weights", "last.pt")
    if os.path.exists(direct):
        return direct
    # Sonra train-N formatindaki klasorlere bak
    pattern = os.path.join("runs", "segment", "*", "weights", "last.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def resume_training():
    weights_path = find_latest_checkpoint()

    if not weights_path:
        print("HATA: Hicbir checkpoint bulunamadi. Once 'python train.py' ile egitimi baslatmaniz gerekiyor.")
        return

    print(f"Checkpoint bulundu: {weights_path}")
    print("Egitim kaldigi yerden devam ettiriliyor...")

    model = YOLO(weights_path)

    # resume=True: epoch sayisi, optimizer state, scheduler vb. aynen devam eder
    model.train(resume=True)

    print("Egitim tamamlandi!")

if __name__ == "__main__":
    resume_training()
