import os
import json
import shutil
import cv2
from pathlib import Path
from src.config import RAW_DATA_DIR, YOLO_DATA_DIR, DATA_YAML_PATH

def convert_coco_to_yolo():
    """
    Converts PlantSeg COCO format annotations to YOLO segmentation format.
    Filters the dataset to ONLY the TOP 9 classes (zucchini powdery mildew excluded
    due to very low mAP50 of 0.187 caused by high instance density per image).
    """
    print(f"Checking dataset at {RAW_DATA_DIR}")
    if not RAW_DATA_DIR.exists():
        print(f"Error: Raw dataset not found at {RAW_DATA_DIR}.")
        print("Please download the PlantSeg dataset from Zenodo and place it there.")
        return

    annotations_file = RAW_DATA_DIR / "coco_annotations.json"
    if not annotations_file.exists():
        print(f"Error: COCO annotations not found at {annotations_file}")
        return

    with open(annotations_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 1. Find Top 10 classes with the most images
    class_image_counts = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        img_id = ann['image_id']
        if cat_id not in class_image_counts:
            class_image_counts[cat_id] = set()
        class_image_counts[cat_id].add(img_id)

    category_counts = [(cat_id, len(img_set)) for cat_id, img_set in class_image_counts.items()]
    category_counts.sort(key=lambda x: x[1], reverse=True)
    top_10_categories = category_counts[:9]  # Top 9: zucchini powdery mildew excluded
    
    valid_category_ids = {cat_id for cat_id, count in top_10_categories}
    
    original_categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    yolo_categories = {}
    current_yolo_id = 0
    for cat_id in sorted(list(valid_category_ids)):
        yolo_categories[cat_id] = current_yolo_id
        current_yolo_id += 1

    print(f"Filtered classes from {len(original_categories)} down to TOP 9 classes for maximum mAP.")
    for cat_id, count in top_10_categories:
        print(f" - {original_categories[cat_id]}: {count} images")

    # Clean old YOLO directory to prevent ghost labels
    if YOLO_DATA_DIR.exists():
        print(f"Cleaning old YOLO dataset at {YOLO_DATA_DIR}...")
        shutil.rmtree(YOLO_DATA_DIR)

    # 2. Map images to their original splits (train, val, test)
    img_filename_to_split = {}
    for split in ['train', 'val', 'test']:
        split_dir = RAW_DATA_DIR / "images" / split
        if split_dir.exists():
            for f in os.listdir(split_dir):
                img_filename_to_split[f] = split

    # 3. Setup YOLO directory structure
    for split in ['train', 'val', 'test']:
        (YOLO_DATA_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (YOLO_DATA_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    images = {img['id']: img for img in coco_data['images']}

    print("Converting annotations to YOLO segmentation format...")
    # 4. Process annotations
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        if cat_id not in valid_category_ids:
            continue
            
        img_id = ann['image_id']
        img_info = images[img_id]
        raw_file_name = img_info['file_name']
        
        cat_name = original_categories[cat_id].replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
        actual_file_name = None
        if raw_file_name in img_filename_to_split:
            actual_file_name = raw_file_name
        elif f"{cat_name}_{raw_file_name}" in img_filename_to_split:
            actual_file_name = f"{cat_name}_{raw_file_name}"
            
        if not actual_file_name:
            continue # Image file not found in any split directory
            
        split = img_filename_to_split[actual_file_name]
            
        img_width = img_info['width']
        img_height = img_info['height']
        
        yolo_class_id = yolo_categories[cat_id]
        
        segmentations = ann['segmentation']
        for poly in segmentations:
            normalized_poly = []
            for i in range(0, len(poly), 2):
                x = poly[i] / img_width
                y = poly[i+1] / img_height
                normalized_poly.extend([x, y])
            
            label_path = YOLO_DATA_DIR / split / "labels" / f"{actual_file_name.rsplit('.', 1)[0]}.txt"
            with open(label_path, 'a') as f:
                line = f"{yolo_class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_poly]) + "\n"
                f.write(line)

    print("Copying images to YOLO structure...")
    # 5. Copy images
    for img_id, img_info in images.items():
        raw_file_name = img_info['file_name']
        
        valid_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id and ann['category_id'] in valid_category_ids]
        if not valid_anns:
            continue
            
        cat_id = valid_anns[0]['category_id']
        cat_name = original_categories[cat_id].replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
        
        actual_file_name = None
        if raw_file_name in img_filename_to_split:
            actual_file_name = raw_file_name
        elif f"{cat_name}_{raw_file_name}" in img_filename_to_split:
            actual_file_name = f"{cat_name}_{raw_file_name}"
            
        if not actual_file_name:
            continue
            
        split = img_filename_to_split[actual_file_name]
            
        src_img = RAW_DATA_DIR / "images" / split / actual_file_name
        dst_img = YOLO_DATA_DIR / split / "images" / actual_file_name
        
        if src_img.exists() and not dst_img.exists():
            shutil.copy(src_img, dst_img)

    # 6. Create data.yaml
    yaml_content = f"""path: {YOLO_DATA_DIR.resolve()}
train: train/images
val: val/images
test: test/images

nc: {len(valid_category_ids)}
names:
"""
    for orig_id, yolo_id in yolo_categories.items():
        yaml_content += f"  {yolo_id}: {original_categories[orig_id]}\n"

    with open(DATA_YAML_PATH, 'w') as f:
        f.write(yaml_content)

    print(f"Conversion complete! data.yaml created at {DATA_YAML_PATH}")

if __name__ == "__main__":
    convert_coco_to_yolo()
