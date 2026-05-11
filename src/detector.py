import cv2
import numpy as np
from ultralytics import YOLO

class DiseaseDetector:
    def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.45):
        """
        Initializes the YOLO26 model.
        """
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
    def detect(self, image):
        """
        Runs inference on an image and extracts boxes, masks, and classes.
        """
        # Run inference
        results = self.model(image, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)[0]
        
        detections = []
        
        if results.boxes is not None and results.masks is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            masks = results.masks.data.cpu().numpy()
            
            # Masks from ultralytics might be in different size than original image, we need to resize them
            orig_h, orig_w = image.shape[:2]
            
            for box, conf, cls_id, mask in zip(boxes, confidences, class_ids, masks):
                # Resize mask to original image size
                mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                detections.append({
                    "box": box, # [x1, y1, x2, y2]
                    "mask": mask_resized,
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": self.model.names[cls_id]
                })
                
        return detections
