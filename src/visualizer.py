import cv2
import numpy as np
import supervision as sv

class Visualizer:
    def __init__(self):
        """
        Initializes the visualizer module. We'll use a mix of OpenCV and supervision.
        """
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
        self.mask_annotator = sv.MaskAnnotator()

    def draw(self, image, detections_with_severity):
        """
        Draws boxes, masks, and severity labels on the image.
        detections_with_severity is a list of dicts containing 'box', 'mask', 'confidence', 'class_id', 'class_name', 'severity'
        """
        if not detections_with_severity:
            return image
            
        annotated_image = image.copy()
        
        # Format detections for supervision
        boxes = []
        masks = []
        class_ids = []
        confidences = []
        labels = []
        colors = []
        
        for det in detections_with_severity:
            boxes.append(det["box"])
            masks.append(det["mask"] > 0.5) # ensure boolean
            class_ids.append(det["class_id"])
            confidences.append(det["confidence"])
            
            # BGR color for supervision Palette
            # sv.Color takes RGB, so we convert from BGR (as defined in config)
            b, g, r = det["severity"]["color"]
            colors.append(sv.Color(r, g, b))

        # We can't directly map individual colors easily with supervision if they are dynamically assigned per object.
        # So we'll draw them manually using OpenCV for better control over individual colors based on severity.
        
        for i, det in enumerate(detections_with_severity):
            box = boxes[i].astype(int)
            mask = masks[i]
            color = det["severity"]["color"] # BGR format
            
            x1, y1, x2, y2 = box
            
            # Draw mask overlay manually
            color_mask = np.zeros_like(annotated_image, dtype=np.uint8)
            color_mask[mask] = color
            
            annotated_image = cv2.addWeighted(annotated_image, 1.0, color_mask, 0.5, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Label format: Concise version to prevent overflow
            class_abbr = det["class_name"][:15] + ".." if len(det["class_name"]) > 15 else det["class_name"]
            sev_lvl = det["severity"]["level"][:4] # E.g. 'High' or 'Crit'
            label = f'{class_abbr} ({sev_lvl} {det["severity"]["ratio"]*100:.0f}%)'
            
            font_scale = 0.45
            font_thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # Ensure text box doesn't go above or left of the image frame
            y_text = max(y1, text_h + 10)
            x_text = max(x1, 0)
            # Ensure it doesn't go off the right edge either
            if x_text + text_w > annotated_image.shape[1]:
                x_text = annotated_image.shape[1] - text_w
            
            # Add background for text to make it readable
            cv2.rectangle(annotated_image, (x_text, y_text - text_h - 10), (x_text + text_w, y_text), color, -1)
            
            # Black text or white text depending on background brightness
            text_color = (0, 0, 0) if (color[0] + color[1] + color[2]) > 382 else (255, 255, 255)
            cv2.putText(annotated_image, label, (x_text, y_text - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

        return annotated_image
