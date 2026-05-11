import numpy as np

class SeverityAnalyzer:
    def __init__(self, thresholds):
        """
        Initializes the severity analyzer with predefined thresholds.
        """
        self.thresholds = thresholds

    def analyze(self, detection):
        """
        Calculates severity ratio and determines the level based on mask_area / bbox_area.
        """
        box = detection["box"]
        mask = detection["mask"]

        # Calculate bounding box area
        x1, y1, x2, y2 = map(int, box)
        bbox_area = max(1, (x2 - x1) * (y2 - y1))

        # Calculate mask area (sum of True pixels)
        # Assuming mask is a binary mask where 1 represents the segmented object
        mask_crop = mask[y1:y2, x1:x2]
        mask_area = np.sum(mask_crop > 0.5)

        # Calculate ratio
        ratio = mask_area / bbox_area
        
        # Determine severity level
        severity_level = "Unknown"
        severity_color = (255, 255, 255)

        for t in self.thresholds:
            if ratio <= t["max_ratio"]:
                severity_level = t["level"]
                severity_color = t["color"]
                break
        
        # If ratio > highest max_ratio (should theoretically be handled if max is 1.0)
        if severity_level == "Unknown":
            severity_level = self.thresholds[-1]["level"]
            severity_color = self.thresholds[-1]["color"]

        return {
            "ratio": ratio,
            "level": severity_level,
            "color": severity_color,
            "mask_area": int(mask_area),
            "bbox_area": int(bbox_area)
        }
