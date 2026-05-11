import json
import os
import time

class ReportGenerator:
    def __init__(self, output_dir):
        """
        Initializes the report generator.
        """
        self.output_dir = output_dir
        self.report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames_analyzed": 0,
            "overall_severity_counts": {
                "Low": 0,
                "Moderate": 0,
                "High": 0,
                "Critical": 0
            },
            "detections": []
        }

    def add_frame(self, frame_index, detections_with_severity):
        """
        Adds detections from a single frame to the report.
        """
        self.report_data["frames_analyzed"] += 1
        
        frame_summary = {
            "frame_index": frame_index,
            "timestamp_ms": time.time() * 1000,
            "objects": []
        }
        
        for det in detections_with_severity:
            sev_level = det["severity"]["level"]
            self.report_data["overall_severity_counts"][sev_level] += 1
            
            frame_summary["objects"].append({
                "class_name": det["class_name"],
                "confidence": float(det["confidence"]),
                "box": [float(x) for x in det["box"]],
                "severity_level": sev_level,
                "severity_ratio": float(det["severity"]["ratio"]),
                "mask_area": det["severity"]["mask_area"],
                "bbox_area": det["severity"]["bbox_area"]
            })
            
        self.report_data["detections"].append(frame_summary)

    def generate_json(self, filename="report.json"):
        """
        Exports the aggregated report to a JSON file.
        """
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.report_data, f, indent=4)
        return output_path

    def generate_text_report(self, filename="report.txt"):
        """
        Exports the aggregated report to a human-readable text file.
        """
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write(" 🌿 CROP DISEASE INTELLIGENCE SYSTEM - ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Date & Time: {self.report_data['timestamp']}\n")
            f.write(f"Total Frames Analyzed: {self.report_data['frames_analyzed']}\n\n")
            
            f.write("--- OVERALL SEVERITY SUMMARY ---\n")
            for level, count in self.report_data['overall_severity_counts'].items():
                f.write(f" > {level} Cases: {count}\n")
            f.write("\n")
            
            f.write("--- DETAILED DETECTIONS ---\n")
            for frame in self.report_data["detections"]:
                if self.report_data['frames_analyzed'] > 1:
                    f.write(f"\n[Frame {frame['frame_index']}]\n")
                
                if not frame["objects"]:
                    f.write("No diseases detected.\n")
                else:
                    for i, obj in enumerate(frame["objects"], 1):
                        f.write(f"{i}. Disease: {obj['class_name'].upper()}\n")
                        f.write(f"   Severity : {obj['severity_level']} (Affects {obj['severity_ratio']*100:.1f}% of the leaf)\n")
                        f.write(f"   AI Conf. : {obj['confidence']*100:.1f}%\n")
                f.write("-" * 40 + "\n")
                
        return output_path
