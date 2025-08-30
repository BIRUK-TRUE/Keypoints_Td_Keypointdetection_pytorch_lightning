import numpy as np
import torch
from PIL import Image
from models.detector import KeypointDetector
from models.person_detector import PersonDetector
from utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from inference import _predict_keypoints_on_crop, get_model_from_local_checkpoint

def debug_keypoint_detection():
    # Load model
    model = get_model_from_local_checkpoint("snapshots/SmartJointsBest_resnet34_adamw_0.001.pth")
    
    # Load image
    image = Image.open("testimages/test1.jpg").convert("RGB")
    print(f"Image size: {image.size}")
    
    # Detect people
    person_detector = PersonDetector("fasterrcnn", person_conf_threshold=0.6)
    boxes = person_detector.detect(image)
    print(f"Detected {len(boxes)} people")
    
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        print(f"Person {i}: bbox=({x1}, {y1}, {x2}, {y2})")
        
        # Process crop
        expand_ratio = 0.05
        bw = x2 - x1
        bh = y2 - y1
        ex1 = max(0, int(x1 - expand_ratio * bw))
        ey1 = max(0, int(y1 - expand_ratio * bh))
        ex2 = min(image.size[0], int(x2 + expand_ratio * bw))
        ey2 = min(image.size[1], int(y2 + expand_ratio * bh))
        
        crop = image.crop((ex1, ey1, ex2, ey2))
        print(f"  Crop size: {crop.size}")
        
        # Test different thresholds
        for threshold in [0.1, 0.2, 0.25, 0.3]:
            keypoints = _predict_keypoints_on_crop(model, crop, abs_max_threshold=threshold)
            detected = sum(1 for kp in keypoints if kp is not None)
            print(f"  Threshold {threshold}: {detected}/{len(keypoints)} keypoints")

if __name__ == "__main__":
    debug_keypoint_detection()
