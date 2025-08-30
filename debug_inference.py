"""
Debug script for keypoint detection issues with multi-person images
"""
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from models.detector import KeypointDetector
from models.person_detector import PersonDetector
from utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from utils.load_checkpoints import get_model_from_wandb_checkpoint
from inference import _predict_keypoints_on_crop, get_model_from_local_checkpoint


def debug_heatmaps(heatmaps: torch.Tensor, model: KeypointDetector, crop: Image.Image, threshold: float = 0.25):
    """Debug heatmap values and keypoint extraction"""
    print(f"\n=== HEATMAP DEBUG ===")
    print(f"Crop size: {crop.size}")
    print(f"Heatmap shape: {heatmaps.shape}")
    print(f"Threshold: {threshold}")
    
    # Check heatmap statistics
    for i in range(heatmaps.shape[1]):  # for each channel
        channel_heatmap = heatmaps[0, i]
        max_val = channel_heatmap.max().item()
        min_val = channel_heatmap.min().item()
        mean_val = channel_heatmap.mean().item()
        print(f"Channel {i} ({model.keypoint_channel_configuration[i]}): max={max_val:.4f}, min={min_val:.4f}, mean={mean_val:.4f}")
        
        # Check if any values are above threshold
        above_threshold = (channel_heatmap > threshold).sum().item()
        print(f"  Values above threshold {threshold}: {above_threshold}")
        
        if max_val > threshold:
            # Find the maximum location
            max_idx = channel_heatmap.argmax()
            h, w = channel_heatmap.shape
            max_y, max_x = max_idx // w, max_idx % w
            print(f"  Max location: ({max_x}, {max_y}) with value {max_val:.4f}")
    
    # Try keypoint extraction with different thresholds
    print(f"\n=== KEYPOINT EXTRACTION DEBUG ===")
    for test_threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps, abs_max_threshold=test_threshold)
        num_detected = sum(len(kps) for kps in keypoints[0])
        print(f"Threshold {test_threshold}: {num_detected} keypoints detected")
        if num_detected > 0:
            print(f"  Keypoints: {keypoints[0]}")


def debug_person_detection(image: Image.Image, person_detector: PersonDetector):
    """Debug person detection results"""
    print(f"\n=== PERSON DETECTION DEBUG ===")
    print(f"Image size: {image.size}")
    
    boxes = person_detector.detect(image)
    print(f"Detected {len(boxes)} people")
    
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        print(f"Person {i}: bbox=({x1}, {y1}, {x2}, {y2}), size=({x2-x1}, {y2-y1})")
    
    return boxes


def debug_crop_processing(image: Image.Image, boxes: List[Tuple[int, int, int, int]], model: KeypointDetector):
    """Debug crop processing and keypoint detection"""
    print(f"\n=== CROP PROCESSING DEBUG ===")
    
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        print(f"\n--- Processing Person {idx} ---")
        
        # Clamp to image bounds
        x1c = max(0, min(x1, image.size[0] - 1))
        y1c = max(0, min(y1, image.size[1] - 1))
        x2c = max(0, min(x2, image.size[0]))
        y2c = max(0, min(y2, image.size[1]))
        
        if x2c <= x1c or y2c <= y1c:
            print(f"  Invalid crop bounds, skipping")
            continue
            
        print(f"  Original bbox: ({x1}, {y1}, {x2}, {y2})")
        print(f"  Clamped bbox: ({x1c}, {y1c}, {x2c}, {y2c})")
        
        # Expand bbox slightly
        expand_ratio = 0.05
        bw = x2c - x1c
        bh = y2c - y1c
        ex1 = max(0, int(x1c - expand_ratio * bw))
        ey1 = max(0, int(y1c - expand_ratio * bh))
        ex2 = min(image.size[0], int(x2c + expand_ratio * bw))
        ey2 = min(image.size[1], int(y2c + expand_ratio * bh))
        
        print(f"  Expanded bbox: ({ex1}, {ey1}, {ex2}, {ey2})")
        
        crop = image.crop((ex1, ey1, ex2, ey2))
        print(f"  Crop size: {crop.size}")
        
        # Test different thresholds
        for threshold in [0.1, 0.2, 0.25, 0.3, 0.4]:
            crop_kps = _predict_keypoints_on_crop(model, crop, abs_max_threshold=threshold)
            detected_count = sum(1 for kp in crop_kps if kp is not None)
            print(f"  Threshold {threshold}: {detected_count}/{len(crop_kps)} keypoints detected")
            
            if detected_count > 0:
                print(f"    Detected keypoints: {crop_kps}")


def visualize_debug_results(image: Image.Image, boxes: List[Tuple[int, int, int, int]], 
                          model: KeypointDetector, person_detector: PersonDetector):
    """Create debug visualization"""
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)
    
    # Draw person detection boxes
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Original box in red
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1, y1-20), f"Person {i}", fill=(255, 0, 0))
        
        # Expanded box in blue
        expand_ratio = 0.05
        bw = x2 - x1
        bh = y2 - y1
        ex1 = max(0, int(x1 - expand_ratio * bw))
        ey1 = max(0, int(y1 - expand_ratio * bh))
        ex2 = min(image.size[0], int(x2 + expand_ratio * bw))
        ey2 = min(image.size[1], int(y2 + expand_ratio * bh))
        draw.rectangle([ex1, ey1, ex2, ey2], outline=(0, 0, 255), width=1)
    
    return debug_img


def main():
    """Main debugging function"""
    print("=== KEYPOINT DETECTION DEBUG SCRIPT ===")
    
    # Configuration - adjust these paths for your setup
    local_checkpoint_path = "snapshots/SmartJointsBest_resnet34_adamw_0.001.pth"
    image_path = "testimages/test1.jpg"  # Change to your test image
    
    # Load model
    print(f"Loading model from: {local_checkpoint_path}")
    try:
        model = get_model_from_local_checkpoint(local_checkpoint_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load image
    print(f"Loading image from: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"✓ Image loaded: {image.size}")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return
    
    # Initialize person detector
    person_detector = PersonDetector("fasterrcnn", person_conf_threshold=0.6)
    print("✓ Person detector initialized")
    
    # Debug person detection
    boxes = debug_person_detection(image, person_detector)
    
    if len(boxes) == 0:
        print("✗ No people detected! Try lowering the confidence threshold.")
        return
    
    # Debug crop processing
    debug_crop_processing(image, boxes, model)
    
    # Create debug visualization
    debug_img = visualize_debug_results(image, boxes, model, person_detector)
    debug_img.save("debug_visualization.png")
    print("✓ Debug visualization saved as 'debug_visualization.png'")
    
    # Test with a single crop to see heatmaps
    if len(boxes) > 0:
        print(f"\n=== DETAILED HEATMAP ANALYSIS ===")
        x1, y1, x2, y2 = boxes[0]
        expand_ratio = 0.05
        bw = x2 - x1
        bh = y2 - y1
        ex1 = max(0, int(x1 - expand_ratio * bw))
        ey1 = max(0, int(y1 - expand_ratio * bh))
        ex2 = min(image.size[0], int(x2 + expand_ratio * bw))
        ey2 = min(image.size[1], int(y2 + expand_ratio * bh))
        
        crop = image.crop((ex1, ey1, ex2, ey2))
        
        # Get heatmaps directly
        tensored_image = torch.from_numpy(np.array(crop)).float()
        tensored_image = tensored_image / 255.0
        tensored_image = tensored_image.permute(2, 0, 1)
        tensored_image = tensored_image.unsqueeze(0)
        
        with torch.no_grad():
            heatmaps = model(tensored_image)
        
        debug_heatmaps(heatmaps, model, crop)
    
    print(f"\n=== DEBUG COMPLETE ===")
    print("Check the output above for potential issues:")
    print("1. Are people being detected?")
    print("2. Are crop sizes reasonable?")
    print("3. Are heatmap values above the threshold?")
    print("4. Are keypoints being extracted from heatmaps?")


if __name__ == "__main__":
    main()
