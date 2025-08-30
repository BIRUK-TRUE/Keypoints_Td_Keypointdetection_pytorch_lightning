import numpy as np
import torch
from PIL import Image
from models.detector import KeypointDetector
from models.person_detector import PersonDetector
from utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from inference import _predict_keypoints_on_crop, get_model_from_local_checkpoint

def debug_keypoint_detection():
    print("=== COMPREHENSIVE KEYPOINT DEBUGGING ===")
    
    # Step 1: Check if model file exists
    model_path = "snapshots/SmartJointsBest_resnet34_adamw_0.001.pth"
    print(f"1. Checking model file: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"   ✓ Model file loaded successfully")
        print(f"   ✓ Checkpoint keys: {list(checkpoint.keys())}")
        if 'model' in checkpoint:
            print(f"   ✓ Model object found in checkpoint")
        if 'model_state_dict' in checkpoint:
            print(f"   ✓ Model state dict found in checkpoint")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    # Step 2: Load model
    print(f"\n2. Loading model...")
    try:
        model = get_model_from_local_checkpoint(model_path)
        print(f"   ✓ Model loaded successfully")
        print(f"   ✓ Model device: {next(model.parameters()).device}")
        print(f"   ✓ Keypoint channels: {model.keypoint_channel_configuration}")
        print(f"   ✓ Number of channels: {len(model.keypoint_channel_configuration)}")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    # Step 3: Check image
    image_path = "testimages/test1.jpg"
    print(f"\n3. Loading image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"   ✓ Image loaded: {image.size}")
        print(f"   ✓ Image mode: {image.mode}")
    except Exception as e:
        print(f"   ✗ Error loading image: {e}")
        return
    
    # Step 4: Person detection
    print(f"\n4. Person detection...")
    try:
        person_detector = PersonDetector("fasterrcnn", person_conf_threshold=0.3)  # Lower threshold
        boxes = person_detector.detect(image)
        print(f"   ✓ Person detector initialized")
        print(f"   ✓ Detected {len(boxes)} people")
        
        if len(boxes) == 0:
            print(f"   ⚠️  No people detected! Trying with even lower threshold...")
            person_detector = PersonDetector("fasterrcnn", person_conf_threshold=0.1)
            boxes = person_detector.detect(image)
            print(f"   ✓ With threshold 0.1: {len(boxes)} people detected")
        
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            print(f"   Person {i}: bbox=({x1}, {y1}, {x2}, {y2}), size=({x2-x1}, {y2-y1})")
    except Exception as e:
        print(f"   ✗ Error in person detection: {e}")
        return
    
    if len(boxes) == 0:
        print(f"\n❌ No people detected! This is the main issue.")
        print(f"   Try using a different image or lowering the person detection threshold further.")
        return
    
    # Step 5: Test with first person crop
    print(f"\n5. Testing keypoint detection on first person...")
    x1, y1, x2, y2 = boxes[0]
    
    # Expand bbox
    expand_ratio = 0.05
    bw = x2 - x1
    bh = y2 - y1
    ex1 = max(0, int(x1 - expand_ratio * bw))
    ey1 = max(0, int(y1 - expand_ratio * bh))
    ex2 = min(image.size[0], int(x2 + expand_ratio * bw))
    ey2 = min(image.size[1], int(y2 + expand_ratio * bh))
    
    crop = image.crop((ex1, ey1, ex2, ey2))
    print(f"   ✓ Crop size: {crop.size}")
    
    # Step 6: Test preprocessing
    print(f"\n6. Testing preprocessing...")
    try:
        # Convert to tensor
        tensored_image = torch.from_numpy(np.array(crop)).float()
        print(f"   ✓ Converted to tensor: {tensored_image.shape}")
        
        # Normalize
        tensored_image = tensored_image / 255.0
        print(f"   ✓ Normalized to [0,1]: min={tensored_image.min():.3f}, max={tensored_image.max():.3f}")
        
        # Permute channels
        tensored_image = tensored_image.permute(2, 0, 1)
        print(f"   ✓ Permuted channels: {tensored_image.shape}")
        
        # Add batch dimension
        tensored_image = tensored_image.unsqueeze(0)
        print(f"   ✓ Added batch dimension: {tensored_image.shape}")
        
    except Exception as e:
        print(f"   ✗ Error in preprocessing: {e}")
        return
    
    # Step 7: Test model inference
    print(f"\n7. Testing model inference...")
    try:
        model.eval()
        with torch.no_grad():
            heatmaps = model(tensored_image)
        print(f"   ✓ Model inference successful")
        print(f"   ✓ Heatmap shape: {heatmaps.shape}")
        print(f"   ✓ Heatmap device: {heatmaps.device}")
        
        # Check heatmap statistics
        print(f"   ✓ Heatmap stats: min={heatmaps.min():.4f}, max={heatmaps.max():.4f}, mean={heatmaps.mean():.4f}")
        
    except Exception as e:
        print(f"   ✗ Error in model inference: {e}")
        return
    
    # Step 8: Test keypoint extraction with different thresholds
    print(f"\n8. Testing keypoint extraction...")
    for threshold in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        try:
            keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps, abs_max_threshold=threshold)
            num_detected = sum(len(kps) for kps in keypoints[0])
            print(f"   Threshold {threshold}: {num_detected} keypoints detected")
            
            if num_detected > 0:
                print(f"     Keypoints: {keypoints[0]}")
                break
        except Exception as e:
            print(f"   ✗ Error with threshold {threshold}: {e}")
    
    # Step 9: Test the _predict_keypoints_on_crop function
    print(f"\n9. Testing _predict_keypoints_on_crop function...")
    for threshold in [0.01, 0.05, 0.1, 0.2, 0.3]:
        try:
            keypoints = _predict_keypoints_on_crop(model, crop, abs_max_threshold=threshold)
            detected = sum(1 for kp in keypoints if kp is not None)
            print(f"   Threshold {threshold}: {detected}/{len(keypoints)} keypoints")
            
            if detected > 0:
                print(f"     Detected keypoints: {keypoints}")
                break
        except Exception as e:
            print(f"   ✗ Error with threshold {threshold}: {e}")
    
    # Step 10: Check if model was trained properly
    print(f"\n10. Model training check...")
    print(f"   Check if your model was trained with:")
    print(f"   - Sufficient epochs (current: 10 in config)")
    print(f"   - Proper dataset with keypoint annotations")
    print(f"   - Correct keypoint channel configuration")
    print(f"   - Model converged (check training logs)")

if __name__ == "__main__":
    debug_keypoint_detection()
