#!/usr/bin/env python3
"""
Test script to demonstrate keypoint detection with bounding box drawing functionality.
This script shows how to use the enhanced inference functions to detect keypoints
and draw bounding boxes around detected persons.
"""

import os
from PIL import Image
from inference import get_model_from_local_checkpoint, run_multiperson_inference, run_inference


def test_inference_with_bboxes():
    """Test the enhanced inference functionality with bounding box drawing."""
    
    # Configuration
    local_checkpoint_path = "snapshots/SmartJointsBest_LightHRNet_adamw_0.001.pth"
    image_path = "testimages/test1.jpg"
    image_size = (256, 256)
    
    # Check if files exist
    if not os.path.exists(local_checkpoint_path):
        print(f"❌ Model checkpoint not found: {local_checkpoint_path}")
        print("Please ensure you have a trained model checkpoint in the snapshots/ directory")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        print("Please ensure you have test images in the testimages/ directory")
        return
    
    print("🔄 Loading model and image...")
    
    # Load model
    try:
        model = get_model_from_local_checkpoint(local_checkpoint_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load and resize image
    try:
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        image = image.resize(image_size)
        print(f"✅ Image loaded and resized from {original_size} to {image_size}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    print("\n🔍 Running inference with bounding box detection...")
    
    # Test 1: Multi-person inference with bounding boxes (recommended approach)
    try:
        result_image, results = run_multiperson_inference(
            model=model,
            image=image.copy(),
            person_conf_threshold=0.6,  # Lower threshold to detect more people
            keypoint_abs_threshold=0.2,  # Lower threshold for more keypoint detections
            bbox_color=(255, 255, 0),  # Yellow bounding boxes
            bbox_width=3
        )
        
        # Save result
        output_path = "inference_result_with_bboxes.png"
        result_image.save(output_path)
        
        print(f"✅ Multi-person inference completed!")
        print(f"📊 Results:")
        print(f"   - Detected {len(results)} person(s)")
        print(f"   - Output saved as: {output_path}")
        
        for i, result in enumerate(results):
            keypoint_count = sum(1 for kp in result['keypoints'] if kp is not None)
            bbox = result['bbox']
            print(f"   - Person {i}: {keypoint_count}/17 keypoints detected, bbox: {bbox}")
        
    except Exception as e:
        print(f"❌ Error during multi-person inference: {e}")
    
    # Test 2: Single-person inference with keypoint-based bounding box
    try:
        single_result_image = run_inference(
            model=model,
            image=image.copy(),
            confidence_threshold=0.2,
            draw_bbox=True  # Enable bounding box drawing
        )
        
        output_path_single = "inference_result_single_with_bbox.png"
        single_result_image.save(output_path_single)
        print(f"✅ Single-person inference with bbox completed!")
        print(f"   - Output saved as: {output_path_single}")
        
    except Exception as e:
        print(f"❌ Error during single-person inference: {e}")
    
    print("\n🎉 Testing completed!")
    print("\nℹ️  Usage Tips:")
    print("1. For multi-person images, use run_multiperson_inference() - it detects persons first, then keypoints")
    print("2. For single-person images, use run_inference(draw_bbox=True) - it draws bbox around all detected keypoints")
    print("3. Adjust person_conf_threshold (0.5-0.8) to control person detection sensitivity")
    print("4. Adjust keypoint_abs_threshold (0.1-0.4) to control keypoint detection sensitivity")
    print("5. Customize bbox_color and bbox_width for different visual styles")


if __name__ == "__main__":
    test_inference_with_bboxes()
