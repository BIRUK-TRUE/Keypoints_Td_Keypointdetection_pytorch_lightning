#!/usr/bin/env python3
"""
Example configuration for using Light HRNet backbone
"""

# Example configuration for training with Light HRNet backbone
light_hrnet_config = {
    # Backbone configuration
    "backbone_type": "LightHRNet",
    "n_channels": 64,           # Number of output channels
    "num_stages": 2,            # Number of HRNet stages
    "num_branches": 3,          # Number of parallel branches
    "num_blocks": 2,            # Number of residual blocks per stage
    
    # Training configuration
    "learning_rate": 0.001,
    "batch_size": 8,
    "epochs": 100,
    
    # Model configuration
    "heatmap_sigma": 2,
    "minimal_keypoint_extraction_pixel_distance": 1,
    "maximal_gt_keypoint_pixel_distances": "2 4",
    "max_keypoints": 20,
    
    # Data configuration
    "img_width": 256,
    "img_height": 256,
    "stride": 8.0,
}

# Example command line usage:
# python train.py --backbone_type LightHRNet --n_channels 64 --num_stages 2 --num_branches 3 --num_blocks 2

# Example Python usage:
"""
from backbones.backbone_factory import BackboneFactory
from models.detector import KeypointDetector

# Create Light HRNet backbone
backbone = BackboneFactory.create_backbone(
    "LightHRNet",
    n_channels_in=3,
    n_channels=64,
    num_stages=2,
    num_branches=3,
    num_blocks=2
)

# Create detector with Light HRNet backbone
detector = KeypointDetector(
    backbone=backbone,
    heatmap_sigma=2,
    maximal_gt_keypoint_pixel_distances="2 4",
    minimal_keypoint_extraction_pixel_distance=1,
    learning_rate=0.001,
    keypoint_channel_configuration=[['nose'], ['left_eye', 'right_eye'], ['left_shoulder', 'right_shoulder']],
    ap_epoch_start=1,
    ap_epoch_freq=2,
    lr_scheduler_relative_threshold=0.0,
    max_keypoints=20
)
"""

if __name__ == "__main__":
    print("Light HRNet Configuration Example")
    print("=" * 40)
    for key, value in light_hrnet_config.items():
        print(f"{key}: {value}")
    
    print("\nTo use this configuration:")
    print("1. Pass the parameters via command line arguments")
    print("2. Or modify the config_file.py to include these settings")
    print("3. Or use the Python API as shown in the comments above")
