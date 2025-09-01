#!/usr/bin/env python3
"""
Kaggle compatibility test script to verify all fixes work in the Kaggle environment.
Run this before starting training to ensure everything is properly configured.
"""

import torch
import albumentations as alb
import numpy as np
from pathlib import Path

def test_albumentations_compatibility():
    """Test that all albumentations transforms work in Kaggle environment."""
    print("🔧 Testing Albumentations Compatibility")
    print("=" * 50)
    
    # Test image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_keypoints = [[[100, 100], [150, 150]], [[200, 200]]]
    
    try:
        # Create the exact transforms used in datamodule.py
        from data.augmentations import MultiChannelKeypointsCompose
        
        transforms = MultiChannelKeypointsCompose([
            alb.Resize(256, 256),
            # Fixed transforms for Kaggle compatibility
            alb.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), rotate=(-30, 30), p=0.7),
            alb.HorizontalFlip(p=0.5),
            alb.Perspective(scale=(0.05, 0.1), p=0.3),
            alb.ElasticTransform(alpha=1, sigma=50, p=0.2),
            alb.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
            alb.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            alb.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
            alb.RandomGamma(gamma_limit=(80, 120), p=0.4),
            alb.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            alb.GaussianBlur(blur_limit=(3, 7), p=0.3),
            alb.MotionBlur(blur_limit=7, p=0.2),
            alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            alb.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
            alb.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            alb.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, 
                            min_height=8, min_width=8, fill_value=0, p=0.5),
        ])
        
        # Test the transforms
        result = transforms(image=test_image, keypoints=test_keypoints)
        
        print("   ✅ All albumentations transforms work correctly")
        print(f"   📐 Output image shape: {result['image'].shape}")
        print(f"   🎯 Output keypoints channels: {len(result['keypoints'])}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False


def test_dataset_paths():
    """Test that dataset paths are correctly configured for Kaggle."""
    print("\n📁 Testing Dataset Paths")
    print("=" * 50)
    
    import config_file as confs
    
    print(f"   📂 Base path: {confs.base_path}")
    
    # Check if paths exist in Kaggle environment
    base_path = Path(confs.base_path)
    annotations_path = base_path / "annotations"
    images_path = base_path / "images"
    
    paths_to_check = [
        (base_path, "Base dataset directory"),
        (annotations_path, "Annotations directory"),
        (images_path, "Images directory"),
        (annotations_path / "person_keypoints_train2017.json", "Train annotations"),
        (annotations_path / "person_keypoints_val2017.json", "Val annotations"),
        (images_path / "train2017", "Train images directory"),
        (images_path / "val2017", "Val images directory"),
    ]
    
    all_exist = True
    for path, description in paths_to_check:
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {description}: {path}")
        if not exists:
            all_exist = False
    
    return all_exist


def test_model_creation():
    """Test that the enhanced model can be created successfully."""
    print("\n🏗️ Testing Enhanced Model Creation")
    print("=" * 50)
    
    try:
        import config_file as confs
        from backbones.light_hrnet import LightHRNet
        from models.detector import KeypointDetector
        
        # Create enhanced backbone
        backbone = LightHRNet(
            n_channels=confs.light_hrnet_channels,
            num_stages=confs.light_hrnet_stages,
            num_branches=confs.light_hrnet_branches,
            num_blocks=confs.light_hrnet_blocks
        )
        
        # Create enhanced detector
        detector = KeypointDetector(
            heatmap_sigma=6,
            maximal_gt_keypoint_pixel_distances="2 4",
            backbone=backbone,
            minimal_keypoint_extraction_pixel_distance=2,
            learning_rate=3e-4,
            keypoint_channel_configuration=confs.joints_name,
            ap_epoch_start=1,
            ap_epoch_freq=2,
            lr_scheduler_relative_threshold=0.0,
            max_keypoints=20
        )
        
        print("   ✅ Enhanced model created successfully")
        
        # Test forward pass
        test_input = torch.randn(1, 3, 256, 256)
        detector.eval()
        
        with torch.no_grad():
            output = detector(test_input)
        
        print(f"   📐 Model output shape: {list(output.shape)}")
        
        # Test optimizer configuration
        opt_config = detector.configure_optimizers()
        print(f"   ⚙️ Optimizer: {type(opt_config['optimizer']).__name__}")
        print(f"   📈 Scheduler: {type(opt_config['lr_scheduler']['scheduler']).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False


def test_training_config():
    """Test that training configuration is properly set."""
    print("\n⚙️ Testing Training Configuration")
    print("=" * 50)
    
    import config_file as confs
    
    config_checks = [
        ("Base path", confs.base_path, "/kaggle/input/key-point-data/dataset/ms_coco"),
        ("Batch size", confs.batch_size, 16),
        ("Learning rate", confs.init_lr, 3e-4),
        ("Epochs", confs.epochs, 10),  # Updated for quick testing
        ("Light HRNet channels", confs.light_hrnet_channels, 48),
        ("Light HRNet branches", confs.light_hrnet_branches, 4),
        ("Heatmap sigma", getattr(confs, 'heatmap_sigma', 6), 6),
    ]
    
    all_correct = True
    for name, actual, expected in config_checks:
        correct = actual == expected
        status = "✅" if correct else "❌"
        print(f"   {status} {name}: {actual} (expected: {expected})")
        if not correct:
            all_correct = False
    
    return all_correct


def main():
    """Run all compatibility tests."""
    print("🧪 KAGGLE COMPATIBILITY TEST")
    print("=" * 60)
    
    tests = [
        ("Albumentations", test_albumentations_compatibility),
        ("Dataset Paths", test_dataset_paths),
        ("Model Creation", test_model_creation),
        ("Training Config", test_training_config),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed: {str(e)}")
            results[test_name] = False
    
    # Summary
    print(f"\n📋 COMPATIBILITY TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:15}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print(f"\n🎉 ALL COMPATIBILITY TESTS PASSED!")
        print(f"✅ Your system is ready for Kaggle training")
        print(f"🚀 Run: python train.py")
    else:
        print(f"\n⚠️  Some compatibility issues detected.")
        print(f"💡 Please fix the failed tests before training.")
    
    return passed == total


if __name__ == "__main__":
    main()
