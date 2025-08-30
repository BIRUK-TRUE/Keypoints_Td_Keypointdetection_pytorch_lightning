import torch
import numpy as np
from PIL import Image
from inference import get_model_from_local_checkpoint

def test_model_basic():
    """Basic test to see if the model works at all"""
    print("=== BASIC MODEL TEST ===")
    
    # Load model
    try:
        model = get_model_from_local_checkpoint("snapshots/SmartJointsBest_resnet34_adamw_0.001.pth")
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Create a simple test image (random noise)
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_image)
    print(f"✓ Created test image: {test_image.size}")
    
    # Preprocess
    tensored_image = torch.from_numpy(np.array(test_image)).float()
    tensored_image = tensored_image / 255.0
    tensored_image = tensored_image.permute(2, 0, 1)
    tensored_image = tensored_image.unsqueeze(0)
    print(f"✓ Preprocessed image: {tensored_image.shape}")
    
    # Run inference
    try:
        model.eval()
        with torch.no_grad():
            heatmaps = model(tensored_image)
        print(f"✓ Model inference successful")
        print(f"✓ Heatmap shape: {heatmaps.shape}")
        print(f"✓ Heatmap range: [{heatmaps.min():.4f}, {heatmaps.max():.4f}]")
        
        # Check each channel
        for i in range(heatmaps.shape[1]):
            channel = heatmaps[0, i]
            print(f"  Channel {i}: min={channel.min():.4f}, max={channel.max():.4f}, mean={channel.mean():.4f}")
            
    except Exception as e:
        print(f"✗ Model inference failed: {e}")
        return
    
    print("\n✓ Basic model test completed successfully!")

if __name__ == "__main__":
    test_model_basic()
