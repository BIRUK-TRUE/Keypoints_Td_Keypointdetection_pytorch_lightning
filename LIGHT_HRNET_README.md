# Light HRNet Backbone for Keypoint Detection

This project now includes **Light HRNet** as the **default backbone** for keypoint detection tasks, replacing the previous UNet default.

## What is Light HRNet?

Light HRNet is a lightweight version of the High-Resolution Network (HRNet) architecture, designed specifically for keypoint detection. It features:

- **Multi-scale feature extraction**: Maintains high-resolution representations throughout the network
- **Parallel branches**: Multiple resolution streams that are fused together
- **Residual connections**: Efficient gradient flow and feature reuse
- **Lightweight design**: Reduced computational complexity compared to full HRNet

## Architecture Overview

```
Input (3, H, W)
    ↓
Initial Conv (stride=2) → BatchNorm → ReLU
    ↓
Stage 1: Single Branch (ResNet blocks)
    ↓
Transition Layer (creates multiple branches)
    ↓
Stage 2: Multi-Branch Processing
    ↓
Feature Fusion (concatenate + 1x1 conv)
    ↓
Output (n_channels, H/2, W/2)
```

## Key Features

- **Configurable stages**: 2-3 stages for different complexity levels
- **Flexible branches**: 2-4 parallel branches for multi-scale processing
- **Adjustable channels**: Configurable number of output channels
- **Residual blocks**: Basic ResNet blocks for feature extraction

## Usage

**Note**: Light HRNet is now the default backbone in `train.py`. You can still override it using command-line arguments.

### 1. Command Line Training

```bash
python train.py --backbone_type LightHRNet \
                --n_channels 64 \
                --num_stages 2 \
                --num_branches 2 \
                --num_blocks 2
```

### 2. Python API

```python
from backbones.backbone_factory import BackboneFactory
from models.detector import KeypointDetector

# Create Light HRNet backbone
backbone = BackboneFactory.create_backbone(
    "LightHRNet",
    n_channels_in=3,
    n_channels=64,
    num_stages=2,
    num_branches=2,
    num_blocks=2
)

# Create detector
detector = KeypointDetector(
    backbone=backbone,
    # ... other parameters
)
```

### 3. Configuration File

The default Light HRNet configuration is already set in `config_file.py`:

```python
# Light HRNet backbone configuration
light_hrnet_channels = 64
light_hrnet_stages = 2
light_hrnet_branches = 2  # Reduced from 3 to 2 for better stability
light_hrnet_blocks = 2
```

You can modify these values or override them using command-line arguments:

```python
# In config_file.py
backbone_type = 'LightHRNet'
n_channels = 64
num_stages = 2
num_branches = 3
num_blocks = 2
```

## Parameters

| Parameter       | Default  | Description                         |
| --------------- | -------- | ----------------------------------- |
| `n_channels_in` | 3        | Input channels (RGB images)         |
| `n_channels`    | 32       | Output channels                     |
| `num_stages`    | 2        | Number of HRNet stages              |
| `num_branches`  | 2        | Number of parallel branches         |
| `num_blocks`    | 2        | Number of residual blocks per stage |
| `num_channels`  | [32, 64] | Channel configuration per branch    |

## Performance Characteristics

- **Memory efficient**: Lower memory footprint than full HRNet
- **Fast inference**: Optimized for real-time applications
- **Scalable**: Easy to adjust complexity based on requirements
- **Compatible**: Drop-in replacement for existing backbones

## Comparison with Other Backbones

| Backbone        | Parameters | Memory | Speed  | Accuracy  |
| --------------- | ---------- | ------ | ------ | --------- |
| UNet            | ~1M        | Low    | Fast   | Good      |
| **Light HRNet** | ~2-5M      | Medium | Medium | Very Good |
| Full HRNet      | ~20M+      | High   | Slow   | Excellent |

## Testing

To test the Light HRNet backbone:

```bash
python test_light_hrnet.py
```

This will verify:

- Backbone creation
- Forward pass functionality
- Factory integration
- Different configurations

## Integration

The Light HRNet backbone is fully integrated into the existing codebase:

- ✅ Registered in `BackboneFactory`
- ✅ Compatible with `KeypointDetector`
- ✅ Command-line argument support
- ✅ Configuration file support
- ✅ Training script compatibility

## Examples

See `examples/light_hrnet_config.py` for complete configuration examples and usage patterns.

## Troubleshooting

### Common Issues

1. **IndexError: index out of range**: This was a bug in the original implementation that has been fixed. The current version properly handles transition layers between stages.

2. **Out of Memory**: Reduce `n_channels` or `num_branches`
3. **Slow Training**: Reduce `num_stages` or `num_blocks`
4. **Low Accuracy**: Increase `n_channels` or `num_stages`

### Performance Tips

- Start with default parameters and adjust based on your dataset
- Use `num_stages=2` for most applications
- `num_branches=2` provides good balance of speed and accuracy
- Monitor memory usage during training

## Contributing

The Light HRNet backbone follows the same architecture pattern as other backbones in the project. To extend or modify:

1. Inherit from `Backbone` base class
2. Implement required methods (`forward`, `get_n_channels_out`)
3. Add argument parsing support
4. Register in `BackboneFactory`
5. Add tests

## License

Same license as the main project.
