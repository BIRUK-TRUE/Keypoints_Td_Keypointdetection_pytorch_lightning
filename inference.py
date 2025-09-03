""" run inference on a provided image and save the result to a file """

import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple

from models.detector import KeypointDetector
from utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from utils.load_checkpoints import get_model_from_wandb_checkpoint
from utils.visualization import draw_keypoints_on_image
from models.person_detector import PersonDetector
from utils.draw import draw_person_keypoints_and_skeleton, draw_person_bounding_box


def get_model_from_local_checkpoint(checkpoint_path: str, device: Optional[str] = None) -> KeypointDetector:
    """
    Load a model saved locally via utils_file.SaveBestModel/save_model (plain torch.save dict).
    Prefers the serialized model object if present, otherwise constructs a compatible model
    and loads the state_dict.
    """
    # checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    checkpoint = torch.load(
    checkpoint_path,
    map_location=lambda storage, loc: storage,
    weights_only=False   #  this makes it work with PyTorch 2.6+
)


    if "model" in checkpoint:
        model: KeypointDetector = checkpoint["model"]
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])  # ensure latest weights
    else:
        # Fallback: reconstruct the model similarly to train.py defaults
        from backbones.unet import Unet
        import config_file as confs
        backbone = Unet()
        model = KeypointDetector(
            heatmap_sigma=2,
            maximal_gt_keypoint_pixel_distances="2 4",
            backbone=backbone,
            minimal_keypoint_extraction_pixel_distance=1,
            learning_rate=3e-3,
            keypoint_channel_configuration=confs.joints_name,
            ap_epoch_start=1,
            ap_epoch_freq=2,
            lr_scheduler_relative_threshold=0.0,
            max_keypoints=20,
        )
        model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[arg-type]

    if device is not None:
        model.to(device)

    model.eval()
    return model

def run_inference(model: KeypointDetector, image, confidence_threshold: float = 0.1, draw_bbox: bool = False) -> Image:
    """Run inference on a single image and optionally draw bounding boxes around detected keypoint groups.
    
    Args:
        model: Trained keypoint detection model
        image: Input PIL Image
        confidence_threshold: Minimum confidence for keypoint detection
        draw_bbox: Whether to draw bounding boxes around detected keypoint groups
    """
    model.eval()
    tensored_image = torch.from_numpy(np.array(image)).float()
    tensored_image = tensored_image / 255.0
    tensored_image = tensored_image.permute(2, 0, 1)
    tensored_image = tensored_image.unsqueeze(0)
    with torch.no_grad():
        heatmaps = model(tensored_image)

    keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps, abs_max_threshold=confidence_threshold)
    image_keypoints = keypoints[0]
    
    for keypoints_channel, channel_config in zip(image_keypoints, model.keypoint_channel_configuration):
        print(f"Keypoints for {channel_config}: {keypoints_channel}")
    
    # Draw keypoints on image
    image = draw_keypoints_on_image(image, image_keypoints, model.keypoint_channel_configuration)
    
    # Optionally draw bounding boxes around keypoint groups
    if draw_bbox:
        _draw_keypoint_group_bboxes(image, image_keypoints)
    
    return image


def _predict_keypoints_on_crop(model: KeypointDetector, crop: Image, abs_max_threshold: float = 0.25) -> List[Optional[Tuple[int, int]]]:
    tensored_image = torch.from_numpy(np.array(crop)).float()
    tensored_image = tensored_image / 255.0
    # Change HWC -> CHW before normalization
    tensored_image = tensored_image.permute(2, 0, 1)
    # Apply ImageNet normalization to match training
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensored_image = (tensored_image - mean) / std
    # tensored_image = tensored_image.permute(2, 0, 1)
    tensored_image = tensored_image.unsqueeze(0)
    with torch.no_grad():
        heatmaps = model(tensored_image)
    # [batch][channel][kps][(x,y)]
    nested = get_keypoints_from_heatmap_batch_maxpool(heatmaps, abs_max_threshold=abs_max_threshold)[0]
    flat: List[Optional[Tuple[int, int]]] = []
    for channel_kps in nested:
        if len(channel_kps) > 0:
            x, y = channel_kps[0]
            flat.append((int(x), int(y)))
        else:
            flat.append(None)
    return flat


def _draw_keypoint_group_bboxes(image: Image, image_keypoints: List[List[Tuple[int, int]]]) -> None:
    """Draw bounding boxes around groups of detected keypoints.
    
    This function calculates bounding boxes for each detected person/object
    based on their keypoint locations and draws them on the image.
    """
    from PIL import ImageDraw
    
    draw = ImageDraw.Draw(image)
    
    # Group all keypoints together to find individual persons/objects
    all_keypoints = []
    for channel_keypoints in image_keypoints:
        all_keypoints.extend(channel_keypoints)
    
    if not all_keypoints:
        return
    
    # For simplicity, draw one bounding box around all detected keypoints
    # In a more sophisticated approach, you could cluster keypoints by proximity
    x_coords = [kp[0] for kp in all_keypoints]
    y_coords = [kp[1] for kp in all_keypoints]
    
    if x_coords and y_coords:
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        
        # Add some padding around the keypoints
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.size[0], x2 + padding)
        y2 = min(image.size[1], y2 + padding)
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 255), width=3)  # Cyan color
        
        # Draw label
        label = "Detected Person"
        try:
            text_bbox = draw.textbbox((0, 0), label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(label)
        
        label_y = max(0, y1 - text_height - 5)
        draw.rectangle(
            [x1, label_y, x1 + text_width + 10, label_y + text_height + 5],
            fill=(0, 255, 255),
            outline=(0, 255, 255)
        )
        draw.text((x1 + 5, label_y + 2), label, fill=(0, 0, 0))


def run_multiperson_inference(
    model: KeypointDetector,
    image: Image,
    person_detector: Optional[PersonDetector] = None,
    person_conf_threshold: float = 0.7,
    keypoint_abs_threshold: float = 0.25,
    bbox_color: Tuple[int, int, int] = (255, 255, 0),
    bbox_width: int = 3,
) -> Tuple[Image, List[Dict[str, Any]]]:
    """
    Returns a drawn image and structured results list with entries:
        { person_id, bbox: (x1,y1,x2,y2), keypoints: [(x,y)|None]*C, skeleton: List[(i,j)] }
    """
    if person_detector is None:
        person_detector = PersonDetector("fasterrcnn", person_conf_threshold)

    boxes = person_detector.detect(image)
    results: List[Dict[str, Any]] = []

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        x1c = max(0, min(x1, image.size[0] - 1))
        y1c = max(0, min(y1, image.size[1] - 1))
        x2c = max(0, min(x2, image.size[0]))
        y2c = max(0, min(y2, image.size[1]))
        if x2c <= x1c or y2c <= y1c:
            continue
        # expand bbox slightly to avoid cutting off extremities
        expand_ratio = 0.05
        bw = x2c - x1c
        bh = y2c - y1c
        ex1 = max(0, int(x1c - expand_ratio * bw))
        ey1 = max(0, int(y1c - expand_ratio * bh))
        ex2 = min(image.size[0], int(x2c + expand_ratio * bw))
        ey2 = min(image.size[1], int(y2c + expand_ratio * bh))
        crop = image.crop((ex1, ey1, ex2, ey2))
        crop_kps = _predict_keypoints_on_crop(model, crop, abs_max_threshold=keypoint_abs_threshold)
        global_kps: List[Optional[Tuple[int, int]]] = []
        for p in crop_kps:
            if p is None:
                global_kps.append(None)
            else:
                global_kps.append((p[0] + ex1, p[1] + ey1))

        draw_person_keypoints_and_skeleton(image, global_kps)
        
        # Draw bounding box around the detected person
        draw_person_bounding_box(image, (x1c, y1c, x2c, y2c), idx, color=bbox_color, width=bbox_width)

        results.append({
            "person_id": idx,
            "bbox": (x1c, y1c, x2c, y2c),
            "keypoints": global_kps,
            "skeleton": "coco_17",
        })

    return image, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run keypoint detection inference on an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file (.pth).")
    parser.add_argument("--output", type=str, default="inference_result.png", help="Path to save the output image.")
    parser.add_argument("--mode", type=str, choices=["single", "multi"], default="multi", help="Inference mode: 'single' for single-person, 'multi' for multi-person detection.")
    parser.add_argument("--img-size", type=int, default=256, help="Size to which the image will be resized (width and height).")
    parser.add_argument("--person-threshold", type=float, default=0.7, help="Confidence threshold for person detection in multi-person mode.")
    parser.add_argument("--keypoint-threshold", type=float, default=0.2, help="Confidence threshold for keypoint detection.")
# 
    args = parser.parse_args()

    image_size = (args.img_size, args.img_size)
    image = Image.open(args.image).convert("RGB")
    
    # Load the model from the local checkpoint
    print(f"Loading model from {args.checkpoint}...")
    model = get_model_from_local_checkpoint(args.checkpoint)
    print("Model loaded successfully.")

    if args.mode == 'single':
        print("Running in single-person inference mode...")
        image = image.resize(image_size)
        image = run_inference(model, image, confidence_threshold=args.keypoint_threshold, draw_bbox=True)
        print(f"Detected keypoints saved to {args.output}")

    else: # multi-person mode
        print("Running in multi-person inference mode...")
        # Note: multi-person inference does not resize the whole image, but crops detections.
        image, results = run_multiperson_inference(
            model,
            image,
            person_conf_threshold=args.person_threshold,
            keypoint_abs_threshold=args.keypoint_threshold,
            bbox_color=(255, 255, 0),  # Yellow
            bbox_width=5
        )
        print("\nDetection Results:")
        for result in results:
            print(f"  Person {result['person_id']}: bbox={result['bbox']}, keypoints_detected={sum(1 for kp in result['keypoints'] if kp is not None)}")
        print(f"\nDetected {len(results)} person(s) in the image.")

    image.save(args.output)
    print(f"Result saved to '{args.output}'")
