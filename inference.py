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
from utils.draw import draw_person_keypoints_and_skeleton


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

def run_inference(model: KeypointDetector, image, confidence_threshold: float = 0.1) -> Image:
    model.eval()
    tensored_image = torch.from_numpy(np.array(image)).float()
    tensored_image = tensored_image / 255.0
    tensored_image = tensored_image.permute(2, 0, 1)
    tensored_image = tensored_image.unsqueeze(0)
    with torch.no_grad():
        heatmaps = model(tensored_image)

    keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps, abs_max_threshold=confidence_threshold)
    image_keypoints = keypoints[0]
    for keypoints, channel_config in zip(image_keypoints, model.keypoint_channel_configuration):
        print(f"Keypoints for {channel_config}: {keypoints}")
    image = draw_keypoints_on_image(image, image_keypoints, model.keypoint_channel_configuration)
    return image


def _predict_keypoints_on_crop(model: KeypointDetector, crop: Image, abs_max_threshold: float = 0.25) -> List[Optional[Tuple[int, int]]]:
    tensored_image = torch.from_numpy(np.array(crop)).float()
    tensored_image = tensored_image / 255.0
    # Apply ImageNet normalization to match training
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensored_image = (tensored_image - mean) / std
    tensored_image = tensored_image.permute(2, 0, 1)
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


def run_multiperson_inference(
    model: KeypointDetector,
    image: Image,
    person_detector: Optional[PersonDetector] = None,
    person_conf_threshold: float = 0.7,
    keypoint_abs_threshold: float = 0.25,
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

        results.append({
            "person_id": idx,
            "bbox": (x1c, y1c, x2c, y2c),
            "keypoints": global_kps,
            "skeleton": "coco_17",
        })

    return image, results


if __name__ == "__main__":
    wandb_checkpoint = "tlips/synthetic-lego-battery-keypoints/model-tbzd50z8:v0"
    # path to locally saved checkpoint produced by SaveBestModel/save_model
    local_checkpoint_path = "snapshots/SmartJointsBest_resnet34_adamw_0.001.pth"
    image_path = "testimages/test1.jpg"
    # image_path = "/home/tlips/Documents/synthetic-cloth-data/synthetic-cloth-data/data/datasets/LEGO-battery/01/images/0.jpg"
    image_size = (256, 256)
    image = Image.open(image_path)
    image = image.resize(image_size)
    # Load locally saved model (weights & biases)
    model = get_model_from_local_checkpoint(local_checkpoint_path)
    # image = run_inference(model, image)
    image, results = run_multiperson_inference(model, image)
    print(results)
    image.save("inference_result.png")
