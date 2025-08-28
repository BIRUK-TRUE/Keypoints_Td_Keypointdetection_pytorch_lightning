import argparse
from typing import List, Optional, Tuple

import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from models.detector import KeypointDetector
from utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from utils.load_checkpoints import get_model_from_wandb_checkpoint


# COCO 17-keypoint skeleton (indices 0..16). Update if your channel order differs.
COCO_SKELETON: List[Tuple[int, int]] = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6), (5, 11), (6, 12),
    (11, 12),
    (5, 1), (6, 1), (1, 2), (2, 0), (1, 3), (3, 4),
]


def load_person_detector(model: str, conf_threshold: float):
    if model == "fasterrcnn":
        detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").eval()
    elif model == "retinanet":
        detector = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT").eval()
    else:
        raise ValueError(f"Unknown detector: {model}")
    return detector, conf_threshold


def detect_people(detector, image: Image.Image, conf_threshold: float) -> List[Tuple[int, int, int, int]]:
    with torch.no_grad():
        pred = detector([to_tensor(image)])[0]
    boxes = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    person_boxes: List[Tuple[int, int, int, int]] = []
    for b, l, s in zip(boxes, labels, scores):
        if int(l) == 1 and float(s) >= conf_threshold:
            x1, y1, x2, y2 = [int(v) for v in b]
            person_boxes.append((x1, y1, x2, y2))
    return person_boxes


def load_keypoint_model(wandb_artifact: Optional[str]) -> KeypointDetector:
    if wandb_artifact is None:
        raise ValueError("Please provide --wandb_artifact to load the keypoint model.")
    model = get_model_from_wandb_checkpoint(wandb_artifact)
    model.eval()
    return model


def run_keypoints_on_crop(model: KeypointDetector, crop: Image.Image, abs_max_threshold: float) -> List[Optional[Tuple[int, int]]]:
    tensored = to_tensor(crop).unsqueeze(0)
    with torch.no_grad():
        heatmaps = model(tensored)

    # Nested list: [batch][channel][kp_idx][(x,y)]
    keypoints_nested = get_keypoints_from_heatmap_batch_maxpool(heatmaps, abs_max_threshold=abs_max_threshold)[0]

    # Take the top-1 per channel (or None if not found)
    flat: List[Optional[Tuple[int, int]]] = []
    for channel_kps in keypoints_nested:
        if len(channel_kps) > 0:
            x, y = channel_kps[0]
            flat.append((int(x), int(y)))
        else:
            flat.append(None)
    return flat


def draw_keypoints_and_skeleton(img: Image.Image, keypoints: List[Optional[Tuple[int, int]]], color=(0, 255, 0)) -> None:
    draw = ImageDraw.Draw(img)
    for i, j in COCO_SKELETON:
        if i < len(keypoints) and j < len(keypoints):
            if keypoints[i] is not None and keypoints[j] is not None:
                draw.line([keypoints[i], keypoints[j]], fill=color, width=2)
    for p in keypoints:
        if p is not None:
            x, y = p
            r = 3
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))


def main():
    parser = argparse.ArgumentParser(description="Multi-person keypoint inference and skeleton drawing")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default="multiperson_pose_result.png", help="Output image path")
    parser.add_argument("--detector", type=str, default="fasterrcnn", choices=["fasterrcnn", "retinanet"], help="Detector backbone")
    parser.add_argument("--det_conf", type=float, default=0.6, help="Person detection confidence threshold")
    parser.add_argument("--kp_abs_thr", type=float, default=0.1, help="Keypoint absolute max threshold on heatmaps")
    parser.add_argument("--wandb_artifact", type=str, required=True, help="Weights & Biases artifact for keypoint model")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")

    detector, det_thr = load_person_detector(args.detector, args.det_conf)
    people_boxes = detect_people(detector, image, det_thr)

    if len(people_boxes) == 0:
        print("No persons detected above threshold.")
        image.save(args.output)
        print(f"Saved: {args.output}")
        return

    model = load_keypoint_model(args.wandb_artifact)

    for (x1, y1, x2, y2) in people_boxes:
        # Clamp to image bounds
        x1c = max(0, min(x1, image.width - 1))
        y1c = max(0, min(y1, image.height - 1))
        x2c = max(0, min(x2, image.width))
        y2c = max(0, min(y2, image.height))
        if x2c <= x1c or y2c <= y1c:
            continue

        person_crop = image.crop((x1c, y1c, x2c, y2c))
        crop_kps = run_keypoints_on_crop(model, person_crop, abs_max_threshold=args.kp_abs_thr)

        # Map back to global image coords
        kps_global: List[Optional[Tuple[int, int]]] = []
        for p in crop_kps:
            if p is None:
                kps_global.append(None)
            else:
                gx = p[0] + x1c
                gy = p[1] + y1c
                kps_global.append((gx, gy))

        draw_keypoints_and_skeleton(image, kps_global, color=(0, 255, 0))

    image.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()


