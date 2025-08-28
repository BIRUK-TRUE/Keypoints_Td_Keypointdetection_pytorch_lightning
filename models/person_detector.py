import argparse
from typing import List, Tuple

import torch
import torchvision
from torchvision.transforms.functional import to_tensor
from PIL import Image


class PersonDetector:
    """Thin wrapper around TorchVision detectors to return person bounding boxes.

    Returns boxes as (x1, y1, x2, y2) ints in image pixel coordinates.
    """

    @staticmethod
    def add_argparse_args(parent: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parent.add_argument_group("PersonDetector")
        group.add_argument("--person_detector", default="fasterrcnn", choices=["fasterrcnn", "retinanet"],
                           help="Backbone for person detection")
        group.add_argument("--person_conf_threshold", type=float, default=0.6,
                           help="Confidence threshold for person detections")
        return parent

    def __init__(self, model_name: str = "fasterrcnn", conf_threshold: float = 0.6):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        if model_name == "fasterrcnn":
            self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").eval()
        elif model_name == "retinanet":
            self._model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT").eval()
        else:
            raise ValueError(f"Unknown detector model: {model_name}")

    def detect(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        with torch.no_grad():
            pred = self._model([to_tensor(image)])[0]
        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        people: List[Tuple[int, int, int, int]] = []
        for b, l, s in zip(boxes, labels, scores):
            if int(l) == 1 and float(s) >= self.conf_threshold:
                x1, y1, x2, y2 = [int(v) for v in b]
                people.append((x1, y1, x2, y2))
        return people


