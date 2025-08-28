from typing import List, Optional, Tuple

from PIL import Image, ImageDraw


COCO_SKELETON: List[Tuple[int, int]] = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6), (5, 11), (6, 12),
    (11, 12),
    (5, 1), (6, 1), (1, 2), (2, 0), (1, 3), (3, 4),
]


def draw_person_keypoints_and_skeleton(
    image: Image.Image,
    keypoints: List[Optional[Tuple[int, int]]],
    color=(0, 255, 0),
) -> None:
    draw = ImageDraw.Draw(image)
    for i, j in COCO_SKELETON:
        if i < len(keypoints) and j < len(keypoints):
            if keypoints[i] is not None and keypoints[j] is not None:
                draw.line([keypoints[i], keypoints[j]], fill=color, width=2)
    for p in keypoints:
        if p is not None:
            x, y = p
            r = 3
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))


