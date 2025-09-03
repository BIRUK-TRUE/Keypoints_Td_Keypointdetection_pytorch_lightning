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
                draw.line([keypoints[i], keypoints[j]], fill=color, width=4)
    for p in keypoints:
        if p is not None:
            x, y = p
            # r = 6
            r=12
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 255))


def draw_person_bounding_box(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    person_id: int,
    color=(255, 255, 0),  # Yellow by default
    width: int = 3
) -> None:
    """Draw a bounding box around a detected person with person ID label.
    
    Args:
        image: PIL Image to draw on
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        person_id: ID number of the person
        color: RGB color tuple for the bounding box
        width: Width of the bounding box lines
    """
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    
    # Draw the bounding box rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    
    # Draw person ID label
    label = f"Person {person_id}"
    # Get text size for background rectangle
    try:
        # Try to get text bbox (newer Pillow versions)
        text_bbox = draw.textbbox((0, 0), label)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions
        text_width, text_height = draw.textsize(label)
    
    # Draw background rectangle for text
    label_y = max(0, y1 - text_height - 5)
    draw.rectangle(
        [x1, label_y, x1 + text_width + 10, label_y + text_height + 5],
        fill=color,
        outline=color
    )
    
    # Draw the text
    draw.text((x1 + 5, label_y + 2), label, fill=(0, 0, 0))


