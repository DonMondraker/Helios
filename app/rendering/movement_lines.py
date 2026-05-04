import math
import numpy as np
import cv2
from PIL import Image


LINE_COLOR = (0, 0, 0)


def pil_to_np(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def np_to_pil(image_np: np.ndarray) -> Image.Image:
    return Image.fromarray(image_np.astype(np.uint8))


def get_mask_bbox(mask: Image.Image) -> tuple[int, int, int, int] | None:
    """
    Returns bounding box of binary mask as (x_min, y_min, x_max, y_max).
    """
    mask_np = np.array(mask.convert("L")) > 0
    ys, xs = np.where(mask_np)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_movement_lines(
        image: Image.Image,
        mask: Image.Image,
        direction: str = "right",
        line_count: int = 3,
        line_length: int = 80,
        spacing: int = 14,
        thickness: int = 2,
        offset_from_part: int = 10,
) -> Image.Image:
    """
    Draw parallel movement lines near the masked component.

    Supported directions:
        - right
        - left
        - up
        - down
        - up-right
        - up-left
        - down-right
        - down-left
    """
    img = pil_to_np(image).copy()
    bbox = get_mask_bbox(mask)

    if bbox is None:
        return image

    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    direction_vectors = {
        "right": (1, 0),
        "left": (-1, 0),
        "up": (0, -1),
        "down": (0, 1),
        "up-right": (1, -1),
        "up-left": (-1, -1),
        "down-right": (1, 1),
        "down-left": (-1, 1),
    }

    if direction not in direction_vectors:
        raise ValueError(
            "direction must be one of: "
            "right, left, up, down, up-right, up-left, down-right, down-left"
        )

    dx_unit, dy_unit = direction_vectors[direction]

    # Normalize so diagonal movement has the same overall line length
    vec_length = math.sqrt(dx_unit ** 2 + dy_unit ** 2)
    dx_unit /= vec_length
    dy_unit /= vec_length

    # Main line direction
    dx = int(round(dx_unit * line_length))
    dy = int(round(dy_unit * line_length))

    # Perpendicular vector for spacing between parallel lines
    perp_x_unit = -dy_unit
    perp_y_unit = dx_unit

    perp_x = int(round(perp_x_unit * spacing))
    perp_y = int(round(perp_y_unit * spacing))

    # Start just outside the component in the chosen direction
    start_x = cx + int(round(dx_unit * offset_from_part))
    start_y = cy + int(round(dy_unit * offset_from_part))

    # Center the line set around the start point
    center_offset = (line_count - 1) / 2.0

    for i in range(line_count):
        shift = i - center_offset

        x1 = start_x + int(round(shift * perp_x))
        y1 = start_y + int(round(shift * perp_y))
        x2 = x1 + dx
        y2 = y1 + dy

        cv2.line(img, (x1, y1), (x2, y2), LINE_COLOR, thickness)

    return np_to_pil(img)