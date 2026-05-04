from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image

from rendering.callouts import draw_callouts


def pil_to_np(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def np_to_pil(image_np: np.ndarray) -> Image.Image:
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)


def extract_canvas_lines(canvas_json: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    """
    Extract line objects from streamlit-drawable-canvas JSON output.

    Expected Fabric.js-style fields for line objects:
    - type == "line"
    - left, top
    - x1, y1, x2, y2
    - strokeWidth
    """
    if not canvas_json or "objects" not in canvas_json:
        return []

    lines: List[Dict[str, Any]] = []

    for obj in canvas_json.get("objects", []):
        if obj.get("type") != "line":
            continue

        left = float(obj.get("left", 0))
        top = float(obj.get("top", 0))
        x1 = float(obj.get("x1", 0))
        y1 = float(obj.get("y1", 0))
        x2 = float(obj.get("x2", 0))
        y2 = float(obj.get("y2", 0))
        stroke_width = int(round(float(obj.get("strokeWidth", 2))))

        # Fabric line coordinates are relative to the object's left/top
        abs_x1 = left + x1
        abs_y1 = top + y1
        abs_x2 = left + x2
        abs_y2 = top + y2

        lines.append(
            {
                "x1": abs_x1,
                "y1": abs_y1,
                "x2": abs_x2,
                "y2": abs_y2,
                "stroke_width": stroke_width,
            }
        )

    return lines


def scale_lines(
        lines: List[Dict[str, Any]],
        scale_x: float,
        scale_y: float,
) -> List[Dict[str, Any]]:
    scaled: List[Dict[str, Any]] = []

    avg_scale = (scale_x + scale_y) / 2.0

    for line in lines:
        scaled.append(
            {
                "x1": int(round(line["x1"] * scale_x)),
                "y1": int(round(line["y1"] * scale_y)),
                "x2": int(round(line["x2"] * scale_x)),
                "y2": int(round(line["y2"] * scale_y)),
                "stroke_width": max(1, int(round(line["stroke_width"] * avg_scale))),
            }
        )

    return scaled


def draw_lines_on_image(
        image: Image.Image,
        lines: list[dict],
        color: tuple[int, int, int] = (0, 0, 0),
        halo: bool = False,
        halo_color: tuple[int, int, int] = (255, 255, 255),
        halo_extra_thickness: int = 3,
) -> Image.Image:
    img = pil_to_np(image).copy()

    for line in lines:
        x1 = int(line["x1"])
        y1 = int(line["y1"])
        x2 = int(line["x2"])
        y2 = int(line["y2"])
        stroke_width = int(line.get("stroke_width", 2))

        if halo:
            cv2.line(
                img,
                (x1, y1),
                (x2, y2),
                halo_color,
                stroke_width + halo_extra_thickness,
                lineType=cv2.LINE_AA,
                )

        cv2.line(
            img,
            (x1, y1),
            (x2, y2),
            color,
            stroke_width,
            lineType=cv2.LINE_AA,
        )

    return np_to_pil(img)


def scale_callouts(
        callouts: List[Dict[str, Any]],
        scale_x: float,
        scale_y: float,
) -> List[Dict[str, Any]]:
    scaled: List[Dict[str, Any]] = []

    for c in callouts:
        scaled.append(
            {
                "label": c["label"],
                "circle_x": int(round(c["circle_x"] * scale_x)),
                "circle_y": int(round(c["circle_y"] * scale_y)),
                "end_x": int(round(c["end_x"] * scale_x)),
                "end_y": int(round(c["end_y"] * scale_y)),
            }
        )

    return scaled


def get_mask_bbox(mask_image: Image.Image):
    import numpy as np

    mask_np = np.array(mask_image.convert("L")) > 0
    ys, xs = np.where(mask_np)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    )


def draw_suggested_callouts(
        image: Image.Image,
        callouts: list[dict],
        radius: int = 18,
        line_thickness: int = 2,
        font_scale: float = 0.65,
        text_thickness: int = 2,
):
    return draw_callouts(
        image=image,
        callouts=[
            {
                "label": c["label"],
                "circle_x": c["circle"][0],
                "circle_y": c["circle"][1],
                "end_x": c["end"][0],
                "end_y": c["end"][1],
            }
            for c in callouts
        ],
        radius=radius,
        line_thickness=line_thickness,
        font_scale=font_scale,
        text_thickness=text_thickness,
    )


def draw_ai_suggested_lines(
        image: Image.Image,
        movement_lines: list[dict],
        color: tuple[int, int, int] = (0, 0, 0),
        stroke_width: int = 2,
        halo: bool = True,
        halo_color: tuple[int, int, int] = (255, 255, 255),
        halo_extra_thickness: int = 3,
):
    converted = []
    for line in movement_lines:
        converted.append(
            {
                "x1": line["start"][0],
                "y1": line["start"][1],
                "x2": line["end"][0],
                "y2": line["end"][1],
                "stroke_width": stroke_width,
            }
        )

    return draw_lines_on_image(
        image,
        converted,
        color=color,
        halo=halo,
        halo_color=halo_color,
        halo_extra_thickness=halo_extra_thickness,
    )


def draw_temporary_callout_points(
        image: Image.Image,
        points: list[tuple[int, int]],
        label: str = "A",
        point_radius: int = 6,
        line_thickness: int = 2,
        callout_radius: int = 18,
        font_scale: float = 0.65,
        text_thickness: int = 2,
) -> Image.Image:
    """
    Draw temporary unsaved callout placement feedback.

    Behavior:
    - 1 point: show first point marker
    - 2 points: show circle marker, leader line, and preview label
    """
    img = pil_to_np(image).copy()

    if not points:
        return image

    # Draw clicked points as visible red markers
    for x, y in points[-2:]:
        cv2.circle(img, (int(x), int(y)), point_radius, (255, 0, 0), thickness=-1)
        cv2.circle(img, (int(x), int(y)), point_radius + 1, (255, 255, 255), thickness=1)

    if len(points) >= 2:
        circle_point = points[-2]
        leader_point = points[-1]

        cx, cy = int(circle_point[0]), int(circle_point[1])
        ex, ey = int(leader_point[0]), int(leader_point[1])

        # Temporary leader line
        cv2.line(img, (cx, cy), (ex, ey), (0, 0, 0), line_thickness)

        # Temporary blocking circle
        cv2.circle(img, (cx, cy), callout_radius, (255, 255, 255), thickness=-1)
        cv2.circle(img, (cx, cy), callout_radius, (0, 0, 0), thickness=1)

        preview_label = str(label).strip().upper() or "A"

        (text_w, text_h), baseline = cv2.getTextSize(
            preview_label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )

        text_x = int(cx - text_w / 2)
        text_y = int(cy + text_h / 2) - 2

        cv2.putText(
            img,
            preview_label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            text_thickness,
            lineType=cv2.LINE_AA,
        )

    return np_to_pil(img)

def scale_point(point: list[int] | tuple[int, int], scale_x: float, scale_y: float) -> list[int]:
    return [int(round(point[0] * scale_x)), int(round(point[1] * scale_y))]

def scale_insets(
        insets: list[dict],
        scale_x: float,
        scale_y: float,
) -> list[dict]:
    scaled = []
    for inset in insets:
        scaled.append(
            {
                **inset,
                "placement": {
                    "x": int(round(inset["placement"]["x"] * scale_x)),
                    "y": int(round(inset["placement"]["y"] * scale_y)),
                    "width": int(round(inset["placement"]["width"] * scale_x)),
                    "height": int(round(inset["placement"]["height"] * scale_y)),
                },
                "leader": {
                    **inset["leader"],
                    "end": scale_point(inset["leader"]["end"], scale_x, scale_y),
                },
                "label": {
                    **inset.get("label", {}),
                    "position": scale_point(inset.get("label", {}).get("position", [0, 0]), scale_x, scale_y),
                },
            }
        )
    return scaled

def build_inset_instance_from_preview_point(
        asset_id: str,
        point: tuple[int, int],
        preview_size: tuple[int, int],
        full_size: tuple[int, int],
) -> dict:
    preview_w, preview_h = preview_size
    full_w, full_h = full_size
    scale_x = full_w / preview_w
    scale_y = full_h / preview_h

    x = int(round(point[0] * scale_x))
    y = int(round(point[1] * scale_y))
    width = 520  # default Large
    height = int(round(width * 0.75))
    size_preset = "Large"

    return {
        "id": f"inset_{asset_id}_{x}_{y}",
        "asset_id": asset_id,
        "visible": True,
        "locked": False,
        "size_preset": "Large",
        "placement": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
        "leader": {
            "enabled": True,
            "end": [max(0, x - 80), y + height // 2],
            "style": "straight",
            "halo": True,
        },
        "border_style": {
            "enabled": True,
            "color": "#000000",
            "width": 2,
            "halo": True,
            "background_fill": "#FFFFFF",
        },
        "label": {
            "enabled": False,
            "text": "",
            "position": [x, max(0, y - 20)],
        },
    }