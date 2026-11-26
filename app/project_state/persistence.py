from typing import Any
from PIL import Image


def build_inset_asset(
        asset_id: str,
        original_filename: str,
        raw_image: Image.Image | None,
        focus_mask: Image.Image | None,
        rendered_image: Image.Image | None,
        focus_bbox: tuple[int, int, int, int] | None = None,
        pipeline_version: int = 1,
) -> dict[str, Any]:
    image_size = None
    if rendered_image is not None:
        image_size = list(rendered_image.size)
    elif raw_image is not None:
        image_size = list(raw_image.size)

    return {
        "asset_id": asset_id,
        "original_filename": original_filename,
        "pipeline_version": pipeline_version,
        "image_size": image_size,
        "focus_bbox": list(focus_bbox) if focus_bbox is not None else None,
        "raw_image": raw_image,
        "focus_mask": focus_mask,
        "rendered_image": rendered_image,
    }


def build_inset_instance(
        inset_id: str,
        asset_id: str,
        x: int,
        y: int,
        width: int,
        height: int,
        leader_end: tuple[int, int] | None = None,
        label_text: str = "",
) -> dict[str, Any]:
    if leader_end is None:
        leader_end = (max(0, x - 80), y + height // 2)

    return {
        "id": inset_id,
        "asset_id": asset_id,
        "visible": True,
        "locked": False,
        "z_index": 30,
        "placement": {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "rotation": 0,
        },
        "leader": {
            "enabled": True,
            "end": [int(leader_end[0]), int(leader_end[1])],
            "style": "straight",
            "halo": True,
        },
        "source_target": {
            "type": "point",
            "point": [int(leader_end[0]), int(leader_end[1])],
            "bbox": None,
        },
        "border_style": {
            "enabled": True,
            "color": "#000000",
            "width": 2,
            "halo": True,
            "background_fill": "#FFFFFF",
        },
        "label": {
            "enabled": bool(label_text),
            "text": str(label_text),
            "position": [int(x), max(0, int(y) - 20)],
        },
    }
