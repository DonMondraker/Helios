import base64
import io
import json
from datetime import datetime
from typing import Any
from io import BytesIO
import numpy as np
from PIL import Image


PROJECT_STATE_VERSION = "1.2"


def _mask_to_list(mask: Image.Image | None) -> list[list[int]] | None:
    if mask is None:
        return None
    arr = np.array(mask.convert("L")).astype(int)
    return arr.tolist()


def _list_to_mask(data: list[list[int]] | None) -> Image.Image | None:
    if data is None:
        return None
    arr = np.array(data, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _image_to_base64_png(image: Image.Image | None) -> str | None:
    if image is None:
        return None

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _base64_png_to_image(data: str | None, mode: str | None = None) -> Image.Image | None:
    if not data:
        return None

    raw = base64.b64decode(data.encode("utf-8"))
    image = Image.open(io.BytesIO(raw))

    if mode is not None:
        return image.convert(mode)

    if image.mode not in ("RGB", "RGBA", "L"):
        return image.convert("RGB")

    return image.copy()


def build_inset_asset(
        asset_id: str,
        original_filename: str,
        raw_image: Image.Image | None,
        normalized_image: Image.Image | None,
        focus_mask: Image.Image | None,
        context_mask: Image.Image | None,
        rendered_image: Image.Image | None,
        focus_bbox: tuple[int, int, int, int] | None = None,
        pipeline_version: int = 1,
) -> dict[str, Any]:
    image_size = None
    if rendered_image is not None:
        image_size = list(rendered_image.size)
    elif normalized_image is not None:
        image_size = list(normalized_image.size)
    elif raw_image is not None:
        image_size = list(raw_image.size)

    return {
        "asset_id": asset_id,
        "original_filename": original_filename,
        "pipeline_version": pipeline_version,
        "image_size": image_size,
        "focus_bbox": list(focus_bbox) if focus_bbox is not None else None,
        "raw_image": raw_image,
        "normalized_image": normalized_image,
        "focus_mask": focus_mask,
        "context_mask": context_mask,
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


def _serialize_inset_assets(inset_assets: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}

    for asset_id, asset in inset_assets.items():
        serialized[asset_id] = {
            "asset_id": asset.get("asset_id", asset_id),
            "original_filename": asset.get("original_filename"),
            "pipeline_version": asset.get("pipeline_version", 1),
            "image_size": asset.get("image_size"),
            "focus_bbox": asset.get("focus_bbox"),
            "raw_image_b64": _image_to_base64_png(asset.get("raw_image")),
            "normalized_image_b64": _image_to_base64_png(asset.get("normalized_image")),
            "focus_mask_b64": _image_to_base64_png(asset.get("focus_mask")),
            "context_mask_b64": _image_to_base64_png(asset.get("context_mask")),
            "rendered_image_b64": _image_to_base64_png(asset.get("rendered_image")),
        }

    return serialized


def _deserialize_inset_assets(inset_assets_data: dict[str, Any]) -> dict[str, Any]:
    deserialized: dict[str, Any] = {}

    for asset_id, asset in inset_assets_data.items():
        deserialized[asset_id] = {
            "asset_id": asset.get("asset_id", asset_id),
            "original_filename": asset.get("original_filename"),
            "pipeline_version": asset.get("pipeline_version", 1),
            "image_size": asset.get("image_size"),
            "focus_bbox": asset.get("focus_bbox"),
            "raw_image": _base64_png_to_image(asset.get("raw_image_b64"), mode="RGB"),
            "normalized_image": _base64_png_to_image(asset.get("normalized_image_b64"), mode="RGB"),
            "focus_mask": _base64_png_to_image(asset.get("focus_mask_b64"), mode="L"),
            "context_mask": _base64_png_to_image(asset.get("context_mask_b64"), mode="L"),
            "rendered_image": _base64_png_to_image(asset.get("rendered_image_b64"), mode="RGB"),
        }

    return deserialized


def _image_to_data_url(image) -> str | None:
    if image is None:
        return None

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_project_state(
        uploaded_filename: str | None,
        session_state: Any,
        raw_image=None,
) -> dict[str, Any]:
    react_editor_state = session_state.get("react_editor_state")

    state = {
        "project_version": PROJECT_STATE_VERSION,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "source": {
            "uploaded_filename": uploaded_filename,
            "source_image": {
                "filename": uploaded_filename,
                "data_url": _image_to_data_url(raw_image),
            },
        },
        "workflow": {
            "mask_step_completed": bool(session_state.get("mask_step_completed", False)),
        },
        "mask_settings": {
            "mask_mode": "HSV",
            "focus_tolerance": session_state.get("focus_tolerance", 50),
            "context_tolerance": session_state.get("context_tolerance", 80),
            "hue_tolerance": session_state.get("hue_tolerance", 18),
            "min_saturation": session_state.get("min_saturation", 45),
            "min_value": session_state.get("min_value", 20),
        },
        "render_settings": {
            "background_threshold": session_state.get("background_threshold", 210),
            "enhance_focus": bool(session_state.get("enhance_focus", True)),
            "enhance_contour": bool(session_state.get("enhance_contour", True)),
        },
        "masks": {
            "focus_mask": _mask_to_list(session_state.get("focus_mask")),
            "context_mask": _mask_to_list(session_state.get("context_mask")),
        },
        "annotations": {
            "react_editor_state": react_editor_state,
            "callouts": session_state.get("callouts", []),
            "approved_ai_lines": session_state.get("approved_ai_lines", []),
            "manual_lines": session_state.get("manual_lines", []),
            "insets": session_state.get("insets", []),
        },
        "inset_assets": _serialize_inset_assets(session_state.get("inset_assets", {})),
        "ai": {
            "strategy": session_state.get("ai_strategy"),
            "annotation_suggestions": session_state.get("ai_annotation_suggestions"),
            "notes": session_state.get("ai_notes", []),
            "suggestion_template_key": session_state.get("suggestion_template_key"),
            "suggestion_extra_details": session_state.get("suggestion_extra_details", ""),
            "camera_preset": session_state.get("camera_preset"),
            "operation_type": session_state.get("operation_type"),
        },
    }

    return state


def serialize_project_state(state: dict[str, Any]) -> bytes:
    return json.dumps(state, indent=2, ensure_ascii=False).encode("utf-8")


def load_project_state_from_bytes(data: bytes) -> dict[str, Any]:
    return json.loads(data.decode("utf-8"))


def apply_project_state_to_session(state: dict[str, Any], session_state: Any) -> None:
    workflow = state.get("workflow", {})
    mask_settings = state.get("mask_settings", {})
    render_settings = state.get("render_settings", {})
    masks = state.get("masks", {})
    annotations = state.get("annotations", {})
    if annotations.get("react_editor_state") is not None:
        session_state["react_editor_state"] = annotations["react_editor_state"]
    ai = state.get("ai", {})
    inset_assets_data = state.get("inset_assets", {})

    session_state["mask_step_completed"] = workflow.get("mask_step_completed", False)

    session_state["mask_mode"] = mask_settings.get("mask_mode", "HSV (recommended)")
    session_state["focus_tolerance"] = mask_settings.get("focus_tolerance", 50)
    session_state["context_tolerance"] = mask_settings.get("context_tolerance", 80)
    session_state["hue_tolerance"] = mask_settings.get("hue_tolerance", 18)
    session_state["min_saturation"] = mask_settings.get("min_saturation", 45)
    session_state["min_value"] = mask_settings.get("min_value", 20)

    session_state["background_threshold"] = render_settings.get("background_threshold", 210)
    session_state["enhance_focus"] = render_settings.get("enhance_focus", True)
    session_state["enhance_contour"] = render_settings.get("enhance_contour", True)

    session_state["focus_mask"] = _list_to_mask(masks.get("focus_mask"))
    session_state["context_mask"] = _list_to_mask(masks.get("context_mask"))

    session_state["callouts"] = annotations.get("callouts", [])
    session_state["approved_ai_lines"] = annotations.get("approved_ai_lines", [])
    session_state["manual_lines"] = annotations.get("manual_lines", [])
    session_state["insets"] = annotations.get("insets", [])

    session_state["inset_assets"] = _deserialize_inset_assets(inset_assets_data)
    session_state["selected_inset_id"] = None
    session_state["pending_inset_asset_id"] = None

    session_state["ai_strategy"] = ai.get("strategy")
    session_state["ai_annotation_suggestions"] = ai.get("annotation_suggestions")
    session_state["ai_notes"] = ai.get("notes", [])
    session_state["suggestion_template_key"] = ai.get("suggestion_template_key")
    session_state["suggestion_extra_details"] = ai.get("suggestion_extra_details", "")
    session_state["camera_preset"] = ai.get("camera_preset")
    session_state["operation_type"] = ai.get("operation_type")