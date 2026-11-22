import hashlib
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import math
import base64
import cv2
import uuid
from uuid import uuid4
from io import BytesIO
from copy import deepcopy

from ui.uploader import upload_image
from vision.preprocessor import normalize_creoview_image
from vision.mask_handler import (
    create_color_mask,
    clean_mask,
    clean_context_mask,
    inspect_pixel,
    get_unique_colors,
)
from analysis.target_analyzer import analyze_focus_mask
from rendering.color_mapper import render_standardized_illustration
from rendering.callouts import draw_callouts
from rendering.editor_annotations import (
    draw_lines_on_image,
    get_mask_bbox,
)
from rendering.insets import draw_insets_on_image
from exporting.export_config import EXPORT_PRESETS, EXPORT_BUNDLES
from exporting.image_export import draw_focus_halos, draw_react_inset_images
from exporting.exporter import (
    export_illustrations_zip,
)
from project_state.project_model import (
    create_empty_project,
    add_illustration,
    encode_image,
    decode_image,
    update_illustration_fields,
    get_illustration,
    rename_illustration,
    duplicate_illustration,
    move_illustration,
    remove_illustration,
)
from project_state.project_repository import (
    save_project,
    list_projects,
    load_project,
)
from project_state.project_file import (
    save_project_file,
    load_project_file,
)
from project_state.persistence import build_inset_asset
from config.camera_presets import CAMERA_PRESETS, resolve_direction, resolve_direction_vector
from vision.mask_handler import (
    subtract_context_supported_regions_from_focus,
    subtract_non_focus_supported_regions,
    subtract_focus_overlap_from_context,
)
from helios_editor.helios_editor_component import helios_editor
from exporting.react_state_export import draw_react_editor_state

version = "2.0.1"

EXPORT_FINAL_SIZE = (599, 415)

INSET_SIZE_PRESETS = {
    "Small": 340,
    "Medium": 420,
    "Large": 480,
}



def _create_inset_asset_from_upload(uploaded_file, session_state) -> tuple[str, dict]:
    inset_raw = Image.open(uploaded_file).convert("RGB")
    inset_normalized = normalize_creoview_image(inset_raw)

    inset_focus_mask = create_color_mask(
        image=inset_normalized,
        mode="hsv",
        target_name="magenta",
        hue_tolerance=session_state.get("hue_tolerance", 18),
        min_saturation=session_state.get("min_saturation", 45),
        min_value=session_state.get("min_value", 20),
    )
    inset_focus_mask = clean_mask(inset_focus_mask)

    inset_context_mask = create_color_mask(
        image=inset_normalized,
        mode="hsv",
        target_name="cyan",
        hue_tolerance=session_state.get("hue_tolerance", 18),
        min_saturation=session_state.get("min_saturation", 45),
        min_value=session_state.get("min_value", 20),
    )
    inset_context_mask = clean_context_mask(
        inset_context_mask,
        open_kernel=0,
        close_kernel=2,
        min_component_area=8,
    )

    inset_focus_mask = subtract_context_supported_regions_from_focus(
        focus_mask=inset_focus_mask,
        context_mask=inset_context_mask,
        min_region_area=20,
        max_region_area_ratio=0.08,
        final_open_kernel=0,
    )

    inset_focus_mask = subtract_non_focus_supported_regions(
        image=inset_normalized,
        focus_mask=inset_focus_mask,
        hue_tolerance=st.session_state.hue_tolerance,
        min_saturation=st.session_state.min_saturation,
        min_value=st.session_state.min_value,
        min_region_area=20,
        max_region_area_ratio=0.08,
        enclosure_ratio_threshold=0.55,
    )

    inset_context_mask = subtract_focus_overlap_from_context(
        context_mask=inset_context_mask,
        focus_mask=inset_focus_mask,
        erosion_kernel=3,
    )

    inset_rendered = render_standardized_illustration(
        image=inset_normalized,
        focus_mask=inset_focus_mask,
        context_mask=inset_context_mask,
        background_threshold=session_state.get("background_threshold", 180),
        enhance_focus=session_state.get("enhance_focus", True),
        enhance_contour=session_state.get("enhance_contour", True),
    )

    asset_id = f"inset_asset_{hashlib.sha1((uploaded_file.name + str(inset_rendered.size)).encode('utf-8')).hexdigest()[:12]}"

    asset = build_inset_asset(
        asset_id=asset_id,
        original_filename=uploaded_file.name,
        raw_image=inset_raw,
        normalized_image=inset_normalized,
        focus_mask=inset_focus_mask,
        context_mask=inset_context_mask,
        rendered_image=inset_rendered,
        focus_bbox=get_mask_bbox(inset_focus_mask),
    )
    return asset_id, asset


def _duplicate_selected_inset(session_state) -> None:
    selected_inset_id = session_state.get("selected_inset_id")
    if not selected_inset_id:
        return

    insets = session_state.get("insets", [])
    selected = next((i for i in insets if i["id"] == selected_inset_id), None)
    if selected is None:
        return

    duplicated = {
        **selected,
        "id": f'{selected["id"]}_copy_{len(insets) + 1}',
        "placement": {
            **selected["placement"],
            "x": int(selected["placement"]["x"]) + 30,
            "y": int(selected["placement"]["y"]) + 30,
        },
        "leader": {
            **selected.get("leader", {}),
            "end": [
                int(selected.get("leader", {}).get("end", [0, 0])[0]) + 30,
                int(selected.get("leader", {}).get("end", [0, 0])[1]) + 30,
                ],
        },
        "label": {
            **selected.get("label", {}),
            "position": [
                int(selected.get("label", {}).get("position", [0, 0])[0]) + 30,
                int(selected.get("label", {}).get("position", [0, 0])[1]) + 30,
                ],
        },
    }

    session_state.insets.append(duplicated)
    session_state.selected_inset_id = duplicated["id"]


def _delete_selected_inset(session_state) -> None:
    selected_inset_id = session_state.get("selected_inset_id")
    if not selected_inset_id:
        return

    session_state.insets = [
        inset for inset in session_state.get("insets", [])
        if inset["id"] != selected_inset_id
    ]
    session_state.selected_inset_id = None
    session_state.pending_inset_leader_target_for = None
    session_state.pending_inset_reposition_for = None


def _apply_inset_size_preset(inset: dict, asset: dict | None, preset_name: str) -> dict:
    target_width = INSET_SIZE_PRESETS.get(preset_name, 420)

    image_size = asset.get("image_size") if asset else None
    if image_size and len(image_size) == 2 and image_size[0] > 0:
        aspect_ratio = image_size[1] / image_size[0]
    else:
        aspect_ratio = 0.75

    target_height = int(round(target_width * aspect_ratio))

    updated = {
        **inset,
        "size_preset": preset_name,
        "placement": {
            **inset["placement"],
            "width": int(target_width),
            "height": int(target_height),
        },
    }
    return updated


def _update_selected_inset_leader_target(
        session_state,
        preview_point: tuple[int, int],
        editor_size: tuple[int, int],
        full_size: tuple[int, int],
) -> bool:
    inset_id = session_state.get("pending_inset_leader_target_for")
    if not inset_id:
        return False

    editor_w, editor_h = editor_size
    full_w, full_h = full_size
    scale_x = full_w / editor_w
    scale_y = full_h / editor_h

    full_x = int(round(preview_point[0] * scale_x))
    full_y = int(round(preview_point[1] * scale_y))

    updated = False
    for idx, inset in enumerate(session_state.get("insets", [])):
        if inset["id"] == inset_id:
            session_state.insets[idx] = {
                **inset,
                "leader": {
                    **inset.get("leader", {}),
                    "enabled": True,
                    "end": [full_x, full_y],
                    "style": inset.get("leader", {}).get("style", "straight"),
                    "halo": inset.get("leader", {}).get("halo", True),
                },
                "source_target": {
                    "type": "point",
                    "point": [full_x, full_y],
                    "bbox": None,
                },
            }
            updated = True
            break

    session_state.pending_inset_leader_target_for = None
    return updated


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _update_selected_inset_position_from_preview_click(
        session_state,
        preview_point: tuple[int, int],
        editor_size: tuple[int, int],
        full_size: tuple[int, int],
) -> bool:
    inset_id = session_state.get("pending_inset_reposition_for")
    if not inset_id:
        return False

    editor_w, editor_h = editor_size
    full_w, full_h = full_size
    scale_x = full_w / editor_w
    scale_y = full_h / editor_h

    full_x = int(round(preview_point[0] * scale_x))
    full_y = int(round(preview_point[1] * scale_y))

    updated = False
    for idx, inset in enumerate(session_state.get("insets", [])):
        if inset["id"] == inset_id:
            placement = inset["placement"]
            inset_w = int(placement["width"])
            inset_h = int(placement["height"])

            # center inset on click
            new_x = full_x - inset_w // 2
            new_y = full_y - inset_h // 2

            # clamp fully inside image bounds
            new_x = _clamp(new_x, 0, max(0, full_w - inset_w))
            new_y = _clamp(new_y, 0, max(0, full_h - inset_h))

            session_state.insets[idx] = {
                **inset,
                "placement": {
                    **placement,
                    "x": int(new_x),
                    "y": int(new_y),
                },
                "label": {
                    **inset.get("label", {}),
                    "position": [int(new_x), max(0, int(new_y) - 20)],
                },
            }
            updated = True
            break

    session_state.pending_inset_reposition_for = None
    return updated


def _build_preview_repositioned_inset(
        inset: dict,
        preview_point: tuple[int, int],
        editor_size: tuple[int, int],
        full_size: tuple[int, int],
) -> dict:
    editor_w, editor_h = editor_size
    full_w, full_h = full_size
    scale_x = full_w / editor_w
    scale_y = full_h / editor_h

    full_x = int(round(preview_point[0] * scale_x))
    full_y = int(round(preview_point[1] * scale_y))

    placement = inset["placement"]
    inset_w = int(placement["width"])
    inset_h = int(placement["height"])

    new_x = full_x - inset_w // 2
    new_y = full_y - inset_h // 2

    new_x = _clamp(new_x, 0, max(0, full_w - inset_w))
    new_y = _clamp(new_y, 0, max(0, full_h - inset_h))

    return {
        **inset,
        "placement": {
            **placement,
            "x": int(new_x),
            "y": int(new_y),
        },
        "label": {
            **inset.get("label", {}),
            "position": [int(new_x), max(0, int(new_y) - 20)],
        },
    }

def encode_inset_assets(inset_assets: dict) -> dict:
    """
    Convert inset assets into JSON-compatible data.
    Preserve images as base64 PNG strings.
    """
    encoded_assets = {}

    for asset_id, asset in inset_assets.items():
        encoded_asset = {}

        for key, value in asset.items():
            if isinstance(value, Image.Image):
                encoded_asset[key] = {
                    "__type__": "pil_image",
                    "data_b64": encode_image(value),
                }
            else:
                encoded_asset[key] = deepcopy(value)

        encoded_assets[asset_id] = encoded_asset

    return encoded_assets

def decode_inset_assets(encoded_assets: dict) -> dict:
    """
    Restore inset assets from JSON-compatible data.
    Convert encoded PNG images back into PIL images.
    """
    decoded_assets = {}

    for asset_id, asset in encoded_assets.items():
        decoded_asset = {}

        for key, value in asset.items():
            if (
                    isinstance(value, dict)
                    and value.get("__type__") == "pil_image"
            ):
                decoded_asset[key] = decode_image(
                    value["data_b64"]
                )
            else:
                decoded_asset[key] = deepcopy(value)

        decoded_assets[asset_id] = decoded_asset

    return decoded_assets

def restore_illustration_state(
        session_state,
        illustration_state: dict,
) -> None:
    """
    Restore a saved illustration into Streamlit session state.
    """

    # Restore original source image
    source = illustration_state.get("source", {})

    session_state.original_image = decode_image(
        source.get("image_b64"),
        mode="RGB",
    )

    # Restore masks
    masks = illustration_state.get("masks", {})

    session_state.focus_mask = decode_image(
        masks.get("focus_mask_b64"),
        mode="L",
    )

    session_state.context_mask = decode_image(
        masks.get("context_mask_b64"),
        mode="L",
    )

    # Restore rendered images
    rendered_images = illustration_state.get("rendered_images", {})

    session_state.result_image = decode_image(
        rendered_images.get("result_image_b64"),
        mode="RGB",
    )

    session_state.final_result_image = decode_image(
        rendered_images.get("final_result_image_b64"),
        mode="RGB",
    )

    # Restore settings
    for section in ("render_settings", "mask_settings"):
        for key, value in illustration_state.get(section, {}).items():
            session_state[key] = deepcopy(value)

    # Restore workflow
    workflow = illustration_state.get("workflow", {})

    session_state.mask_step_completed = workflow.get(
        "mask_step_completed", False
    )

    # Restore annotations
    annotations = illustration_state.get("annotations", {})

    session_state.react_editor_state = deepcopy(
        annotations.get("react_editor_state")
    )

    session_state.callouts = deepcopy(
        annotations.get("callouts", [])
    )

    session_state.manual_lines = deepcopy(
        annotations.get("manual_lines", [])
    )

    # Restore insets
    session_state.insets = deepcopy(
        illustration_state.get("insets", [])
    )

    session_state.inset_assets = decode_inset_assets(
        illustration_state.get("inset_assets", {})
    )

    # Clear temporary editor actions
    session_state.selected_inset_id = None
    session_state.pending_inset_asset_id = None
    session_state.pending_inset_leader_target_for = None
    session_state.pending_inset_reposition_for = None

    session_state.export_request_id = None
    session_state.last_consumed_export_request_id = None
    session_state.export_ready = False

def save_active_illustration_state(session_state) -> bool:
    """
    Save the active illustration's working state
    into the Helios 2.0 project.
    """
    project = session_state.get("helios_project")
    illustration_id = session_state.get("active_illustration_id")

    if project is None or illustration_id is None:
        return False

    focus_mask = session_state.get("focus_mask")
    context_mask = session_state.get("context_mask")
    result_image = session_state.get("result_image")
    final_result_image = session_state.get("final_result_image")

    fields = {
        "insets": deepcopy(session_state.get("insets", [])),
        "inset_assets": encode_inset_assets(
            session_state.get("inset_assets", {})
        ),
        "rendered_images": {
            "result_image_b64": encode_image(result_image),
            "final_result_image_b64": encode_image(final_result_image),
        },
        "masks": {
            "focus_mask_b64": encode_image(focus_mask),
            "context_mask_b64": encode_image(context_mask),
        },
        "render_settings": {
            "background_threshold": session_state.get(
                "background_threshold", 180
            ),
            "enhance_focus": session_state.get(
                "enhance_focus", True
            ),
            "enhance_contour": session_state.get(
                "enhance_contour", True
            ),
            "remove_background_details": session_state.get(
                "remove_background_details", False
            ),
            "reduce_background_details": session_state.get(
                "reduce_background_details", False
            ),
            "background_reduce_strength": session_state.get(
                "background_reduce_strength", 0.45
            ),
            "background_target_gray": session_state.get(
                "background_target_gray", 140
            ),
            "background_reduce_grayscale": session_state.get(
                "background_reduce_grayscale", False
            ),
            "background_reduce_blur": session_state.get(
                "background_reduce_blur", False
            ),
            "background_reduce_desaturate": session_state.get(
                "background_reduce_desaturate", True
            ),
        },
        "mask_settings": {
            "hue_tolerance": session_state.get(
                "hue_tolerance", 18
            ),
            "min_saturation": session_state.get(
                "min_saturation", 45
            ),
            "min_value": session_state.get(
                "min_value", 20
            ),
        },
        "workflow": {
            "mask_step_completed": session_state.get(
                "mask_step_completed", False
            ),
        },
        "annotations": {
            "react_editor_state": session_state.get(
                "react_editor_state"
            ),
            "callouts": session_state.get("callouts", []),
            "manual_lines": session_state.get("manual_lines", []),
        },
    }

    return update_illustration_fields(
        project,
        illustration_id,
        fields,
    )


def persist_active_project(session_state) -> bool:
    """
    Capture the active illustration and persist the
    complete Helios project to SQLite.
    """
    project = session_state.get("helios_project")

    if project is None:
        return False

    illustration_id = session_state.get("active_illustration_id")

    if illustration_id is not None:
        if not save_active_illustration_state(session_state):
            return False

    try:
        save_project(project)
    except Exception:
        import traceback
        traceback.print_exc()
        return False

    return True


def process_dropped_images(session_state, dropped_files: list) -> None:
    """
    Validate and add images received from the React editor.
    The last dropped image becomes the active illustration.
    """
    validated_images = []

    if not isinstance(dropped_files, list) or not dropped_files:
        raise ValueError("No image files were received.")

    if len(dropped_files) > 10:
        raise ValueError("You can drop a maximum of 10 images at once.")

    for item in dropped_files:
        if not isinstance(item, dict):
            raise ValueError("Invalid image data received.")

        filename = item.get("name")
        data_url = item.get("dataUrl")

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("An image is missing its filename.")

        if not isinstance(data_url, str) or not data_url.startswith(
                "data:image/"
        ):
            raise ValueError(f"{filename} is not a valid image data URL.")

        try:
            header, encoded = data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded, validate=True)

            if len(image_bytes) > 25 * 1024 * 1024:
                raise ValueError(f"{filename} exceeds the 25 MB size limit.")

            image = Image.open(BytesIO(image_bytes))
            image.verify()

            image = Image.open(BytesIO(image_bytes)).convert("RGB")

        except Exception as exc:
            raise ValueError(f"Could not read {filename}: {exc}") from exc

        validated_images.append((filename, image, image_bytes))

    project = session_state.get("helios_project")

    if project is None:
        raise ValueError("No active Helios project is available.")

    if session_state.get("active_illustration_id") is not None:
        if not persist_active_project(session_state):
            raise RuntimeError("Could not save the current illustration.")

    session_state.library_source_active = False

    for filename, image, image_bytes in validated_images:
        illustration_state = {
            "source": {
                "filename": filename,
                "image_b64": encode_image(image),
            },
            "masks": {
                "focus_mask_b64": None,
                "context_mask_b64": None,
            },
            "annotations": {
                "react_editor_state": None,
            },
        }

        illustration_id = add_illustration(
            project,
            filename,
            illustration_state,
        )

        session_state.active_illustration_id = illustration_id
        session_state.editor_project_key = f"illustration_{illustration_id}"
        session_state.uploaded_image_bytes = image_bytes
        session_state.uploaded_image_filename = filename

        restore_illustration_state(session_state, illustration_state)

    session_state.illustration_count_debug = len(project["illustrations"])

    if not persist_active_project(session_state):
        raise RuntimeError("Could not save the dropped illustrations.")


def pil_image_to_data_url(image: Image.Image) -> str:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def react_lines_to_renderer_lines(react_state: dict | None) -> list[dict]:
    if not react_state:
        return []

    return [
        {
            "x1": int(line["x1"]),
            "y1": int(line["y1"]),
            "x2": int(line["x2"]),
            "y2": int(line["y2"]),
        }
        for line in react_state.get("lines", [])
    ]


def react_callouts_to_renderer_callouts(react_state: dict | None) -> list[dict]:
    if not react_state:
        return []

    return [
        {
            "label": callout.get("label", ""),
            "circle_x": int(callout["circleX"]),
            "circle_y": int(callout["circleY"]),
            "end_x": int(callout["anchorX"]),
            "end_y": int(callout["anchorY"]),
        }
        for callout in react_state.get("callouts", [])
    ]


def draw_react_detail_views(image: Image.Image, react_state: dict | None) -> Image.Image:
    if not react_state:
        return image

    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)

    for inset in react_state.get("detailViews", []):

        x = int(inset["x"])
        y = int(inset["y"])
        w = int(inset["width"])
        h = int(inset["height"])

        sx = int(inset["sourceX"])
        sy = int(inset["sourceY"])
        sw = int(inset["sourceWidth"])
        sh = int(inset["sourceHeight"])

        if inset.get("showLeader", True):
            anchor_x = int(inset["leaderAnchorX"])
            anchor_y = int(inset["leaderAnchorY"])
            center_x = x + w // 2
            center_y = y + h // 2

            # leader halo
            draw.line(
                [(anchor_x, anchor_y), (center_x, center_y)],
                fill=(255, 255, 255),
                width=8,
            )

            # leader line
            draw.line(
                [(anchor_x, anchor_y), (center_x, center_y)],
                fill=(0, 0, 0),
                width=3,
            )

        crop = result.crop((sx, sy, sx + sw, sy + sh))
        crop = crop.resize((w, h))

        result.paste(crop, (x, y))

        # white halo border
        draw.rectangle(
            [x, y, x + w, y + h],
            outline=(255, 255, 255),
            width=8,
        )

        # black border
        draw.rectangle(
            [x, y, x + w, y + h],
            outline=(0, 0, 0),
            width=2,
        )

    return result


def focus_mask_to_focus_objects(focus_mask, min_area: int = 100) -> list[dict]:
    if focus_mask is None:
        return []

    mask_array = np.array(focus_mask)

    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]

    binary_mask = (mask_array > 0).astype("uint8") * 255

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    focus_objects = []

    for index, contour in enumerate(contours, start=1):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        arc_len = cv2.arcLength(contour, True)
        epsilon = max(0.5, 0.001 * arc_len)
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        polygon = [
            {
                "x": int(point[0][0]),
                "y": int(point[0][1]),
            }
            for point in simplified
        ]

        if len(polygon) < 3:
            continue

        focus_objects.append(
            {
                "id": f"focus_{index}",
                "name": f"Focus Object {index}",
                "haloEnabled": False,
                "polygon": polygon,
            }
        )

    return focus_objects


def pending_inset_asset_to_react(session_state):
    asset_id = session_state.get("pending_inset_asset_id")
    if not asset_id:
        return None

    asset = session_state.get("inset_assets", {}).get(asset_id)
    if not asset:
        return None

    rendered_image = asset.get("rendered_image")
    if rendered_image is None:
        return None

    return {
        "id": asset_id,
        "name": asset.get("original_filename", "Inset Image"),
        "imageSrc": pil_image_to_data_url(rendered_image),
    }


def apply_background_detail_mode(
        image: Image.Image,
        focus_mask: Image.Image | None,
        mode: str,
        reduce_strength: float = 0.65,
        padding_px: int = 3,
        grayscale_background: bool = False,
        blur_background: bool = False,
        desaturate_background: bool = False,
        blur_radius: float = 3.5,
        target_gray: int = 140,
) -> Image.Image:
    """
    Applies post-render background control.

    mode:
    - "normal": no change
    - "remove": white out everything outside focus objects
    - "reduce": visually soften everything outside focus objects
    """

    if focus_mask is None or mode == "normal":
        return image

    base = image.convert("RGB")
    image_np = np.array(base).astype(np.float32)

    mask_np = np.array(focus_mask.convert("L"))
    focus = mask_np > 0

    if padding_px > 0:
        kernel = np.ones((padding_px, padding_px), np.uint8)
        focus = cv2.dilate(focus.astype(np.uint8), kernel, iterations=1).astype(bool)

    non_focus = ~focus

    if mode == "remove":
        output = image_np.copy()
        output[non_focus] = [255, 255, 255]
        return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))

    if mode != "reduce":
        return image

    # --- Reduce mode ---
    # --- Reduce mode ---
    background = base.copy()

    if blur_background:
        background = background.filter(ImageFilter.GaussianBlur(radius=3.5))

    if grayscale_background:
        # Strong grayscale
        background = ImageOps.grayscale(background).convert("RGB")

    elif desaturate_background:
        # Stronger desaturation
        background = ImageEnhance.Color(background).enhance(0.08)

    background_np = np.array(background).astype(np.float32)

    effective_strength = float(np.clip(reduce_strength, 0.0, 1.0))

    target_gray_value = st.session_state.get("background_target_gray", 140)
    target_gray = np.array(
        [target_gray_value, target_gray_value, target_gray_value],
        dtype=np.float32,
    )

    softened_np = (
            background_np * (1.0 - effective_strength)
            + target_gray * effective_strength
    )

    output = image_np.copy()
    output[non_focus] = softened_np[non_focus]

    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def merge_focus_halo_state(
        focus_objects: list[dict],
        react_state: dict | None,
) -> list[dict]:
    if not react_state:
        return focus_objects

    existing = {
        obj.get("id"): obj
        for obj in react_state.get("focusObjects", [])
    }

    merged = []

    for obj in focus_objects:
        previous = existing.get(obj.get("id"))

        merged.append(
            {
                **obj,
                "haloEnabled": previous.get("haloEnabled", obj.get("haloEnabled", False))
                if previous
                else obj.get("haloEnabled", False),
            }
        )

    return merged


def combine_masks(*masks: Image.Image | None) -> Image.Image | None:
    valid_masks = [m.convert("L") for m in masks if m is not None]

    if not valid_masks:
        return None

    combined = np.zeros_like(np.array(valid_masks[0]), dtype=np.uint8)

    for mask in valid_masks:
        mask_np = np.array(mask)
        combined = np.maximum(combined, mask_np)

    return Image.fromarray(combined, mode="L")



def render_project_startup():
    """Display the Helios project startup screen."""

    # st.title("Welcome to Helios")
    # st.caption("Your persistent technical illustration workspace")

    new_project_column, import_project_column = st.columns(2)

    with new_project_column:
        st.subheader("New Project")

        new_project_image = st.file_uploader(
            "Upload illustrations",
            type=["png", "jpg", "jpeg"],
            key="new_project_first_illustration",
            help="Upload an illustration to create a new Helios project.",
        )


        if new_project_image is not None:
            try:
                image_bytes = new_project_image.getvalue()
                source_image = Image.open(
                    BytesIO(image_bytes)
                ).convert("RGB")
                source_image.load()
            except Exception as exc:
                st.error(f"Could not read illustration: {exc}")
            else:
                new_project = create_empty_project()

                illustration_state = {
                    "source": {
                        "filename": new_project_image.name,
                        "image_b64": encode_image(source_image),
                    },
                    "masks": {
                        "focus_mask_b64": None,
                        "context_mask_b64": None,
                    },
                    "annotations": {
                        "react_editor_state": None,
                    },
                }

                add_illustration(
                    new_project,
                    new_project_image.name,
                    illustration_state,
                )

                try:
                    save_project(new_project)
                except Exception as exc:
                    st.error(f"Could not save new project: {exc}")
                else:
                    st.session_state.pending_loaded_project = new_project
                    st.session_state.pending_project_activation = True
                    st.session_state.workspace_open = True
                    st.rerun()



    with import_project_column:
        st.subheader("Import Helios Project")

        imported_project_file = st.file_uploader(
            "Upload .helios project file",
            type=["helios"],
            key="startup_import_helios",
            help="Import an editable Helios project into your workspace.",
        )

        if imported_project_file is not None:
            try:
                imported_project = load_project_file(
                    imported_project_file.getvalue()
                )
            except ValueError as exc:
                st.error(f"Could not import Helios project: {exc}")


            else:
                try:
                    existing_project = load_project(
                        imported_project["project_id"]
                    )
                except Exception as exc:
                    st.error(
                        f"Could not check the project database: {exc}"
                    )
                    st.session_state.pop(
                        "startup_import_candidate", None
                    )
                else:
                    st.session_state["startup_import_candidate"] = (
                        imported_project
                    )

                    if existing_project is not None:
                        st.warning(
                            "A project with this ID already exists in Helios. "
                            "Importing will create a separate copy, "
                            "leaving the existing project unchanged."
                        )
                    else:
                        st.success(
                            f"Project validated: "
                            f"{len(imported_project['illustrations'])} illustrations."
                        )


        import_candidate = st.session_state.get(
            "startup_import_candidate"
        )

        if import_candidate is not None:
            if st.button(
                    "Import Project",
                    key="confirm_startup_import",
                    use_container_width=True,
            ):
                imported_copy = deepcopy(import_candidate)
                imported_copy["project_id"] = str(uuid4())

                try:
                    save_project(imported_copy)
                except Exception as exc:
                    st.error(f"Could not save imported project: {exc}")
                else:
                    st.session_state.pop(
                        "startup_import_candidate", None
                    )
                    st.session_state.pending_loaded_project = imported_copy
                    st.session_state.pending_project_activation = True
                    st.session_state.workspace_open = True
                    st.rerun()

    st.subheader("Recent Projects")

    try:
        projects = list_projects()
    except Exception as exc:
        st.error(f"Could not load saved projects: {exc}")
        return

    if not projects:
        st.info(
            "No saved projects yet. "
            "Create a new project to get started."
        )
        return

    for project in projects:
        with st.container(border=True):
            st.markdown(f"**{project['project_name']}**")
            st.caption(
                f"Last updated: {project['updated_at']}"
            )

            if st.button(
                    "Open Project",
                    key=f"open_project_{project['project_id']}",
            ):
                try:
                    loaded_project = load_project(
                        project["project_id"]
                    )

                    if loaded_project is None:
                        st.error(
                            "This project could not be found."
                        )
                    else:
                        st.session_state.pending_loaded_project = (
                            loaded_project
                        )
                        st.session_state.pending_project_activation = True
                        st.session_state.workspace_open = True
                        st.rerun()

                except Exception as exc:
                    st.error(
                        f"Could not open project: {exc}"
                    )


def main():

    st.set_page_config(page_title="Helios", layout="wide")

    with open("app/ui/style.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

    def get_base64_image(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    defaults = {
    	"helios_project": None,
    	"active_illustration_id": None,
        "workspace_open": False,
        "focus_mask": None,
        "context_mask": None,
        "result_image": None,
        "final_result_image": None,
        "callouts": [],
        "manual_lines": [],
        "callout_canvas_version": 0,
        "mask_step_completed": False,
        "last_uploaded_filename": None,
        "uploaded_image_bytes": None,
        "uploaded_image_filename": None,
        "last_uploaded_image_hash": None,
        "last_uploaded_batch_hash": None,
        "ui_mode": "Production",
        "annotation_tool": "Line",
        "annotation_canvas_version": 0,
        "camera_preset": "TRUCK_ISO1",
        "operation_type": "Remove",
        "insets": [],
        "inset_assets": {},
        "selected_inset_id": None,
        "pending_inset_asset_id": None,
        "pending_inset_leader_target_for": None,
        "pending_inset_reposition_for": None,
        "inset_label": "",
        "hue_tolerance": 18,
        "min_saturation": 45,
        "min_value": 20,
        "background_threshold": 180,
        "enhance_focus": True,
        "enhance_contour": True,
        "sidebar_section": "Render",
        "export_request_id": None,
        "last_consumed_export_request_id": None,
        "export_ready": False,
        "batch_export_active": False,
        "batch_export_queue": [],
        "batch_export_images": [],
        "batch_export_index": 0,
        "batch_export_zip_bytes": None,
        "batch_export_original_illustration_id": None,
        "batch_export_bundle_key": None,
        "batch_export_requested_illustration_id": None,
        "lock_focus_objects": False,
        "remove_background_details": False,
        "reduce_background_details": False,
        "locked_focus_objects": None,
        "background_reduce_strength": 0.45,
        "background_reduce_grayscale": False,
        "background_reduce_blur": False,
        "background_reduce_desaturate": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


    # Restore the active project after a browser refresh.
    project_id_from_url = st.query_params.get("project")

    if (
            project_id_from_url
            and not st.session_state.workspace_open
    ):
        try:
            restored_project = load_project(project_id_from_url)
        except Exception as exc:
            st.error(
                f"Could not restore the project from the URL: {exc}"
            )
        else:
            if restored_project is None:
                st.warning(
                    "The project in this URL could not be found. "
                    "Please select a project from the main menu."
                )
            else:
                st.session_state.pending_loaded_project = restored_project
                st.session_state.pending_project_activation = True
                st.session_state.workspace_open = True


    # Helios 2.0 project initialization
    if st.session_state.helios_project is None:
        st.session_state.helios_project = create_empty_project()


    if st.session_state.pop("pending_project_activation", False):
        loaded_project = st.session_state.get("pending_loaded_project")

        if loaded_project is not None:
            illustrations = loaded_project["illustrations"]

            # Replace the project without saving the previous
            # illustration into the newly loaded project.
            st.session_state.helios_project = loaded_project
            st.session_state.pending_loaded_project = None

            # Remember the active project in the browser URL.
            st.query_params["project"] = loaded_project["project_id"]

            # Prevent an old pending switch from interfering.
            st.session_state.pending_illustration_switch = None

            # Prevent a previously uploaded image from being
            # registered as an illustration in the loaded project.
            st.session_state.uploaded_image_bytes = None
            st.session_state.uploaded_image_filename = None

            # Keep the current uploader batch marked as processed,
            # so it cannot be added again to the loaded project.
            existing_uploads = st.session_state.get(
                "startup_image_upload"
            )

            if existing_uploads:
                batch_hasher = hashlib.sha256()

                for existing_upload in existing_uploads:
                    batch_hasher.update(
                        existing_upload.name.encode("utf-8")
                    )
                    batch_hasher.update(existing_upload.getvalue())

                st.session_state.last_uploaded_batch_hash = (
                    batch_hasher.hexdigest()
                )
            else:
                st.session_state.last_uploaded_batch_hash = None

            # Clear any previously prepared downloads.
            st.session_state.helios_project_file_bytes = None
            st.session_state.batch_export_zip_bytes = None
            st.session_state.batch_export_active = False

            if illustrations:
                first_illustration = illustrations[0]
                illustration_id = first_illustration["illustration_id"]

                st.session_state.active_illustration_id = illustration_id

                restore_illustration_state(
                    st.session_state,
                    first_illustration["state"],
                )

                st.session_state.library_source_active = True
                st.session_state.library_source_filename = (
                    first_illustration["state"]
                    .get("source", {})
                    .get("filename", first_illustration["name"])
                )

                st.session_state.editor_project_key = (
                    f"illustration_{illustration_id}"
                )

            else:
                st.session_state.active_illustration_id = None
                st.session_state.library_source_active = False
                st.session_state.original_image = None
                st.session_state.focus_mask = None
                st.session_state.context_mask = None
                st.session_state.result_image = None
                st.session_state.final_result_image = None
                st.session_state.react_editor_state = None
                st.session_state.editor_project_key = (
                    f"empty_project_{loaded_project['project_id']}"
                )


    # This is where I put the pending_switch!
    pending_switch = st.session_state.pop(
        "pending_illustration_switch", None
    )

    if pending_switch is not None:
        selected_illustration = get_illustration(
            st.session_state.helios_project,
            pending_switch,
        )

        if selected_illustration is not None:
            if not persist_active_project(st.session_state):
                st.error(
                    "Could not save the current project. "
                    "Illustration switching was cancelled to protect your work."
                )
            else:
                st.session_state.active_illustration_id = pending_switch

                restore_illustration_state(
                    st.session_state,
                    selected_illustration["state"],
                )

                st.session_state.library_source_active = True
                st.session_state.library_source_filename = (
                    selected_illustration["state"]
                    .get("source", {})
                    .get("filename", selected_illustration["name"])
                )

                st.session_state.editor_project_key = (
                    f"illustration_{pending_switch}"
                )

    if not st.session_state.workspace_open:
        logo_base64 = get_base64_image(
            "app/ui/assets/helios_logo_small.png"
        )

        st.markdown(
            f"""
            <div class="app-header">
                <img src="data:image/png;base64,{logo_base64}" class="app-logo"/>
                <div class="app-version">v{version}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Show project selection until a workspace is opened.
    if not st.session_state.workspace_open:
        render_project_startup()
        st.stop()

    if st.session_state.get("library_source_active", False):
        raw_image = st.session_state.original_image
        active_filename = (
                st.session_state.get("library_source_filename")
                or "library_illustration.png"
        )
        uploaded_file = None

    elif st.session_state.uploaded_image_bytes is not None:
        raw_image = Image.open(BytesIO(st.session_state.uploaded_image_bytes)).convert("RGB")
        active_filename = st.session_state.uploaded_image_filename or "uploaded_image.png"
        uploaded_file = None

    else:
        # st.info("Upload an image or load a saved project JSON to begin.")
        return
    normalized_image = normalize_creoview_image(raw_image)
    image = normalized_image

    if (
            st.session_state.uploaded_image_bytes is not None
            and not st.session_state.get("library_source_active", False)
            and st.session_state.last_uploaded_filename != active_filename
    ):
        st.session_state.last_uploaded_filename = active_filename
        st.session_state.focus_mask = None
        st.session_state.context_mask = None
        st.session_state.result_image = None
        st.session_state.final_result_image = None
        st.session_state.callouts = []
        st.session_state.annotation_canvas_version = 0
        st.session_state.mask_step_completed = False
        st.session_state.manual_lines = []

        # React editor state
        st.session_state.react_editor_state = None
        st.session_state.insets = []
        st.session_state.inset_assets = {}
        st.session_state.selected_inset_id = None
        st.session_state.pending_inset_asset_id = None
        st.session_state.pending_inset_leader_target_for = None
        st.session_state.pending_inset_reposition_for = None

    focus_mask_exists = st.session_state.focus_mask is not None
    context_mask_exists = st.session_state.context_mask is not None
    both_masks_exist = focus_mask_exists and context_mask_exists
    full_result = st.session_state.get("result_image")

    with st.sidebar:

        # sidebar_logo_base64 = get_base64_image(
        #     "app/ui/assets/helios_logo_small.png"
        # )
        #
        # st.markdown(
        #     f"""
        #         <div style="display: flex; justify-content: center; padding: 4px 0 16px 0;">
        #             <img
        #                 src="data:image/png;base64,{sidebar_logo_base64}"
        #                 style="width: 150px; max-width: 100%; height: auto;"
        #             />
        #         </div>
        #         """,
        #     unsafe_allow_html=True,
        # )

        if st.button(
                "Main Menu",
                key="return_to_main_menu",
                use_container_width=True,
        ):
            if not persist_active_project(st.session_state):
                st.error(
                    "Could not save the current project. "
                    "Returning to the main menu was cancelled."
                )
            else:
                st.session_state.workspace_open = False
                st.session_state.pending_loaded_project = None
                st.session_state.pending_project_activation = False
                # Explicitly leave the current project.
                st.query_params.pop("project", None)
                st.rerun()

        st.divider()

        # -------------------------
        # PHASE 1: MASK CREATION
        if not both_masks_exist:

            st.subheader("Mask Extraction")

            st.slider(
                "Hue tolerance",
                5,
                40,
                key="hue_tolerance",
                on_change=persist_active_project,
                args=(st.session_state,),
            )

            st.slider(
                "Minimum saturation",
                10,
                255,
                key="min_saturation",
                on_change=persist_active_project,
                args=(st.session_state,),
            )

            st.slider(
                "Minimum brightness",
                0,
                255,
                key="min_value",
                on_change=persist_active_project,
                args=(st.session_state,),
            )

            if st.button("Generate Masks", use_container_width=True):
                focus_mask = create_color_mask(
                    image=image,
                    mode="hsv",
                    target_name="magenta",
                    hue_tolerance=st.session_state.hue_tolerance,
                    min_saturation=st.session_state.min_saturation,
                    min_value=st.session_state.min_value,
                )
                focus_mask = clean_mask(focus_mask)

                context_mask = create_color_mask(
                    image=image,
                    mode="hsv",
                    target_name="cyan",
                    hue_tolerance=st.session_state.hue_tolerance,
                    min_saturation=st.session_state.min_saturation,
                    min_value=st.session_state.min_value,
                )
                context_mask = clean_context_mask(
                    context_mask,
                    open_kernel=0,
                    close_kernel=2,
                    min_component_area=8,
                )

                focus_mask = subtract_context_supported_regions_from_focus(
                    focus_mask=focus_mask,
                    context_mask=context_mask,
                    min_region_area=20,
                    max_region_area_ratio=0.08,
                    final_open_kernel=0,
                )

                focus_mask = subtract_non_focus_supported_regions(
                    image=image,
                    focus_mask=focus_mask,
                    hue_tolerance=st.session_state.hue_tolerance,
                    min_saturation=st.session_state.min_saturation,
                    min_value=st.session_state.min_value,
                    min_region_area=20,
                    max_region_area_ratio=0.08,
                    enclosure_ratio_threshold=0.55,
                )

                context_mask = subtract_focus_overlap_from_context(
                    context_mask=context_mask,
                    focus_mask=focus_mask,
                    erosion_kernel=3,
                )

                st.session_state.focus_mask = focus_mask
                st.session_state.context_mask = context_mask
                st.session_state.result_image = None
                st.session_state.final_result_image = None
                st.session_state.callouts = []
                st.session_state.manual_lines = []
                st.session_state.annotation_canvas_version += 1

                st.session_state.sidebar_section = "Render"

                # Persist the generated masks and updated workflow state.
                if not persist_active_project(st.session_state):
                    st.error(
                        "Could not save the generated masks to SQLite. "
                        "Your changes remain in the current session."
                    )
                    st.stop()

                st.rerun()

        # -------------------------
        # PHASE 2: RENDER + TOOLS
        # -------------------------

        else:

            # This part is new
            # st.subheader("Menu")
            st.radio(
                " ",
                options=["Render", "Masks", "Export", "Settings"],
                key="sidebar_section",
                label_visibility="collapsed",
            )
            st.markdown("---")
            section = st.session_state.sidebar_section

            if section == "Render":
                # st.subheader("Render Illustration")

                with st.expander("Advanced render options", expanded=False):
                    st.slider(
                        "Background threshold",
                        min_value=150,
                        max_value=255,
                        value=st.session_state.get("background_threshold", 180),
                        key="background_threshold",
                        on_change=persist_active_project,
                        args=(st.session_state,),
                    )

                    st.checkbox(
                        "Enhance focus detail",
                        value=True,
                        key="enhance_focus",
                        on_change=persist_active_project,
                        args=(st.session_state,),
                    )

                    st.checkbox(
                        "Enhance contours",
                        value=True,
                        key="enhance_contour",
                        on_change=persist_active_project,
                        args=(st.session_state,),
                    )

                    remove_background = st.checkbox(
                        "Remove background",
                        key="remove_background_details",
                        help="Keep only the focus objects and remove everything else from the rendered image.",
                        on_change=persist_active_project,
                        args=(st.session_state,),
                    )

                    reduce_background = st.checkbox(
                        "Reduce background",
                        key="reduce_background_details",
                        disabled=remove_background,
                        help="Soften non-focus details while keeping visual context.",
                        on_change=persist_active_project,
                        args=(st.session_state,),
                    )

                    if reduce_background and not remove_background:
                        st.slider(
                            "Background reduction intensity",
                            min_value=0.0,
                            max_value=1.0,
                            value=st.session_state.get("background_reduce_strength", 0.65),
                            step=0.05,
                            key="background_reduce_strength",
                            on_change=persist_active_project,
                            args=(st.session_state,)
                        )

                        st.slider(
                            "Background tone",
                            min_value=50,
                            max_value=180,
                            value=140,
                            key="background_target_gray",
                            on_change=persist_active_project,
                            args=(st.session_state,)
                        )

                        st.checkbox(
                            "Blur background",
                            key="background_reduce_blur",
                            help="Softly blur non-focus background details.",
                            on_change=persist_active_project,
                            args=(st.session_state,)
                        )

                render_button_label = (
                    "Render Illustration"
                    if st.session_state.result_image is None
                    else "Re-render Illustration"
                )

                if st.button(render_button_label, use_container_width=True):
                    result = render_standardized_illustration(
                        image=image,
                        focus_mask=st.session_state.focus_mask,
                        context_mask=st.session_state.context_mask,
                        background_threshold=st.session_state.background_threshold,
                        enhance_focus=st.session_state.enhance_focus,
                        enhance_contour=st.session_state.enhance_contour,
                    )

                    background_mode = "normal"

                    if st.session_state.get("remove_background_details"):
                        background_mode = "remove"
                    elif (
                            st.session_state.get("reduce_background_details")
                            and not st.session_state.get("remove_background_details")
                    ):
                        background_mode = "reduce"

                    protected_mask = combine_masks(
                        st.session_state.focus_mask,
                        st.session_state.context_mask,
                    )

                    result = apply_background_detail_mode(
                        image=result,
                        focus_mask=protected_mask,
                        mode=background_mode,
                        reduce_strength=st.session_state.get("background_reduce_strength", 0.65),
                        grayscale_background=st.session_state.get("background_reduce_grayscale", False),
                        blur_background=st.session_state.get("background_reduce_blur", False),
                        desaturate_background=st.session_state.get("background_reduce_desaturate", False),
                    )

                    if st.session_state.get("lock_focus_objects") or st.session_state.get("remove_background_details"):
                        focus_objects = focus_mask_to_focus_objects(
                            st.session_state.get("focus_mask"),
                            min_area=100,
                        )

                        st.session_state.locked_focus_objects = merge_focus_halo_state(
                            focus_objects,
                            st.session_state.get("react_editor_state"),
                        )
                    else:
                        st.session_state.locked_focus_objects = None

                    st.session_state.result_image = result
                    st.session_state.final_result_image = result
                    st.session_state.manual_lines = []

                    # Persist the rendered illustration and its settings.
                    if not persist_active_project(st.session_state):
                        st.error(
                            "Could not save the rendered illustration to SQLite. "
                            "Your changes remain in the current session."
                        )
                        st.stop()

                    st.rerun()

                st.markdown("---")
                # st.subheader("Inset / Detail View")

                inset_upload = st.file_uploader(
                    "Upload inset image",
                    type=["png", "jpg", "jpeg"],
                    key="inset_upload",
                )

                process_inset_clicked = st.button(
                    "Process Inset Image",
                    use_container_width=True,
                    disabled=inset_upload is None,
                )

                if process_inset_clicked:
                    try:
                        asset_id, asset = _create_inset_asset_from_upload(
                            inset_upload,
                            st.session_state,
                        )
                        st.session_state.inset_assets[asset_id] = asset
                        st.session_state.pending_inset_asset_id = asset_id
                        st.success("Inset processed. Select Inset tool and click the image to place it.")
                    except Exception as e:
                        st.error(f"Inset processing error: {e}")

                if st.session_state.get("pending_inset_asset_id"):
                    pending_asset = st.session_state.inset_assets.get(st.session_state.pending_inset_asset_id)
                    if pending_asset and pending_asset.get("rendered_image") is not None:
                        st.caption("Pending inset preview")
                        st.image(pending_asset["rendered_image"], use_column_width=True)

                if st.session_state.insets:
                    inset_ids = [inset["id"] for inset in st.session_state.insets]

                    selected_idx = 0
                    if st.session_state.get("selected_inset_id") in inset_ids:
                        selected_idx = inset_ids.index(st.session_state.selected_inset_id)

                    selected_inset_id = st.selectbox(
                        "Select inset",
                        options=inset_ids,
                        index=selected_idx,
                        key="selected_inset_id_selector",
                    )
                    st.session_state.selected_inset_id = selected_inset_id

                    inset_action_col1, inset_action_col2 = st.columns(2)

                    with inset_action_col1:
                        if st.button("Duplicate Inset", use_container_width=True):
                            _duplicate_selected_inset(st.session_state)
                            st.rerun()

                    with inset_action_col2:
                        if st.button("Delete Inset", use_container_width=True):
                            _delete_selected_inset(st.session_state)
                            st.rerun()

            elif section == "Masks":
                st.subheader("Masks")

                if st.button("Regenerate Masks", use_container_width=True):
                    st.session_state.focus_mask = None
                    st.session_state.context_mask = None
                    st.session_state.result_image = None
                    st.session_state.final_result_image = None
                    st.session_state.callouts = []
                    st.session_state.manual_lines = []
                    st.session_state.annotation_canvas_version += 1

                    # Persist the cleared masks and rendering state.
                    if not persist_active_project(st.session_state):
                        st.error(
                            "Could not save the cleared masks to SQLite. "
                            "Your changes remain in the current session."
                        )
                        st.stop()

                    st.rerun()

            elif section == "Export":
                st.subheader("Export")


                st.markdown("#### Native Project")

                if st.button(
                        "Prepare Helios Project",
                        use_container_width=True,
                        disabled=st.session_state.get("batch_export_active", False),
                ):
                    save_active_illustration_state(st.session_state)

                    st.session_state.helios_project_file_bytes = (
                        save_project_file(st.session_state.helios_project)
                    )

                project_file_bytes = st.session_state.get(
                    "helios_project_file_bytes"
                )

                if project_file_bytes is not None:
                    st.download_button(
                        label="Download .helios Project",
                        data=project_file_bytes,
                        file_name="helios_project.helios",
                        mime="application/zip",
                        use_container_width=True,
                    )

                st.divider()


                export_base_name = st.text_input(
                    "Base file name",
                    value=active_filename.rsplit(".", 1)[0],
                )

                export_bundle_key = st.selectbox(
                    "Export package",
                    options=list(EXPORT_BUNDLES.keys()),
                    format_func=lambda key: EXPORT_BUNDLES[key]["label"],
                )

                # Display export package content in debug mode
                bundle_presets = EXPORT_BUNDLES[export_bundle_key]["presets"]
                if st.session_state.get("ui_mode", "Production") == "Debug":
                    with st.expander("Package includes", expanded=False):
                        for preset_key in bundle_presets:
                            st.write(f"• {EXPORT_PRESETS[preset_key]['label']}")

                        st.write("• Original CAD PNG")
                        st.write("• Project JSON")

                if st.session_state.helios_project.get("illustrations"):
                    try:
                        if st.button(
                                "Render & Export Illustrations",
                                use_container_width=True,
                                disabled=st.session_state.get("batch_export_active", False),
                        ):
                            save_active_illustration_state(st.session_state)

                            illustrations = st.session_state.helios_project.get(
                                "illustrations", []
                            )

                            st.session_state.batch_export_queue = [
                                illustration["illustration_id"]
                                for illustration in illustrations
                            ]

                            st.session_state.batch_export_images = []
                            st.session_state.batch_export_index = 0
                            st.session_state.batch_export_zip_bytes = None
                            st.session_state.batch_export_requested_illustration_id = None
                            st.session_state.batch_export_current_illustration_id = None
                            st.session_state.export_request_id = None
                            st.session_state.export_ready = False

                            st.session_state.batch_export_original_illustration_id = (
                                st.session_state.active_illustration_id
                            )

                            st.session_state.batch_export_bundle_key = export_bundle_key
                            st.session_state.batch_export_active = True


                        if st.session_state.get("batch_export_active"):
                            queue = st.session_state.batch_export_queue
                            index = st.session_state.batch_export_index

                            if index < len(queue):
                                next_illustration_id = queue[index]

                                st.info(
                                    f"Batch export: preparing illustration "
                                    f"{index + 1} of {len(queue)}"
                                )

                                st.session_state.batch_export_current_illustration_id = (
                                    next_illustration_id
                                )

                                if (
                                    st.session_state.active_illustration_id
                                    != next_illustration_id
                                ):
                                    st.session_state.pending_illustration_switch = (
                                        next_illustration_id
                                    )
                                    st.rerun()

                                else:
                                    if (
                                            st.session_state.get("batch_export_requested_illustration_id")
                                            != next_illustration_id
                                    ):
                                        st.session_state.batch_export_requested_illustration_id = (
                                            next_illustration_id
                                        )
                                        st.session_state.export_request_id = str(uuid.uuid4())

                            else:
                                if st.session_state.batch_export_images:
                                    st.session_state.batch_export_zip_bytes = (
                                        export_illustrations_zip(
                                            illustrations=st.session_state.batch_export_images,
                                            bundle_key=st.session_state.batch_export_bundle_key,
                                        )
                                    )

                                st.session_state.batch_export_active = False

                                st.success(
                                    f"Batch export complete: "
                                    f"{len(st.session_state.batch_export_images)} "
                                    f"illustrations rendered."
                                )

                        batch_zip_bytes = st.session_state.get("batch_export_zip_bytes")

                        if batch_zip_bytes is not None:
                            st.download_button(
                                label="Download ZIP",
                                data=batch_zip_bytes,
                                file_name=f"{export_base_name}_{export_bundle_key}.zip",
                                mime="application/zip",
                                use_container_width=True,
                            )

                    except Exception as e:
                        st.error(f"Export error: {e}")
                else:
                    st.info("Add an illustration to enable export.")

            elif section == "Settings":
                st.subheader("Settings")

                if st.session_state.get("ui_mode", "Production") == "Debug":
                    with st.expander("Debug Tools"):
                        if st.button("Inspect Center Pixel"):
                            inspect_pixel(image)

                        if st.button("Show Unique Colors"):
                            get_unique_colors(image)

                        if st.session_state.focus_mask is not None:
                            focus_analysis = analyze_focus_mask(st.session_state.focus_mask)
                            st.markdown("### Focus Analysis")
                            st.json(focus_analysis.to_dict())

                st.radio(
                    "UI Mode",
                    options=["Production", "Debug"],
                    index=0 if st.session_state.get("ui_mode", "Production") == "Production" else 1,
                    key="ui_mode",
                )

    # Only show CAD image BEFORE rendering
    preview_width = 900

    if st.session_state.result_image is None:

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(raw_image, caption="Original CAD Image", width=preview_width)

        with col2:
            st.image(normalized_image, caption="Normalized Image (Used for Processing)", width=preview_width)

    if both_masks_exist and st.session_state.result_image is None:
        col3, col4 = st.columns([1, 1])
        with col3:
            st.image(st.session_state.focus_mask, caption="Focus Mask", width=preview_width)
        with col4:
            st.image(st.session_state.context_mask, caption="Context Mask", width=preview_width)

    if full_result is not None:
        # st.markdown("---")

        st.markdown('<div class="annotation-preview-layout">', unsafe_allow_html=True)
        workspace_col = st.container()

        with workspace_col:

            # Illustration navigation beneath the React editor
            project_illustrations = st.session_state.helios_project.get(
                "illustrations", []
            )

            if project_illustrations:

                if project_illustrations:

                    active_navigation_illustration = get_illustration(
                        st.session_state.helios_project,
                        st.session_state.active_illustration_id,
                    )

                    if active_navigation_illustration is not None:
                        st.caption(
                            f"Active illustration: {active_navigation_illustration['name']}"
                        )

                    navigation_columns = st.columns(
                        [1.4] * len(project_illustrations) + [7, 1]
                    )

                manage_navigation_col = navigation_columns[-1]

                for index, illustration in enumerate(project_illustrations):
                    illustration_id = illustration["illustration_id"]
                    illustration_name = illustration["name"]

                    is_active = (
                            illustration_id
                            == st.session_state.active_illustration_id
                    )

                    if st.session_state.get("annotation_switch_debug"):
                        st.write(
                            "Annotation switch debug:",
                            st.session_state["annotation_switch_debug"],
                        )

                    with navigation_columns[index % len(navigation_columns)]:
                        if st.button(
                                f"Img{index + 1}",
                                key=f"illustration_tile_{illustration_id}",
                                help=illustration_name,
                                disabled=is_active,
                                use_container_width=True,
                                type="primary" if is_active else "secondary",
                        ):
                            st.session_state["pending_illustration_switch"] = illustration_id
                            st.rerun()

            # Illustration management: rename the active illustration.
            active_illustration = get_illustration(
                st.session_state.helios_project,
                st.session_state.active_illustration_id,
            )

            if active_illustration is not None:

                # Compact illustration management menu.
                with manage_navigation_col:
                    with st.popover(
                            "⚙",
                            use_container_width=False,
                    ):
                        st.caption("Illustration actions")

                        if st.button(
                                "Rename Image",
                                key="rename_illustration_toggle",
                                use_container_width=True,
                        ):
                            st.session_state.rename_illustration_open = (
                                not st.session_state.get(
                                    "rename_illustration_open", False
                                )
                            )

                        if st.button(
                                "Duplicate Image",
                                key="duplicate_active_illustration",
                                use_container_width=True,
                        ):
                                # Save the latest edits before copying the illustration.
                                if not persist_active_project(st.session_state):
                                    st.error(
                                        "Could not save the current illustration. "
                                        "Duplication was cancelled."
                                    )
                                    st.stop()

                                new_illustration_id = duplicate_illustration(
                                    st.session_state.helios_project,
                                    st.session_state.active_illustration_id,
                                )

                                if new_illustration_id is None:
                                    st.error("Could not duplicate the illustration.")
                                    st.stop()

                                # Persist the newly created illustration.
                                try:
                                    save_project(st.session_state.helios_project)
                                except Exception as exc:
                                    # Roll back the in-memory duplicate if saving fails.
                                    st.session_state.helios_project["illustrations"] = [
                                        illustration
                                        for illustration in st.session_state.helios_project["illustrations"]
                                        if illustration["illustration_id"] != new_illustration_id
                                    ]
                                    st.error(
                                        f"Could not save the duplicated illustration: {exc}"
                                    )
                                    st.stop()

                                st.rerun()

                        # Illustration management: reorder the active illustration.
                        illustration_ids = [
                            item["illustration_id"]
                            for item in st.session_state.helios_project["illustrations"]
                        ]

                        active_index = illustration_ids.index(
                            st.session_state.active_illustration_id
                        )

                        move_left_col, move_right_col = st.columns(2)

                        with move_left_col:
                            move_left_clicked = st.button(
                                "← Move",
                                key="move_illustration_left",
                                disabled=active_index == 0,
                                use_container_width=True,
                            )

                        with move_right_col:
                            move_right_clicked = st.button(
                                "Move →",
                                key="move_illustration_right",
                                disabled=active_index == len(illustration_ids) - 1,
                                use_container_width=True,
                            )

                        if move_left_clicked or move_right_clicked:
                            # Preserve the latest edits before changing project order.
                            if not persist_active_project(st.session_state):
                                st.error(
                                    "Could not save the current illustration. "
                                    "Reordering was cancelled."
                                )
                                st.stop()

                            direction = -1 if move_left_clicked else 1

                            if move_illustration(
                                    st.session_state.helios_project,
                                    st.session_state.active_illustration_id,
                                    direction,
                            ):
                                try:
                                    save_project(st.session_state.helios_project)
                                except Exception as exc:
                                    # Restore the original order if SQLite saving fails.
                                    move_illustration(
                                        st.session_state.helios_project,
                                        st.session_state.active_illustration_id,
                                        -direction,
                                    )
                                    st.error(
                                        f"Could not save the new illustration order: {exc}"
                                    )
                                    st.stop()

                                st.rerun()

                        illustration_count = len(
                            st.session_state.helios_project["illustrations"]
                        )

                        if st.button(
                                "Delete Image",
                                key="delete_illustration_toggle",
                                disabled=illustration_count <= 1,
                                use_container_width=True,
                        ):
                            if st.session_state.get("delete_illustration_open", False):
                                st.session_state.delete_illustration_open = False
                                st.session_state.delete_illustration_target_id = None
                            else:
                                st.session_state.delete_illustration_open = True
                                st.session_state.delete_illustration_target_id = (
                                    st.session_state.active_illustration_id
                                )

                if st.session_state.get("delete_illustration_open", False):
                    deletion_target = get_illustration(
                        st.session_state.helios_project,
                        st.session_state.get("delete_illustration_target_id"),
                    )

                    if deletion_target is None:
                        st.session_state.delete_illustration_open = False
                        st.session_state.delete_illustration_target_id = None
                        st.rerun()

                    st.warning(
                        f"Delete '{deletion_target['name']}'? "
                        "This illustration and its saved editing state "
                        "will be permanently removed from the project."
                    )

                    confirm_col, cancel_col = st.columns(2)

                    with confirm_col:
                        confirm_delete = st.button(
                            "Yes, Delete Illustration",
                            key="confirm_delete_illustration",
                            type="primary",
                            use_container_width=True,
                        )

                    with cancel_col:
                        cancel_delete = st.button(
                            "Cancel",
                            key="cancel_delete_illustration",
                            use_container_width=True,
                        )

                    if cancel_delete:
                        st.session_state.delete_illustration_open = False
                        st.session_state.delete_illustration_target_id = None
                        st.rerun()

                    if confirm_delete:
                        project = st.session_state.helios_project
                        target_id = st.session_state.get(
                            "delete_illustration_target_id"
                        )

                        # Save current edits before modifying the project.
                        if not persist_active_project(st.session_state):
                            st.error(
                                "Could not save the current project. "
                                "Deletion was cancelled."
                            )
                            st.stop()

                        # Recheck the target and minimum illustration count.
                        if (
                                get_illustration(project, target_id) is None
                                or len(project["illustrations"]) <= 1
                        ):
                            st.error(
                                "The illustration cannot be deleted."
                            )
                            st.stop()

                        # Keep the original list for rollback if saving fails.
                        original_illustrations = list(
                            project["illustrations"]
                        )

                        if not remove_illustration(project, target_id):
                            st.error(
                                "Could not remove the illustration."
                            )
                            st.stop()

                        try:
                            save_project(project)
                        except Exception as exc:
                            project["illustrations"] = original_illustrations
                            st.error(
                                f"Could not save the deletion: {exc}"
                            )
                            st.stop()

                        # Reactivate the saved project without writing the
                        # deleted illustration's state into another one.
                        st.session_state.delete_illustration_open = False
                        st.session_state.delete_illustration_target_id = None
                        st.session_state.rename_illustration_open = False
                        st.session_state.pending_illustration_switch = None

                        st.session_state.pending_loaded_project = project
                        st.session_state.pending_project_activation = True

                        st.rerun()

                if st.session_state.get("rename_illustration_open", False):
                    with st.form("rename_illustration_form"):
                        new_name = st.text_input(
                            "Illustration name",
                            value=active_illustration["name"],
                        )

                        submitted = st.form_submit_button("Save Name")

                    if submitted:
                        new_name = new_name.strip()

                        if not new_name:
                            st.error("Illustration name cannot be empty.")
                        else:
                            previous_name = active_illustration["name"]

                            if rename_illustration(
                                    st.session_state.helios_project,
                                    st.session_state.active_illustration_id,
                                    new_name,
                            ):
                                if persist_active_project(st.session_state):
                                    st.session_state.rename_illustration_open = False
                                    st.rerun()
                                else:
                                    active_illustration["name"] = previous_name
                                    st.error(
                                        "Could not save the illustration name."
                                    )

            editor_image_src = pil_image_to_data_url(full_result.convert("RGB"))

            focus_objects_for_editor = (
                st.session_state.get("locked_focus_objects")
                if st.session_state.get("locked_focus_objects") is not None
                else focus_mask_to_focus_objects(
                    st.session_state.get("focus_mask"),
                    min_area=100,
                )
            )

            editor_project_key = st.session_state.get("editor_project_key")

            if not editor_project_key:
                editor_project_key = str(uuid.uuid4())
                st.session_state.editor_project_key = editor_project_key
                st.session_state.react_editor_state = None

            react_editor_state = helios_editor(
                image_src=editor_image_src,
                focus_objects=focus_objects_for_editor,
                initial_state=st.session_state.get("react_editor_state"),
                debug=st.session_state.get("ui_mode", "Production") == "Debug",
                pending_inset_asset=pending_inset_asset_to_react(st.session_state),
                export_request_id=st.session_state.get("export_request_id"),
                project_key=st.session_state.editor_project_key,
                key="helios_react_editor",
            )

            if (
                    isinstance(react_editor_state, dict)
                    and react_editor_state.get("event") == "image_drop"
                    and react_editor_state.get("projectKey")
                    == st.session_state.get("editor_project_key")
            ):
                    event_id = react_editor_state.get("eventId")
                    last_event_id = st.session_state.get(
                        "last_processed_image_drop_event_id"
                    )

                    if event_id and event_id != last_event_id:
                        st.session_state.last_processed_image_drop_event_id = event_id

                        try:
                            process_dropped_images(
                                st.session_state,
                                react_editor_state.get("files"),
                            )
                            st.success("Dropped images added to the project.")
                        except Exception as exc:
                            st.error(f"Could not add dropped images: {exc}")

                        st.rerun()

            if (
                    isinstance(react_editor_state, dict)
                    and react_editor_state
                    and react_editor_state.get("projectKey")
                    == st.session_state.get("editor_project_key")
            ):
                exported_data_url = react_editor_state.get("exportedImageDataUrl")
                returned_export_id = react_editor_state.get("exportRequestId")

                is_export_response = (
                        exported_data_url
                        and returned_export_id
                        and returned_export_id == st.session_state.get("export_request_id")
                        and react_editor_state.get("projectKey")
                        == st.session_state.get("editor_project_key")
                )


                previous_editor_state = st.session_state.get(
                    "react_editor_state"
                )

                temporary_editor_fields = {
                    "exportedImageDataUrl",
                    "exportRequestId",
                }

                current_editable_state = {
                    key: value
                    for key, value in react_editor_state.items()
                    if key not in temporary_editor_fields
                }

                previous_editable_state = {
                    key: value
                    for key, value in (
                        previous_editor_state
                        if isinstance(previous_editor_state, dict)
                        else {}
                    ).items()
                    if key not in temporary_editor_fields
                }

                editor_state_changed = (
                        previous_editable_state != current_editable_state
                )

                st.session_state["react_editor_state"] = current_editable_state


                if editor_state_changed:
                    if persist_active_project(st.session_state):
                        st.session_state["project_save_status"] = "saved"
                    else:
                        st.session_state["project_save_status"] = "failed"
                        st.error(
                            "Helios could not automatically save your "
                            "latest annotation changes."
                        )


                if is_export_response:
                    header, encoded = exported_data_url.split(",", 1)
                    image_bytes = base64.b64decode(encoded)

                    exported_image = Image.open(BytesIO(image_bytes)).convert("RGB")

                    exported_image = ImageOps.fit(
                        exported_image,
                        EXPORT_FINAL_SIZE,
                        method=Image.Resampling.LANCZOS,
                        centering=(0.5, 0.5),
                    )

                    st.session_state.final_result_image = exported_image
                    st.session_state.export_ready = True
                    st.session_state.last_consumed_export_request_id = returned_export_id
                    st.session_state.export_request_id = None

                    # Persist the final image produced by the React editor.
                    if not persist_active_project(st.session_state):
                        st.error(
                            "Could not save the exported illustration to SQLite. "
                            "The exported image remains available in this session."
                        )
                        st.stop()

                    if st.session_state.get("batch_export_active"):
                        st.session_state.batch_export_images.append(
                            {
                                "name": f"illustration_{st.session_state.batch_export_index + 1:02d}",
                                "image": exported_image.copy(),
                            }
                        )

                        st.session_state.batch_export_index += 1

                    st.success("Export rendered from editor.")
                    st.rerun()

if __name__ == "__main__":
    main()
