import hashlib
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import math
import base64
import cv2
import uuid
from io import BytesIO

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
from rendering.strategy_to_annotations import strategy_to_annotations
from exporting.export_config import EXPORT_PRESETS, EXPORT_BUNDLES
from exporting.image_export import draw_focus_halos, draw_react_inset_images
from exporting.exporter import (
    export_bundle_zip,
)
from ai.suggestion_engine import request_ai_suggestions
from project_state.persistence import (
    build_project_state,
    serialize_project_state,
    load_project_state_from_bytes,
    apply_project_state_to_session,
    build_inset_asset,
)
from config.camera_presets import CAMERA_PRESETS, resolve_direction, resolve_direction_vector
from vision.mask_handler import (
    subtract_context_supported_regions_from_focus,
    subtract_non_focus_supported_regions,
    subtract_focus_overlap_from_context,
)
from helios_editor.helios_editor_component import helios_editor
from exporting.react_state_export import draw_react_editor_state

version = "2.5"

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
        open_kernel=3,
        close_kernel=0,
        min_component_area=20,
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

        # if inset.get("showLeader", True):
        #     anchor_x = int(inset["leaderAnchorX"])
        #     anchor_y = int(inset["leaderAnchorY"])
        #     center_x = x + w // 2
        #     center_y = y + h // 2
        #
        #     # leader halo
        #     draw.line(
        #         [(anchor_x, anchor_y), (center_x, center_y)],
        #         fill=(255, 255, 255),
        #         width=8,
        #     )
        #
        #     # leader line
        #     draw.line(
        #         [(anchor_x, anchor_y), (center_x, center_y)],
        #         fill=(0, 0, 0),
        #         width=3,
        #     )

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

    # Softer whitening so grayscale/desaturation remains visible
    white = np.array(
        [st.session_state.background_target_gray]*3,
        dtype=np.float32
    )

    effective_strength = reduce_strength

    if grayscale_background:
        effective_strength = min(reduce_strength, 0.45)

    elif desaturate_background:
        effective_strength = min(reduce_strength, 0.50)

    elif blur_background:
        effective_strength = min(reduce_strength, 0.55)

    softened_np = (
            background_np * (1.0 - effective_strength)
            + white * effective_strength
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


def main():

    st.set_page_config(page_title="Helios", layout="wide")

    with open("app/ui/style.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

    def get_base64_image(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    logo_base64 = get_base64_image("app/ui/assets/helios_logo_small.png")

    st.markdown(
        f"""
        <div class="app-header">
            <img src="data:image/png;base64,{logo_base64}" class="app-logo"/>
            <div class="app-version">v{version}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    defaults = {
        "focus_mask": None,
        "context_mask": None,
        "result_image": None,
        "final_result_image": None,
        "callouts": [],
        "manual_lines": [],
        "callout_canvas_version": 0,
        "mask_step_completed": False,
        "last_uploaded_filename": None,
        "ai_notes": [],
        "ai_strategy": None,
        "ai_annotation_suggestions": None,
        "approved_ai_lines": [],
        "pending_loaded_project_state": None,
        "project_state_loaded_notice": False,
        "last_loaded_project_json_hash": None,
        "auto_rebuild_after_load": False,
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
        "sidebar_section": "Project",
        "export_request_id": None,
        "last_consumed_export_request_id": None,
        "export_ready": False,
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

    uploaded_file = upload_image()
    if not uploaded_file:
        return

    raw_image = Image.open(uploaded_file).convert("RGB")
    normalized_image = normalize_creoview_image(raw_image)
    image = normalized_image

    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.focus_mask = None
        st.session_state.context_mask = None
        st.session_state.result_image = None
        st.session_state.final_result_image = None
        st.session_state.callouts = []
        st.session_state.annotation_canvas_version = 0
        st.session_state.mask_step_completed = False
        st.session_state.ai_notes = []
        st.session_state.ai_strategy = None
        st.session_state.ai_annotation_suggestions = None
        st.session_state.approved_ai_lines = []
        st.session_state.pending_loaded_project_state = None
        st.session_state.project_state_loaded_notice = False
        st.session_state.manual_lines = []

        # React editor state
        st.session_state.react_editor_state = None

        st.session_state.auto_rebuild_after_load = False
        st.session_state.insets = []
        st.session_state.inset_assets = {}
        st.session_state.selected_inset_id = None
        st.session_state.pending_inset_asset_id = None
        st.session_state.pending_inset_leader_target_for = None
        st.session_state.pending_inset_reposition_for = None

    if st.session_state.pending_loaded_project_state is not None:
        try:
            apply_project_state_to_session(
                st.session_state.pending_loaded_project_state,
                st.session_state,
            )
            st.session_state.pending_loaded_project_state = None
            st.session_state.project_state_loaded_notice = True
            st.session_state.auto_rebuild_after_load = True
        except Exception as e:
            st.session_state.pending_loaded_project_state = None
            st.error(f"Could not apply loaded project state: {e}")

    focus_mask_exists = st.session_state.focus_mask is not None
    context_mask_exists = st.session_state.context_mask is not None
    both_masks_exist = focus_mask_exists and context_mask_exists
    full_result = st.session_state.get("result_image")

    if (
            st.session_state.auto_rebuild_after_load
            and st.session_state.focus_mask is not None
            and st.session_state.context_mask is not None
    ):
        try:
            rebuilt_result = render_standardized_illustration(
                image=image,
                focus_mask=st.session_state.focus_mask,
                context_mask=st.session_state.context_mask,
                background_threshold=st.session_state.get("background_threshold", 210),
                enhance_focus=st.session_state.get("enhance_focus", True),
                enhance_contour=st.session_state.get("enhance_contour", True),
            )
            st.session_state.result_image = rebuilt_result
            st.session_state.final_result_image = rebuilt_result
            st.session_state.auto_rebuild_after_load = False
            full_result = rebuilt_result
        except Exception as e:
            st.session_state.auto_rebuild_after_load = False
            st.error(f"Could not auto-rebuild loaded project: {e}")


    with st.sidebar:

        # -------------------------
        # PHASE 1: MASK CREATION
        # -------------------------
        if not both_masks_exist:

            st.subheader("Mask Extraction")

            st.slider("Hue tolerance", 5, 40, key="hue_tolerance")
            st.slider("Minimum saturation", 10, 255, key="min_saturation")
            st.slider("Minimum brightness", 0, 255, key="min_value")

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
                    open_kernel=3,
                    close_kernel=0,
                    min_component_area=20,
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
                st.session_state.ai_notes = []
                st.session_state.ai_strategy = None
                st.session_state.ai_annotation_suggestions = None
                st.session_state.approved_ai_lines = []

                st.session_state.sidebar_section = "Render"

                st.rerun()

            project_state_upload = st.file_uploader(
                "Load project JSON",
                type=["json"],
                key="project_state_upload",
            )

            if project_state_upload is not None:
                file_bytes = project_state_upload.read()
                current_json_hash = hashlib.sha256(file_bytes).hexdigest()

                if st.session_state.last_loaded_project_json_hash != current_json_hash:
                    try:
                        loaded_state = load_project_state_from_bytes(file_bytes)
                        st.session_state.pending_loaded_project_state = loaded_state
                        st.session_state.last_loaded_project_json_hash = current_json_hash
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not load project state: {e}")
            else:
                st.session_state.last_loaded_project_json_hash = None

        # -------------------------
        # PHASE 2: RENDER + TOOLS
        # -------------------------

        else:

            # This part is new
            st.subheader("Menu")
            st.radio(
                " ",
                options=["Render", "Masks", "Export", "Settings"],
                key="sidebar_section",
                label_visibility="collapsed",
            )
            st.markdown("---")
            section = st.session_state.sidebar_section

            if section == "Render":
                st.subheader("Render Illustration")

                with st.expander("Advanced render options", expanded=False):
                    st.slider(
                        "Background threshold",
                        min_value=150,
                        max_value=255,
                        value=st.session_state.get("background_threshold", 180),
                        key="background_threshold",
                    )

                    st.checkbox("Enhance focus detail", value=True, key="enhance_focus")
                    st.checkbox("Enhance contours", value=True, key="enhance_contour")

                    # st.checkbox(
                    #     "Lock focus objects",
                    #     key="lock_focus_objects",
                    #     help="Freeze detected focus objects so re-rendering does not replace or reset them.",
                    # )

                    remove_background = st.checkbox(
                        "Remove background",
                        key="remove_background_details",
                        help="Keep only the focus objects and remove everything else from the rendered image.",
                    )

                    reduce_background = st.checkbox(
                        "Reduce background",
                        key="reduce_background_details",
                        disabled=remove_background,
                        help="Soften non-focus details while keeping visual context.",
                    )

                    if reduce_background and not remove_background:
                        st.slider(
                            "Background reduction intensity",
                            min_value=0.0,
                            max_value=1.0,
                            value=st.session_state.get("background_reduce_strength", 0.65),
                            step=0.05,
                            key="background_reduce_strength",
                        )

                        st.slider(
                            "Background tone",
                            min_value=50,
                            max_value=180,
                            value=140,
                            key="background_target_gray",
                        )

                        # st.checkbox(
                        #     "Grayscale background",
                        #     key="background_reduce_grayscale",
                        #     help="Convert non-focus background to grayscale.",
                        # )

                        st.checkbox(
                            "Blur background",
                            key="background_reduce_blur",
                            help="Softly blur non-focus background details.",
                        )

                        # st.checkbox(
                        #     "Desaturate background",
                        #     key="background_reduce_desaturate",
                        #     help="Reduce background color intensity without making it fully grayscale.",
                        # )

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

                    result = apply_background_detail_mode(
                        image=result,
                        focus_mask=st.session_state.focus_mask,
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
                    st.session_state.ai_notes = []
                    st.session_state.ai_strategy = None
                    st.session_state.ai_annotation_suggestions = None
                    st.session_state.approved_ai_lines = []
                    st.session_state.manual_lines = []
                    st.rerun()

                st.markdown("---")
                st.subheader("Inset / Detail View")

                inset_upload = st.file_uploader(
                    "Upload inset image",
                    type=["png", "jpg", "jpeg"],
                    key="inset_upload",
                )

                if inset_upload is not None:
                    if st.button("Process Inset Image", use_container_width=True):
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

                st.markdown("---")
                st.subheader("AI Suggestion Panel")

                camera_preset = st.selectbox(
                    "Camera Angle",
                    list(CAMERA_PRESETS.keys()),
                    index=list(CAMERA_PRESETS.keys()).index(
                        st.session_state.get("camera_preset", "TRUCK_ISO1")
                    ) if st.session_state.get("camera_preset", "TRUCK_ISO1") in CAMERA_PRESETS else 0,
                )
                st.session_state.camera_preset = camera_preset

                operation_type = st.selectbox(
                    "Operation",
                    ["Remove", "Install", "General"],
                    index=["Remove", "Install", "General"].index(
                        st.session_state.get("operation_type", "Remove")
                    ) if st.session_state.get("operation_type", "Remove") in ["Remove", "Install", "General"] else 0,
                )
                st.session_state.operation_type = operation_type

                if st.session_state.get("ui_mode", "Production") == "Debug":
                    st.write(f"**Resolved direction:** {resolve_direction(camera_preset, operation_type)}")

                if st.button("Generate AI Suggestions", use_container_width=True):
                    if st.session_state.result_image is None:
                        st.warning("Render an illustration first.")
                    else:
                        try:
                            focus_bbox = get_mask_bbox(st.session_state.focus_mask)
                            context_bbox = get_mask_bbox(st.session_state.context_mask)

                            focus_analysis = None
                            if st.session_state.focus_mask is not None:
                                focus_analysis = analyze_focus_mask(st.session_state.focus_mask).to_dict()

                            ai_strategy = request_ai_suggestions(
                                template_instruction="Suggest movement-line strategy for the highlighted focus parts. Do not suggest callouts.",
                                image_width=st.session_state.result_image.size[0],
                                image_height=st.session_state.result_image.size[1],
                                focus_bbox=focus_bbox,
                                context_bbox=context_bbox,
                                focus_analysis=focus_analysis,
                                extra_details="",
                                model="gpt-5.4-mini",
                            )

                            resolved_direction = resolve_direction(camera_preset, operation_type)
                            resolved_direction_vector = resolve_direction_vector(camera_preset, operation_type)

                            extra_snap_angles = []

                            if resolved_direction_vector is not None:
                                dx, dy = resolved_direction_vector
                                extra_snap_angles.append(math.degrees(math.atan2(dy, dx)))

                            ai_strategy["suggested_direction"] = resolved_direction

                            if resolved_direction_vector is not None:
                                ai_strategy["suggested_direction_vector"] = [
                                    resolved_direction_vector[0],
                                    resolved_direction_vector[1],
                                ]
                            else:
                                ai_strategy.pop("suggested_direction_vector", None)

                            deterministic_suggestions = strategy_to_annotations(
                                focus_mask=st.session_state.focus_mask,
                                ai_strategy=ai_strategy,
                                image_size=st.session_state.result_image.size,
                            )

                            deterministic_suggestions["callouts"] = []

                            st.session_state.ai_strategy = ai_strategy
                            st.session_state.ai_annotation_suggestions = deterministic_suggestions
                            st.session_state.ai_notes = deterministic_suggestions.get("notes", [])
                        except Exception as e:
                            st.error(f"AI suggestion error: {e}")

                ai_strategy = st.session_state.get("ai_strategy")
                ai_annotation_suggestions = st.session_state.get("ai_annotation_suggestions")

                if ai_strategy is not None:
                    if st.session_state.get("ui_mode", "Production") == "Debug":
                        st.markdown("### AI Strategy")
                        st.write(f'**Task type:** {ai_strategy.get("task_type", "unknown")}')
                        st.write(f'**Target mode:** {ai_strategy.get("target_mode", "unknown")}')
                        st.write(f'**Suggested direction:** {ai_strategy.get("suggested_direction", "none")}')
                        st.write(f'**Movement line strategy:** {ai_strategy.get("movement_line_strategy", "none")}')
                        st.write(f'**Callout strategy:** {ai_strategy.get("callout_strategy", "none")}')
                        st.write(f'**Max callouts:** {ai_strategy.get("max_callouts", 0)}')
                        st.caption("Preview style: gray = pending AI suggestion, black = approved AI line")
                    else:
                        st.caption("AI suggestion ready.")

                if ai_annotation_suggestions is not None:
                    movement_count = len(ai_annotation_suggestions.get("movement_lines", []))
                    callout_count = len(ai_annotation_suggestions.get("callouts", []))
                    st.caption(
                        f"Deterministic suggestion preview ready: {movement_count} movement line(s), {callout_count} callout(s)."
                    )

                    if st.button("Apply AI Lines", use_container_width=True):
                        for line in ai_annotation_suggestions.get("movement_lines", []):
                            st.session_state.approved_ai_lines.append(
                                {
                                    "start": [int(line["start"][0]), int(line["start"][1])],
                                    "end": [int(line["end"][0]), int(line["end"][1])],
                                }
                            )
                        st.rerun()

                    if st.session_state.approved_ai_lines:
                        st.caption(f"Approved AI lines: {len(st.session_state.approved_ai_lines)}")
                        if st.button("Clear Approved AI Lines", use_container_width=True):
                            st.session_state.approved_ai_lines = []
                            st.rerun()

                    if st.button("Clear AI Suggestions", use_container_width=True):
                        st.session_state.ai_strategy = None
                        st.session_state.ai_annotation_suggestions = None
                        st.session_state.ai_notes = []
                        st.rerun()

                    if st.session_state.ai_notes and st.session_state.get("ui_mode", "Production") == "Debug":
                        with st.expander("AI Notes", expanded=False):
                            for note in st.session_state.ai_notes:
                                st.write(f"- {note}")

            elif section == "Masks":
                st.subheader("Masks")

                if st.button("Regenerate Masks", use_container_width=True):
                    st.session_state.focus_mask = None
                    st.session_state.context_mask = None
                    st.session_state.result_image = None
                    st.session_state.final_result_image = None
                    st.session_state.callouts = []
                    st.session_state.manual_lines = []
                    st.session_state.approved_ai_lines = []
                    st.session_state.ai_strategy = None
                    st.session_state.ai_annotation_suggestions = None
                    st.session_state.ai_notes = []
                    st.session_state.annotation_canvas_version += 1
                    st.rerun()

            elif section == "Export":
                st.subheader("Export")

                export_base_name = st.text_input(
                    "Base file name",
                    value=uploaded_file.name.rsplit(".", 1)[0],
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

                final_image = st.session_state.get("final_result_image")

                if final_image is not None:
                    try:
                        project_state = build_project_state(
                            uploaded_filename=uploaded_file.name,
                            session_state=st.session_state,
                            raw_image=raw_image,
                        )
                        project_state_bytes = serialize_project_state(project_state)

                        zip_bytes = export_bundle_zip(
                            final_image=final_image,
                            raw_image=raw_image,
                            project_state_bytes=project_state_bytes,
                            base_name=export_base_name,
                            bundle_key=export_bundle_key,
                        )
                        # Render button
                        if st.button(
                                "Render export from editor",
                                use_container_width=True,
                                disabled=st.session_state.get("export_request_id") is not None
                        ):
                            st.session_state.export_request_id = str(uuid.uuid4())

                        # Download section
                        if st.session_state.get("export_ready") and st.session_state.get("final_result_image") is not None:
                            st.download_button(
                                label="Download Export Package",
                                data=zip_bytes,
                                file_name=f"{export_base_name}_{export_bundle_key}.zip",
                                mime="application/zip",
                                use_container_width=True,
                            )
                        else:
                            st.button(
                                "Download Export Package",
                                disabled=True,
                                use_container_width=True,
                            )

                    except Exception as e:
                        st.error(f"Export error: {e}")
                else:
                    st.info("Render an illustration first to enable export.")

                project_state = build_project_state(
                    uploaded_filename=uploaded_file.name if uploaded_file is not None else None,
                    session_state=st.session_state,
                    raw_image=raw_image,
                )
                project_state_bytes = serialize_project_state(project_state)

                st.download_button(
                    label="Save Project JSON",
                    data=project_state_bytes,
                    file_name="illustration_project_state.json",
                    mime="application/json",
                    use_container_width=True,
                )

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

    if st.session_state.project_state_loaded_notice:
        st.success("Project state loaded. Re-render the illustration to rebuild the result.")
        st.session_state.project_state_loaded_notice = False

    if both_masks_exist and st.session_state.result_image is None:
        col3, col4 = st.columns([1, 1])
        with col3:
            st.image(st.session_state.focus_mask, caption="Focus Mask", width=preview_width)
        with col4:
            st.image(st.session_state.context_mask, caption="Context Mask", width=preview_width)

    if full_result is not None:
        st.markdown("---")

        st.markdown('<div class="annotation-preview-layout">', unsafe_allow_html=True)
        workspace_col = st.container()

        with workspace_col:
            st.subheader("Annotation Workspace")

            editor_image_src = pil_image_to_data_url(full_result.convert("RGB"))

            # if st.button("Render export from editor"):
            #     st.session_state.export_request_id = str(uuid.uuid4())

            focus_objects_for_editor = (
                st.session_state.get("locked_focus_objects")
                if st.session_state.get("locked_focus_objects") is not None
                else focus_mask_to_focus_objects(
                    st.session_state.get("focus_mask"),
                    min_area=100,
                )
            )

            react_editor_state = helios_editor(
                image_src=editor_image_src,
                focus_objects=focus_objects_for_editor,
                ai_suggestions=st.session_state.get("ai_annotation_suggestions"),
                initial_state=st.session_state.get("react_editor_state"),
                debug=st.session_state.get("ui_mode", "Production") == "Debug",
                pending_inset_asset=pending_inset_asset_to_react(st.session_state),
                export_request_id=st.session_state.get("export_request_id"),
                key="helios_react_editor",
            )

            if react_editor_state:
                exported_data_url = react_editor_state.get("exportedImageDataUrl")
                returned_export_id = react_editor_state.get("exportRequestId")

                is_export_response = (
                        exported_data_url
                        and returned_export_id
                        and returned_export_id != st.session_state.get("last_consumed_export_request_id")
                )

                st.session_state["react_editor_state"] = react_editor_state

                if is_export_response:
                    header, encoded = exported_data_url.split(",", 1)
                    image_bytes = base64.b64decode(encoded)

                    exported_image = Image.open(BytesIO(image_bytes)).convert("RGB")

                    st.session_state.final_result_image = exported_image
                    st.session_state.export_ready = True
                    st.session_state.last_consumed_export_request_id = returned_export_id
                    st.session_state.export_request_id = None

                    st.success("Export rendered from editor.")
                    st.rerun()
                else:
                    # Normal editor update, not an export result
                    st.session_state.export_ready = False

            exported_data_url = react_editor_state.get("exportedImageDataUrl")
            returned_export_id = react_editor_state.get("exportRequestId")

            if (
                    exported_data_url
                    and returned_export_id
                    and returned_export_id != st.session_state.get("last_consumed_export_request_id")
            ):
                header, encoded = exported_data_url.split(",", 1)
                image_bytes = base64.b64decode(encoded)

                exported_image = Image.open(BytesIO(image_bytes)).convert("RGB")

                st.session_state.final_result_image = exported_image
                st.session_state.last_consumed_export_request_id = returned_export_id

                st.session_state.export_ready = True

                st.success("Export rendered from editor.")


if __name__ == "__main__":
    main()
