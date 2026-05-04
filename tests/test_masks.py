import hashlib
import streamlit as st
from PIL import Image
import numpy as np
import math

from ui.uploader import upload_image
from ui.annotation_canvas import (
    render_annotation_canvas,
    extract_canvas_points,
)
from vision.preprocessor import normalize_creoview_image
from vision.mask_handler import (
    create_color_mask,
    clean_mask,
    inspect_pixel,
    get_unique_colors,
)
from analysis.target_analyzer import analyze_focus_mask
from rendering.color_mapper import render_standardized_illustration
from rendering.callouts import draw_callouts
from rendering.editor_annotations import (
    extract_canvas_lines,
    scale_lines,
    draw_lines_on_image,
    scale_callouts,
    scale_insets,
    get_mask_bbox,
    draw_suggested_callouts,
    draw_ai_suggested_lines,
    draw_temporary_callout_points,
    build_inset_instance_from_preview_point,
)
from rendering.insets import draw_insets_on_image
from rendering.strategy_to_annotations import strategy_to_annotations
from exporting.export_config import EXPORT_PRESETS, EXPORT_BUNDLES
from exporting.exporter import (
    export_image_bytes,
    build_export_filename,
    get_export_dimensions,
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
from ui.workspace.inset_tool import render_inset_tool
from ui.workspace.callout_tool import render_callout_tool
from ui.workspace.line_tool import render_line_tool
from vision.mask_handler import (
    subtract_context_supported_regions_from_focus,
    subtract_non_focus_supported_regions,
    subtract_focus_overlap_from_context,
)

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
    inset_context_mask = clean_mask(inset_context_mask)

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

def main():

    st.set_page_config(page_title=f"Pantheon - {version}", layout="wide")

    with open("app/ui/style.css") as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
    st.title(f"Pantheon - {version}")

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
                context_mask = clean_mask(context_mask)

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

        # -------------------------
        # PHASE 2: RENDER + TOOLS
        # -------------------------

        else:

            # This part is new
            st.subheader("Menu")
            st.radio(
                " ",
                options=["Render", "Project", "Masks", "Export", "Settings"],
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

                    st.checkbox("Enhance focus detail", key="enhance_focus")
                    st.checkbox("Enhance contours", key="enhance_contour")

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

            elif section == "Project":
                st.subheader("Project")

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

                project_state = build_project_state(
                    uploaded_filename=uploaded_file.name if uploaded_file is not None else None,
                    session_state=st.session_state,
                )
                project_state_bytes = serialize_project_state(project_state)

                st.download_button(
                    label="Save Project JSON",
                    data=project_state_bytes,
                    file_name="illustration_project_state.json",
                    mime="application/json",
                    use_container_width=True,
                )

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
                        )
                        project_state_bytes = serialize_project_state(project_state)

                        zip_bytes = export_bundle_zip(
                            final_image=final_image,
                            raw_image=raw_image,
                            project_state_bytes=project_state_bytes,
                            base_name=export_base_name,
                            bundle_key=export_bundle_key,
                        )

                        st.download_button(
                            label="Download Export Package",
                            data=zip_bytes,
                            file_name=f"{export_base_name}_{export_bundle_key}.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"Export error: {e}")
                else:
                    st.info("Render an illustration first to enable export.")


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
    if st.session_state.result_image is None:

        if st.session_state.get("ui_mode", "Production") == "Debug":
            col1, col2 = st.columns(2)

            with col1:
                st.image(raw_image, caption="Original CAD Image", use_column_width=True)

            with col2:
                st.image(normalized_image, caption="Normalized Image (Used for Processing)", use_column_width=True)

            st.success("Image loaded and normalized.")

        else:
            st.image(raw_image, caption="Uploaded CAD Image", use_column_width=True)
            st.success("Image loaded.")

    if st.session_state.project_state_loaded_notice:
        st.success("Project state loaded. Re-render the illustration to rebuild the result.")
        st.session_state.project_state_loaded_notice = False

    if both_masks_exist and st.session_state.result_image is None:
        st.image(st.session_state.focus_mask, caption="Focus Mask", use_column_width=True)
        st.image(st.session_state.context_mask, caption="Context Mask", use_column_width=True)

    if full_result is not None:
        st.markdown("---")

        st.markdown('<div class="annotation-preview-layout">', unsafe_allow_html=True)
        workspace_col, preview_col = st.columns([1, 1], gap="large")

        with workspace_col:
            st.subheader("Annotation Workspace")

            current_tool = st.session_state.get("annotation_tool", "Line")
            tool_mode = current_tool.lower()

            with st.container():
                canvas_result, editor_image = render_annotation_canvas(
                    full_result,
                    tool_mode=tool_mode,
                    canvas_version=st.session_state.annotation_canvas_version,
                )

                preview_result = editor_image.convert("RGB")

                editor_w, editor_h = editor_image.size
                full_w, full_h = full_result.size
                scale_x = full_w / editor_w
                scale_y = full_h / editor_h

                # Shared workspace state
                editor_lines = st.session_state.get("manual_lines", [])
                approved_ai_lines = st.session_state.get("approved_ai_lines", [])
                ai_annotation_suggestions = st.session_state.get("ai_annotation_suggestions")

                st.radio(
                    "Tool",
                    options=["Line", "Callout", "Inset"],
                    horizontal=True,
                    key="annotation_tool",
                )
                tool_mode = st.session_state.get("annotation_tool", "Line").lower()

                if tool_mode == "line":
                    preview_result = render_line_tool(
                        canvas_result=canvas_result,
                        editor_image=editor_image,
                        preview_result=preview_result,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        session_state=st.session_state,
                    )

                else:
                    if editor_lines:
                        preview_result = draw_lines_on_image(
                            preview_result,
                            editor_lines,
                            halo=True,
                            halo_color=(255, 255, 255),
                            halo_extra_thickness=3,
                        )

                if tool_mode == "callout":
                    preview_result = render_callout_tool(
                        canvas_result=canvas_result,
                        preview_result=preview_result,
                        session_state=st.session_state,
                    )

                if tool_mode == "inset":
                    preview_result = render_inset_tool(
                        canvas_result=canvas_result,
                        editor_image=editor_image,
                        full_result=full_result,
                        preview_result=preview_result,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        session_state=st.session_state,
                        update_leader_fn=_update_selected_inset_leader_target,
                        update_position_fn=_update_selected_inset_position_from_preview_click,
                        preview_reposition_fn=_build_preview_repositioned_inset,
                        apply_size_preset_fn=_apply_inset_size_preset,
                    )

                if approved_ai_lines:
                    preview_approved_ai_lines = []
                    for line in approved_ai_lines:
                        preview_approved_ai_lines.append(
                            {
                                "start": [
                                    int(round(line["start"][0] / scale_x)),
                                    int(round(line["start"][1] / scale_y)),
                                ],
                                "end": [
                                    int(round(line["end"][0] / scale_x)),
                                    int(round(line["end"][1] / scale_y)),
                                ],
                            }
                        )

                    preview_result = draw_ai_suggested_lines(
                        preview_result,
                        preview_approved_ai_lines,
                        color=(0, 0, 0),
                        stroke_width=2,
                        halo=True,
                        halo_color=(255, 255, 255),
                        halo_extra_thickness=3,
                    )

                if ai_annotation_suggestions is not None and ai_annotation_suggestions.get("movement_lines"):
                    preview_pending_ai_lines = []
                    for line in ai_annotation_suggestions["movement_lines"]:
                        preview_pending_ai_lines.append(
                            {
                                "start": [
                                    int(round(line["start"][0] / scale_x)),
                                    int(round(line["start"][1] / scale_y)),
                                ],
                                "end": [
                                    int(round(line["end"][0] / scale_x)),
                                    int(round(line["end"][1] / scale_y)),
                                ],
                            }
                        )

                    preview_result = draw_ai_suggested_lines(
                        preview_result,
                        preview_pending_ai_lines,
                        color=(0, 0, 0),
                        stroke_width=2,
                        halo=True,
                        halo_color=(255, 255, 255),
                        halo_extra_thickness=3,
                    )

                if st.session_state.callouts and tool_mode != "callout":
                    preview_result = draw_callouts(
                        image=preview_result,
                        callouts=st.session_state.callouts,
                        radius=18,
                        line_thickness=2,
                        font_scale=0.65,
                        text_thickness=2,
                    )

                if ai_annotation_suggestions is not None and ai_annotation_suggestions.get("callouts"):
                    preview_pending_ai_callouts = []
                    for c in ai_annotation_suggestions["callouts"]:
                        preview_pending_ai_callouts.append(
                            {
                                "label": c["label"],
                                "circle": [
                                    int(round(c["circle"][0] / scale_x)),
                                    int(round(c["circle"][1] / scale_y)),
                                ],
                                "end": [
                                    int(round(c["end"][0] / scale_x)),
                                    int(round(c["end"][1] / scale_y)),
                                ],
                            }
                        )

                    preview_result = draw_suggested_callouts(
                        preview_result,
                        preview_pending_ai_callouts,
                        radius=18,
                        line_thickness=2,
                        font_scale=0.65,
                        text_thickness=2,
                    )

                if st.session_state.insets and tool_mode != "inset":
                    preview_insets = scale_insets(
                        st.session_state.insets,
                        scale_x=1 / scale_x,
                        scale_y=1 / scale_y,
                    )
                    preview_result = draw_insets_on_image(
                        image=preview_result,
                        insets=preview_insets,
                        inset_assets=st.session_state.inset_assets,
                    )
                st.markdown('</div>', unsafe_allow_html=True)

        with preview_col:
            st.subheader("Final Preview")
            st.image(
                preview_result,
                caption="Final Annotated Illustration",
                width=editor_image.size[0],
            )

        full_export_result = full_result.convert("RGB")

        editor_lines = st.session_state.get("manual_lines", [])
        approved_ai_lines = st.session_state.get("approved_ai_lines", [])

        if editor_lines:
            full_export_result = draw_lines_on_image(
                full_export_result,
                scale_lines(editor_lines, scale_x=scale_x, scale_y=scale_y),
                halo=True,
                halo_color=(255, 255, 255),
                halo_extra_thickness=max(3, int(round(3 * ((scale_x + scale_y) / 2.0)))),
            )

        if approved_ai_lines:
            avg_scale = (scale_x + scale_y) / 2.0
            approved_ai_line_objects = []
            for line in approved_ai_lines:
                approved_ai_line_objects.append(
                    {
                        "x1": int(line["start"][0]),
                        "y1": int(line["start"][1]),
                        "x2": int(line["end"][0]),
                        "y2": int(line["end"][1]),
                        "stroke_width": max(2, int(round(2 * avg_scale))),
                    }
                )

            full_export_result = draw_lines_on_image(
                full_export_result,
                approved_ai_line_objects,
                halo=True,
                halo_color=(255, 255, 255),
                halo_extra_thickness=max(3, int(round(3 * avg_scale))),
            )

        if st.session_state.callouts:
            avg_scale = (scale_x + scale_y) / 2.0
            scaled_callouts = scale_callouts(
                st.session_state.callouts,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            full_export_result = draw_callouts(
                image=full_export_result,
                callouts=scaled_callouts,
                radius=max(18, int(round(18 * avg_scale))),
                line_thickness=max(2, int(round(2 * avg_scale))),
                font_scale=0.65 * avg_scale,
                text_thickness=max(2, int(round(2 * avg_scale))),
            )

        if st.session_state.insets:
            full_export_result = draw_insets_on_image(
                image=full_export_result,
                insets=st.session_state.insets,
                inset_assets=st.session_state.inset_assets,
            )

        st.session_state.final_result_image = full_export_result


if __name__ == "__main__":
    main()