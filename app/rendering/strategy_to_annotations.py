from __future__ import annotations

from typing import Any
import math

import numpy as np

from analysis.target_analyzer import analyze_focus_mask


VALID_DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "up-right": (1, -1),
    "up-left": (-1, -1),
    "down-right": (1, 1),
    "down-left": (-1, 1),
    "outward": (1, -1),
    "inward": (-1, 1),
    "none": (0, 0),
}


def _normalize_vec(dx: float, dy: float) -> tuple[float, float]:
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return 0.0, 0.0
    return dx / length, dy / length


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)


def _offset_point(point: tuple[int, int], dx: float, dy: float, distance: float) -> tuple[int, int]:
    x, y = point
    return int(round(x + dx * distance)), int(round(y + dy * distance))


def _perp_vec(dx: float, dy: float) -> tuple[float, float]:
    return -dy, dx


def _clip_point(point: tuple[int, int], image_size: tuple[int, int], margin: int = 5) -> tuple[int, int]:
    w, h = image_size
    x = min(max(point[0], margin), max(margin, w - 1 - margin))
    y = min(max(point[1], margin), max(margin, h - 1 - margin))
    return x, y


def _center_anchor(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return _bbox_center(bbox)


def _recommended_line_length(
        bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
        multiplier: float = 1.35,
        minimum: int = 90,
        maximum_ratio: float = 0.22,
) -> int:
    bw, bh = _bbox_size(bbox)
    base = max(bw, bh)
    w, h = image_size
    max_len = int(max(w, h) * maximum_ratio)
    return max(minimum, min(int(round(base * multiplier)), max_len))


def _group_callout_for_bbox(
        bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
        label: str = "A",
) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    w, h = image_size

    circle = (min(x2 + 30, w - 20), max(y1 - 20, 20))
    target = _bbox_center(bbox)

    circle = _clip_point(circle, image_size, margin=20)
    target = _clip_point(target, image_size, margin=5)

    return {
        "label": label,
        "circle": [circle[0], circle[1]],
        "end": [target[0], target[1]],
    }


def _separate_callouts_for_components(
        components: list[dict[str, Any]],
        image_size: tuple[int, int],
        max_callouts: int = 2,
) -> list[dict[str, Any]]:
    labels = ["A", "B", "C", "D", "E", "F", "1", "2", "3", "4"]
    out = []

    for idx, comp in enumerate(components[:max_callouts]):
        bbox = tuple(comp["bbox"])
        callout = _group_callout_for_bbox(
            bbox=bbox,
            image_size=image_size,
            label=labels[idx % len(labels)],
        )
        out.append(callout)

    return out


def _pick_spread_components(
        components: list[dict[str, Any]],
        direction: str,
        max_count: int = 3,
) -> list[dict[str, Any]]:
    if not components:
        return []

    if len(components) <= max_count:
        return components[:]

    dx, dy = VALID_DIRECTIONS.get(direction, (0, -1))
    dx, dy = _normalize_vec(dx, dy)
    px, py = _perp_vec(dx, dy)

    projected = []
    for comp in components:
        cx, cy = comp["centroid"]
        score = cx * px + cy * py
        projected.append((score, comp))

    projected.sort(key=lambda item: item[0])
    sorted_components = [item[1] for item in projected]

    if max_count == 1:
        return [sorted_components[len(sorted_components) // 2]]
    if max_count == 2:
        return [sorted_components[0], sorted_components[-1]]

    return [
        sorted_components[0],
        sorted_components[len(sorted_components) // 2],
        sorted_components[-1],
    ]


def _recommended_repeated_line_count(
        components: list[dict[str, Any]],
        ux: float,
        uy: float,
) -> int:
    if not components:
        return 0

    if len(components) <= 2:
        return len(components)

    ux, uy = _normalize_vec(ux, uy)
    px, py = _perp_vec(ux, uy)

    projections = []
    for comp in components:
        cx, cy = comp["centroid"]
        projections.append(cx * px + cy * py)

    if not projections:
        return min(3, len(components))

    spread = max(projections) - min(projections)

    if spread < 90:
        return 2

    return 3


def _mask_to_binary(mask_image) -> np.ndarray:
    mask_np = np.array(mask_image.convert("L"))
    return mask_np > 0


def _sample_line_points(
        start: tuple[int, int],
        end: tuple[int, int],
        steps: int = 60,
) -> list[tuple[int, int]]:
    x1, y1 = start
    x2, y2 = end
    pts = []

    for i in range(steps + 1):
        t = i / max(steps, 1)
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        pts.append((x, y))

    deduped = []
    seen = set()
    for p in pts:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return deduped


def _line_overlap_score(
        focus_binary: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        ignore_prefix_ratio: float = 0.22,
) -> float:
    h, w = focus_binary.shape
    pts = _sample_line_points(start, end, steps=70)

    if not pts:
        return 0.0

    ignore_count = int(len(pts) * ignore_prefix_ratio)
    relevant = pts[ignore_count:] if ignore_count < len(pts) else []

    if not relevant:
        return 0.0

    hits = 0
    total = 0

    for x, y in relevant:
        if 0 <= x < w and 0 <= y < h:
            total += 1
            if focus_binary[y, x]:
                hits += 1

    if total == 0:
        return 0.0

    return hits / total


def _segment_distance_score(
        start: tuple[int, int],
        end: tuple[int, int],
        existing_lines: list[dict[str, Any]],
) -> float:
    if not existing_lines:
        return 0.0

    my_pts = _sample_line_points(start, end, steps=24)
    if not my_pts:
        return 0.0

    penalty = 0.0

    for line in existing_lines:
        other_start = tuple(line["start"])
        other_end = tuple(line["end"])
        other_pts = _sample_line_points(other_start, other_end, steps=24)

        for x1, y1 in my_pts:
            for x2, y2 in other_pts:
                dist = math.hypot(x1 - x2, y1 - y2)
                if dist < 10:
                    penalty += (10 - dist) / 10.0

    return penalty


def _direction_unit_from_name(direction: str) -> tuple[float, float]:
    dx, dy = VALID_DIRECTIONS.get(direction, (0, 0))
    return _normalize_vec(dx, dy)


def _direction_unit_from_strategy(ai_strategy: dict[str, Any]) -> tuple[float, float]:
    vector = ai_strategy.get("suggested_direction_vector")

    if isinstance(vector, (list, tuple)) and len(vector) == 2:
        try:
            dx = float(vector[0])
            dy = float(vector[1])
            return _normalize_vec(dx, dy)
        except (TypeError, ValueError):
            pass

    direction = ai_strategy.get("suggested_direction", "none")
    return _direction_unit_from_name(direction)


def _strategy_has_custom_direction_vector(ai_strategy: dict[str, Any]) -> bool:
    vector = ai_strategy.get("suggested_direction_vector")
    return isinstance(vector, (list, tuple)) and len(vector) == 2


def _component_edge_anchor(
        component: dict[str, Any],
        ux: float,
        uy: float,
) -> tuple[int, int]:
    x1, y1, x2, y2 = component["bbox"]
    cx, cy = component["centroid"]
    w, h = _bbox_size(component["bbox"])

    half_w = max(1.0, w / 2.0)
    half_h = max(1.0, h / 2.0)

    tx = half_w * ux
    ty = half_h * uy

    ax = int(round(cx + tx))
    ay = int(round(cy + ty))

    ax = min(max(ax, x1), x2)
    ay = min(max(ay, y1), y2)

    return ax, ay


def _local_shape_direction(
        component: dict[str, Any],
) -> tuple[float, float]:
    x1, y1, x2, y2 = component["bbox"]
    w = max(1, x2 - x1 + 1)
    h = max(1, y2 - y1 + 1)

    if w > h * 1.25:
        return 1.0, 0.0

    if h > w * 1.25:
        return 0.0, 1.0

    return 0.0, 0.0


def _align_local_with_global(
        local_ux: float,
        local_uy: float,
        global_ux: float,
        global_uy: float,
) -> tuple[float, float]:
    dot = local_ux * global_ux + local_uy * global_uy
    if dot < 0:
        return -local_ux, -local_uy
    return local_ux, local_uy


def _blend_unit_vectors(
        ux1: float,
        uy1: float,
        ux2: float,
        uy2: float,
        weight_global: float = 0.65,
        weight_local: float = 0.35,
) -> tuple[float, float]:
    bx = ux1 * weight_global + ux2 * weight_local
    by = uy1 * weight_global + uy2 * weight_local
    return _normalize_vec(bx, by)


def _single_part_direction_with_local_bias(
        component: dict[str, Any],
        global_ux: float,
        global_uy: float,
        global_weight: float = 0.65,
        local_weight: float = 0.35,
) -> tuple[tuple[int, int], tuple[float, float]]:
    local_ux, local_uy = _local_shape_direction(component)

    if local_ux == 0.0 and local_uy == 0.0:
        final_ux, final_uy = global_ux, global_uy
    else:
        local_ux, local_uy = _align_local_with_global(
            local_ux, local_uy, global_ux, global_uy
        )

        final_ux, final_uy = _blend_unit_vectors(
            ux1=global_ux,
            uy1=global_uy,
            ux2=local_ux,
            uy2=local_uy,
            weight_global=global_weight,
            weight_local=local_weight,
        )

    blended_anchor = _component_edge_anchor(component, final_ux, final_uy)
    return blended_anchor, (final_ux, final_uy)


def _available_space_in_direction(
        point: tuple[int, int],
        ux: float,
        uy: float,
        image_size: tuple[int, int],
        margin: int = 5,
) -> float:
    x, y = point
    w, h = image_size

    candidates: list[float] = []

    if ux > 0:
        candidates.append((w - 1 - margin - x) / ux)
    elif ux < 0:
        candidates.append((margin - x) / ux)

    if uy > 0:
        candidates.append((h - 1 - margin - y) / uy)
    elif uy < 0:
        candidates.append((margin - y) / uy)

    positive = [c for c in candidates if c >= 0]
    if not positive:
        return 0.0

    return min(positive)


def _softened_anchor_for_tight_forward_space(
        component: dict[str, Any],
        ux: float,
        uy: float,
        image_size: tuple[int, int],
        min_forward_space: float = 55.0,
        full_edge_space: float = 95.0,
) -> tuple[int, int]:
    edge_anchor = _component_edge_anchor(component, ux, uy)
    center_anchor = tuple(component["centroid"])

    edge_space = _available_space_in_direction(
        point=edge_anchor,
        ux=ux,
        uy=uy,
        image_size=image_size,
        margin=5,
    )

    if edge_space >= full_edge_space:
        return edge_anchor

    if edge_space <= min_forward_space:
        return center_anchor

    t = (edge_space - min_forward_space) / max(1.0, (full_edge_space - min_forward_space))

    ax = int(round(center_anchor[0] * (1.0 - t) + edge_anchor[0] * t))
    ay = int(round(center_anchor[1] * (1.0 - t) + edge_anchor[1] * t))
    return ax, ay


def _build_collision_aware_single_line_with_unit(
        anchor: tuple[int, int],
        ux: float,
        uy: float,
        image_size: tuple[int, int],
        focus_binary: np.ndarray,
        line_length: int = 100,
        offset: int = 18,
        spacing: int = 14,
        existing_lines: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    ux, uy = _normalize_vec(ux, uy)
    if ux == 0 and uy == 0:
        return None

    existing_lines = existing_lines or []
    px, py = _perp_vec(ux, uy)

    base_offset = max(4, min(offset, int(line_length * 0.08)))

    if abs(uy) > abs(ux):
        perp_shifts = [-1, 0, 1]
        forward_shifts = [0, 2, 4]
    else:
        perp_shifts = [-2, -1, 0, 1, 2]
        forward_shifts = [0, 4, 8]

    best_line = None
    best_score = float("inf")

    for perp_mult in perp_shifts:
        for forward_extra in forward_shifts:
            shifted_anchor = (
                int(round(anchor[0] + px * spacing * perp_mult)),
                int(round(anchor[1] + py * spacing * perp_mult)),
            )

            available_forward_space = _available_space_in_direction(
                point=shifted_anchor,
                ux=ux,
                uy=uy,
                image_size=image_size,
                margin=5,
            )

            effective_offset = min(base_offset + forward_extra, max(0.0, available_forward_space - 8.0))
            effective_offset = max(0.0, effective_offset)

            raw_start_point = _offset_point(shifted_anchor, ux, uy, effective_offset)
            raw_end_point = _offset_point(raw_start_point, ux, uy, line_length)

            start_point = _clip_point(raw_start_point, image_size)
            end_point = _clip_point(raw_end_point, image_size)

            visible_length = math.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])
            raw_length = max(
                1.0,
                math.hypot(raw_end_point[0] - raw_start_point[0], raw_end_point[1] - raw_start_point[1])
            )

            if visible_length < 28:
                continue

            overlap_score = _line_overlap_score(
                focus_binary=focus_binary,
                start=start_point,
                end=end_point,
                ignore_prefix_ratio=0.22,
            )

            distance_penalty = _segment_distance_score(
                start=start_point,
                end=end_point,
                existing_lines=existing_lines,
            )

            shift_penalty = abs(perp_mult) * 0.03 + forward_extra * 0.001

            clipping_ratio = 1.0 - (visible_length / raw_length)
            clipping_penalty = max(0.0, clipping_ratio) * 2.0

            total_score = (
                    overlap_score * 3.0
                    + distance_penalty * 0.7
                    + shift_penalty
                    + clipping_penalty
            )

            if total_score < best_score:
                best_score = total_score
                best_line = {
                    "start": [start_point[0], start_point[1]],
                    "end": [end_point[0], end_point[1]],
                }

    return best_line


def _build_collision_aware_single_line(
        anchor: tuple[int, int],
        direction: str,
        image_size: tuple[int, int],
        focus_binary: np.ndarray,
        line_length: int = 100,
        offset: int = 18,
        spacing: int = 14,
        existing_lines: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    ux, uy = _direction_unit_from_name(direction)
    return _build_collision_aware_single_line_with_unit(
        anchor=anchor,
        ux=ux,
        uy=uy,
        image_size=image_size,
        focus_binary=focus_binary,
        line_length=line_length,
        offset=offset,
        spacing=spacing,
        existing_lines=existing_lines,
    )


def strategy_to_annotations(
        focus_mask,
        ai_strategy: dict[str, Any],
        image_size: tuple[int, int],
) -> dict[str, Any]:
    analysis = analyze_focus_mask(focus_mask)
    analysis_dict = analysis.to_dict()

    focus_binary = _mask_to_binary(focus_mask)
    components = analysis_dict["components"]
    focus_bbox = analysis_dict["focus_bbox"]

    direction = ai_strategy.get("suggested_direction", "none")
    movement_line_strategy = ai_strategy.get("movement_line_strategy", "none")
    callout_strategy = ai_strategy.get("callout_strategy", "none")
    max_callouts = int(ai_strategy.get("max_callouts", 0) or 0)
    has_custom_direction_vector = _strategy_has_custom_direction_vector(ai_strategy)

    movement_lines: list[dict[str, Any]] = []
    callouts: list[dict[str, Any]] = []
    notes: list[str] = []

    if movement_line_strategy != "none" and direction != "none":
        is_small_parts = analysis_dict["repeated_small_parts"]
        global_ux, global_uy = _direction_unit_from_strategy(ai_strategy)

        if movement_line_strategy == "short_parallel_near_focus":
            if focus_bbox is not None:
                bbox = tuple(focus_bbox)

                if is_small_parts and components:
                    line_length = _recommended_line_length(
                        bbox=bbox,
                        image_size=image_size,
                        multiplier=1.25,
                        minimum=90,
                        maximum_ratio=0.20,
                    )

                    line_count = _recommended_repeated_line_count(
                        components=components,
                        ux=global_ux,
                        uy=global_uy,
                    )

                    selected_components = _pick_spread_components(
                        components=components,
                        direction=direction,
                        max_count=min(3, line_count),
                    )

                    for comp in selected_components:
                        anchor = _component_edge_anchor(comp, global_ux, global_uy)

                        line = _build_collision_aware_single_line_with_unit(
                            anchor=anchor,
                            ux=global_ux,
                            uy=global_uy,
                            image_size=image_size,
                            focus_binary=focus_binary,
                            line_length=line_length,
                            offset=4,
                            spacing=8,
                            existing_lines=movement_lines,
                        )
                        if line is not None:
                            movement_lines.append(line)
                else:
                    if components:
                        primary_component = max(
                            components,
                            key=lambda comp: comp.get("area", 0)
                        )

                        if has_custom_direction_vector:
                            anchor, (single_ux, single_uy) = _single_part_direction_with_local_bias(
                                component=primary_component,
                                global_ux=global_ux,
                                global_uy=global_uy,
                                global_weight=0.65,
                                local_weight=0.35,
                            )
                        else:
                            anchor = _softened_anchor_for_tight_forward_space(
                                component=primary_component,
                                ux=global_ux,
                                uy=global_uy,
                                image_size=image_size,
                                min_forward_space=55.0,
                                full_edge_space=95.0,
                            )
                            single_ux, single_uy = global_ux, global_uy
                    else:
                        anchor = _center_anchor(bbox)
                        single_ux, single_uy = global_ux, global_uy

                    line_length = _recommended_line_length(
                        bbox=bbox,
                        image_size=image_size,
                        multiplier=1.25,
                        minimum=90,
                        maximum_ratio=0.20,
                    )

                    line = _build_collision_aware_single_line_with_unit(
                        anchor=anchor,
                        ux=single_ux,
                        uy=single_uy,
                        image_size=image_size,
                        focus_binary=focus_binary,
                        line_length=line_length,
                        offset=8,
                        spacing=14,
                        existing_lines=movement_lines,
                    )
                    if line is not None:
                        movement_lines.append(line)

        elif movement_line_strategy == "two_lines_per_large_part":
            for comp in components[:2]:
                bbox = tuple(comp["bbox"])

                if has_custom_direction_vector:
                    anchor, (single_ux, single_uy) = _single_part_direction_with_local_bias(
                        component=comp,
                        global_ux=global_ux,
                        global_uy=global_uy,
                        global_weight=0.65,
                        local_weight=0.35,
                    )
                else:
                    anchor = _softened_anchor_for_tight_forward_space(
                        component=comp,
                        ux=global_ux,
                        uy=global_uy,
                        image_size=image_size,
                        min_forward_space=55.0,
                        full_edge_space=95.0,
                    )
                    single_ux, single_uy = global_ux, global_uy

                line_length = _recommended_line_length(
                    bbox=bbox,
                    image_size=image_size,
                    multiplier=1.35,
                    minimum=100,
                    maximum_ratio=0.24,
                )

                line = _build_collision_aware_single_line_with_unit(
                    anchor=anchor,
                    ux=single_ux,
                    uy=single_uy,
                    image_size=image_size,
                    focus_binary=focus_binary,
                    line_length=line_length,
                    offset=8,
                    spacing=12,
                    existing_lines=movement_lines,
                )
                if line is not None:
                    movement_lines.append(line)

        elif movement_line_strategy == "grouped_repeated_targets":
            if focus_bbox is not None:
                bbox = tuple(focus_bbox)

                if is_small_parts and components:
                    line_length = _recommended_line_length(
                        bbox=bbox,
                        image_size=image_size,
                        multiplier=1.15,
                        minimum=85,
                        maximum_ratio=0.18,
                    )

                    line_count = _recommended_repeated_line_count(
                        components=components,
                        ux=global_ux,
                        uy=global_uy,
                    )

                    selected_components = _pick_spread_components(
                        components=components,
                        direction=direction,
                        max_count=min(3, line_count),
                    )

                    for comp in selected_components:
                        anchor = _component_edge_anchor(comp, global_ux, global_uy)

                        line = _build_collision_aware_single_line_with_unit(
                            anchor=anchor,
                            ux=global_ux,
                            uy=global_uy,
                            image_size=image_size,
                            focus_binary=focus_binary,
                            line_length=line_length,
                            offset=4,
                            spacing=8,
                            existing_lines=movement_lines,
                        )
                        if line is not None:
                            movement_lines.append(line)
                else:
                    if components:
                        primary_component = max(
                            components,
                            key=lambda comp: comp.get("area", 0)
                        )

                        if has_custom_direction_vector:
                            anchor, (single_ux, single_uy) = _single_part_direction_with_local_bias(
                                component=primary_component,
                                global_ux=global_ux,
                                global_uy=global_uy,
                                global_weight=0.65,
                                local_weight=0.35,
                            )
                        else:
                            anchor = _softened_anchor_for_tight_forward_space(
                                component=primary_component,
                                ux=global_ux,
                                uy=global_uy,
                                image_size=image_size,
                                min_forward_space=55.0,
                                full_edge_space=95.0,
                            )
                            single_ux, single_uy = global_ux, global_uy
                    else:
                        anchor = _center_anchor(bbox)
                        single_ux, single_uy = global_ux, global_uy

                    line_length = _recommended_line_length(
                        bbox=bbox,
                        image_size=image_size,
                        multiplier=1.15,
                        minimum=85,
                        maximum_ratio=0.18,
                    )

                    line = _build_collision_aware_single_line_with_unit(
                        anchor=anchor,
                        ux=single_ux,
                        uy=single_uy,
                        image_size=image_size,
                        focus_binary=focus_binary,
                        line_length=line_length,
                        offset=8,
                        spacing=12,
                        existing_lines=movement_lines,
                    )
                    if line is not None:
                        movement_lines.append(line)

    if callout_strategy == "single_main_callout":
        if focus_bbox is not None:
            callouts.append(
                _group_callout_for_bbox(
                    tuple(focus_bbox),
                    image_size,
                    label="A",
                )
            )

    elif callout_strategy == "grouped_callout":
        if focus_bbox is not None:
            label = "1" if analysis_dict["repeated_small_parts"] else "A"
            callouts.append(
                _group_callout_for_bbox(
                    tuple(focus_bbox),
                    image_size,
                    label=label,
                )
            )

    elif callout_strategy == "separate_main_parts":
        callouts.extend(
            _separate_callouts_for_components(
                components=components,
                image_size=image_size,
                max_callouts=max(1, min(max_callouts or 2, 4)),
            )
        )

    notes.extend(ai_strategy.get("notes", []))
    notes.append("Improved repeated-part routing enabled.")

    return {
        "movement_lines": movement_lines,
        "callouts": callouts,
        "notes": notes,
        "analysis": analysis_dict,
    }