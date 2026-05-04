from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import cv2
import numpy as np
from PIL import Image


@dataclass
class HoleFeature:
    bbox: tuple[int, int, int, int]   # (x_min, y_min, x_max, y_max)
    centroid: tuple[int, int]
    area: int


@dataclass
class TargetComponent:
    component_id: int
    area: int
    bbox: tuple[int, int, int, int]   # (x_min, y_min, x_max, y_max)
    centroid: tuple[int, int]
    width: int
    height: int
    aspect_ratio: float               # width / height (kept for backward compatibility)
    elongation_ratio: float           # long_side / short_side
    is_thin_component: bool
    hole_count: int
    holes: list[HoleFeature]
    primary_hole_center: tuple[int, int] | None
    is_terminal_like: bool


@dataclass
class TargetAnalysis:
    image_size: tuple[int, int]  # (width, height)
    component_count: int
    total_focus_area: int
    focus_bbox: tuple[int, int, int, int] | None
    largest_component_area: int
    median_component_area: int
    repeated_small_parts: bool
    dominant_structure: str
    components: list[TargetComponent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "component_count": self.component_count,
            "total_focus_area": self.total_focus_area,
            "focus_bbox": self.focus_bbox,
            "largest_component_area": self.largest_component_area,
            "median_component_area": self.median_component_area,
            "repeated_small_parts": self.repeated_small_parts,
            "dominant_structure": self.dominant_structure,
            "components": [asdict(c) for c in self.components],
        }


def _mask_to_binary(mask: Image.Image) -> np.ndarray:
    mask_np = np.array(mask.convert("L"))
    return (mask_np > 0).astype(np.uint8)


def _compute_global_bbox(binary_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)


def _elongation_ratio(width: int, height: int) -> float:
    long_side = max(width, height)
    short_side = max(1, min(width, height))
    return float(long_side) / float(short_side)


def _component_local_mask(
        labels: np.ndarray,
        label_id: int,
        bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    region = labels[y1:y2 + 1, x1:x2 + 1]
    return (region == label_id).astype(np.uint8)


def _detect_holes_in_component_mask(
        local_mask: np.ndarray,
        bbox_offset: tuple[int, int],
        min_hole_area: int = 20,
) -> list[HoleFeature]:
    """
    Detect enclosed empty regions (holes) inside a single component mask.
    local_mask must be binary uint8 with values 0/1.
    """
    x_off, y_off = bbox_offset
    h, w = local_mask.shape

    if h == 0 or w == 0:
        return []

    # Component = 255, background = 0
    comp = (local_mask * 255).astype(np.uint8)

    # Invert: empty space becomes 255
    inv = cv2.bitwise_not(comp)

    # Flood-fill all border-connected empty space with 0
    flood = inv.copy()
    floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood from every border pixel that is empty
    border_points = []

    for x in range(w):
        border_points.append((x, 0))
        border_points.append((x, h - 1))
    for y in range(h):
        border_points.append((0, y))
        border_points.append((w - 1, y))

    for px, py in border_points:
        if flood[py, px] == 255:
            cv2.floodFill(flood, floodfill_mask, (px, py), 0)

    # Remaining white regions are true enclosed holes
    holes_mask = (flood > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        holes_mask,
        connectivity=8,
    )

    holes: list[HoleFeature] = []

    for hole_id in range(1, num_labels):
        area = int(stats[hole_id, cv2.CC_STAT_AREA])
        if area < min_hole_area:
            continue

        x = int(stats[hole_id, cv2.CC_STAT_LEFT])
        y = int(stats[hole_id, cv2.CC_STAT_TOP])
        hole_w = int(stats[hole_id, cv2.CC_STAT_WIDTH])
        hole_h = int(stats[hole_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[hole_id]

        holes.append(
            HoleFeature(
                bbox=(
                    x_off + x,
                    y_off + y,
                    x_off + x + hole_w - 1,
                    y_off + y + hole_h - 1,
                ),
                centroid=(int(round(x_off + cx)), int(round(y_off + cy))),
                area=area,
            )
        )

    holes.sort(key=lambda h0: h0.area, reverse=True)
    return holes


def _classify_structure(
        components: list[TargetComponent],
        image_area: int,
) -> tuple[bool, str]:
    """
    Simple first-pass classification.

    repeated_small_parts:
        True if there are several similarly sized small components.

    dominant_structure:
        - "single_large_part"
        - "multiple_repeated_small_parts"
        - "multiple_mixed_parts"
        - "sparse_small_parts"
        - "none"
    """
    if not components:
        return False, "none"

    areas = np.array([c.area for c in components], dtype=np.float32)
    count = len(areas)

    largest = float(np.max(areas))
    median = float(np.median(areas))
    total = float(np.sum(areas))

    largest_ratio_to_image = largest / max(image_area, 1)
    largest_ratio_to_total = largest / max(total, 1.0)

    if median > 0:
        similar = np.logical_and(areas >= 0.5 * median, areas <= 2.0 * median)
        similar_count = int(np.sum(similar))
    else:
        similar_count = 0

    repeated_small_parts = (
            count >= 4
            and median / max(image_area, 1) < 0.01
            and similar_count >= max(3, count // 2)
    )

    if count >= 2 and largest_ratio_to_total >= 0.80:
        return False, "single_large_part"

    if repeated_small_parts and largest_ratio_to_total < 0.75:
        return True, "multiple_repeated_small_parts"

    if count >= 2 and largest_ratio_to_total < 0.75:
        return False, "multiple_mixed_parts"

    if count >= 2:
        return False, "sparse_small_parts"

    return False, "single_large_part"


def analyze_focus_mask(
        focus_mask: Image.Image,
        min_component_area: int = 20,
        simplify_components: bool = True,
        min_hole_area: int = 20,
        thin_component_ratio: float = 2.5,
) -> TargetAnalysis:
    """
    Analyze a focus mask and extract connected components and summary statistics.

    New fields per component:
    - elongation_ratio
    - is_thin_component
    - hole_count
    - holes
    - primary_hole_center
    - is_terminal_like
    """
    binary = _mask_to_binary(focus_mask)

    if simplify_components:
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    h, w = binary.shape
    image_area = w * h

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    components: list[TargetComponent] = []

    for label_id in range(1, num_labels):  # skip background
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        comp_w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < min_component_area:
            continue

        cx, cy = centroids[label_id]
        bbox = (x, y, x + comp_w - 1, y + comp_h - 1)

        local_mask = _component_local_mask(labels, label_id, bbox)
        holes = _detect_holes_in_component_mask(
            local_mask=local_mask,
            bbox_offset=(x, y),
            min_hole_area=min_hole_area,
        )

        elongation = _elongation_ratio(comp_w, comp_h)
        is_thin_component = elongation >= thin_component_ratio
        hole_count = len(holes)
        primary_hole_center = holes[0].centroid if holes else None

        # First-pass heuristic:
        # thin + has hole => likely terminal/tab/eyelet-like geometry
        is_terminal_like = is_thin_component and hole_count > 0

        components.append(
            TargetComponent(
                component_id=label_id,
                area=area,
                bbox=bbox,
                centroid=(int(round(cx)), int(round(cy))),
                width=comp_w,
                height=comp_h,
                aspect_ratio=(comp_w / comp_h) if comp_h > 0 else 0.0,
                elongation_ratio=elongation,
                is_thin_component=is_thin_component,
                hole_count=hole_count,
                holes=holes,
                primary_hole_center=primary_hole_center,
                is_terminal_like=is_terminal_like,
            )
        )

    components.sort(key=lambda c: c.area, reverse=True)

    global_bbox = _compute_global_bbox(binary)

    if components:
        areas = [c.area for c in components]
        largest_component_area = int(max(areas))
        median_component_area = int(np.median(areas))
        total_focus_area = int(sum(areas))
    else:
        largest_component_area = 0
        median_component_area = 0
        total_focus_area = 0

    repeated_small_parts, dominant_structure = _classify_structure(components, image_area)

    return TargetAnalysis(
        image_size=(w, h),
        component_count=len(components),
        total_focus_area=total_focus_area,
        focus_bbox=global_bbox,
        largest_component_area=largest_component_area,
        median_component_area=median_component_area,
        repeated_small_parts=repeated_small_parts,
        dominant_structure=dominant_structure,
        components=components,
    )