import numpy as np
import cv2
from PIL import Image


def _clean_binary_mask(mask_np: np.ndarray) -> np.ndarray:
    """
    Internal helper to clean a binary mask.
    """
    mask_np = (mask_np > 0).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_DILATE, kernel)

    return mask_np


def create_color_mask_rgb(image, target_rgb=(255, 0, 255), tolerance=50):
    """
    RGB-distance based mask extraction.
    Works best for clean images with minimal shadow/reflection distortion.
    """
    image_np = np.array(image.convert("RGB"))

    diff = np.linalg.norm(image_np - np.array(target_rgb), axis=2)
    mask = diff < tolerance
    mask = (mask * 255).astype(np.uint8)

    mask = _clean_binary_mask(mask)
    return Image.fromarray(mask)


def create_color_mask_hsv(
        image,
        target="magenta",
        hue_tolerance=15,
        min_saturation=60,
        min_value=30,
):
    """
    HSV-based mask extraction.
    Much more robust to shadows than RGB-distance masking.

    target:
        - "magenta"
        - "cyan"
    """
    image_np = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0].astype(np.int16)   # OpenCV hue range: 0-179
    s = hsv[:, :, 1].astype(np.uint8)
    v = hsv[:, :, 2].astype(np.uint8)

    if target == "magenta":
        # Magenta in OpenCV HSV is around ~150
        target_hue = 150
    elif target == "cyan":
        # Cyan in OpenCV HSV is around ~90
        target_hue = 90
    else:
        raise ValueError("target must be 'magenta' or 'cyan'")

    # Circular hue distance
    hue_diff = np.minimum(np.abs(h - target_hue), 180 - np.abs(h - target_hue))

    mask = (
            (hue_diff <= hue_tolerance) &
            (s >= min_saturation) &
            (v >= min_value)
    )

    mask = (mask * 255).astype(np.uint8)
    mask = _clean_binary_mask(mask)

    return Image.fromarray(mask)


def create_color_mask(
        image,
        mode="hsv",
        target_rgb=(255, 0, 255),
        target_name="magenta",
        tolerance=50,
        hue_tolerance=15,
        min_saturation=60,
        min_value=30,
):
    """
    Unified mask creation entry point.

    mode:
        - "rgb"
        - "hsv"
    """
    if mode == "rgb":
        return create_color_mask_rgb(
            image=image,
            target_rgb=target_rgb,
            tolerance=tolerance,
        )

    if mode == "hsv":
        return create_color_mask_hsv(
            image=image,
            target=target_name,
            hue_tolerance=hue_tolerance,
            min_saturation=min_saturation,
            min_value=min_value,
        )

    raise ValueError("mode must be 'rgb' or 'hsv'")


def clean_mask(mask: Image.Image):
    """
    Clean binary mask to remove noise and fill gaps.
    """
    mask_np = np.array(mask.convert("L"))
    mask_np = (mask_np > 0).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_np, contours, -1, 255, thickness=cv2.FILLED)

    return Image.fromarray(mask_np)


def remove_small_components_np(mask_np: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    Remove tiny disconnected mask components while preserving real geometry.
    Input/output: uint8 mask, values 0 or 255.
    """
    binary = (mask_np > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    output = np.zeros_like(binary, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area >= min_area:
            output[labels == label_id] = 1

    return (output * 255).astype(np.uint8)


def clean_context_mask(
        mask: Image.Image,
        open_kernel: int = 3,
        close_kernel: int = 0,
        min_component_area: int = 20,
) -> Image.Image:
    """
    Clean context mask without filling internal holes.

    Important:
    Unlike clean_mask(), this does NOT draw external contours filled.
    That preserves holes, cutouts, slots, and internal openings.
    """
    mask_np = np.array(mask.convert("L"))
    mask_np = (mask_np > 0).astype(np.uint8) * 255

    if open_kernel and open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), np.uint8)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)

    if close_kernel and close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), np.uint8)
        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

    mask_np = remove_small_components_np(
        mask_np,
        min_area=min_component_area,
    )

    return Image.fromarray(mask_np, mode="L")


def overlay_mask(image, mask):
    """
    Overlay mask on image (red highlight for debugging).
    """
    image_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L"))

    if np.sum(mask_np) == 0:
        return image

    overlay = image_np.copy()
    overlay[mask_np > 0] = [255, 0, 0]

    return Image.fromarray(overlay)


def inspect_pixel(image):
    """
    Debug helper: prints center pixel color.
    """
    image_np = np.array(image.convert("RGB"))
    h, w, _ = image_np.shape
    center_pixel = image_np[h // 2, w // 2]
    print("Center pixel RGB:", center_pixel)


def get_unique_colors(image, limit=20):
    """
    Debug helper: prints a sample of unique colors.
    """
    image_np = np.array(image.convert("RGB"))
    pixels = image_np.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)

    print(f"Showing first {limit} unique colors:")
    print(unique_colors[:limit])


def _detect_dark_internal_openings(
        image,
        focus_binary: np.ndarray,
        min_void_area: int = 20,
        max_void_area_ratio: float = 0.08,
        darkness_threshold: int = 95,
) -> np.ndarray:
    """
    Detect likely dark enclosed openings inside the focus mask.

    Returns a binary uint8 mask (0/1) of candidate openings.
    """
    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    filled_focus = _fill_binary_holes(focus_binary)

    # Candidate voids are regions inside the filled shape but outside the original mask
    void_candidates = ((filled_focus == 1) & (focus_binary == 0)).astype(np.uint8)

    if np.count_nonzero(void_candidates) == 0:
        return void_candidates

    # Keep only dark candidates
    dark_candidates = ((gray < darkness_threshold) & (void_candidates == 1)).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_candidates, connectivity=8)

    out = np.zeros_like(dark_candidates, dtype=np.uint8)
    focus_area = int(np.count_nonzero(focus_binary))
    max_void_area = max(20, int(focus_area * max_void_area_ratio))

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < min_void_area:
            continue
        if area > max_void_area:
            continue

        component_mask = (labels == label_id).astype(np.uint8)

        # Optional shape sanity: reject extremely thin noisy regions
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        if min(w, h) <= 1:
            continue

        out[component_mask > 0] = 1

    return out


def preserve_internal_openings_in_focus_mask(
        image,
        focus_mask: Image.Image,
        min_void_area: int = 20,
        max_void_area_ratio: float = 0.08,
        darkness_threshold: int = 95,
        final_open_kernel: int = 0,
) -> Image.Image:
    """
    Restore likely internal openings inside the focus mask.

    This is meant for cases where highlighted parts contain eyelets,
    holes, or dark internal voids that were lost during mask extraction.
    """
    focus_binary = (np.array(focus_mask.convert("L")) > 0).astype(np.uint8)

    opening_mask = _detect_dark_internal_openings(
        image=image,
        focus_binary=focus_binary,
        min_void_area=min_void_area,
        max_void_area_ratio=max_void_area_ratio,
        darkness_threshold=darkness_threshold,
    )

    refined = focus_binary.copy()
    refined[opening_mask > 0] = 0

    if final_open_kernel and final_open_kernel > 1:
        kernel = np.ones((final_open_kernel, final_open_kernel), np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

    return Image.fromarray((refined * 255).astype(np.uint8), mode="L")


def _fill_binary_holes(mask_binary: np.ndarray) -> np.ndarray:
    h, w = mask_binary.shape
    mask_255 = (mask_binary * 255).astype(np.uint8)

    flood = mask_255.copy()
    floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)

    border_points = []
    for x in range(w):
        border_points.append((x, 0))
        border_points.append((x, h - 1))
    for y in range(h):
        border_points.append((0, y))
        border_points.append((w - 1, y))

    for px, py in border_points:
        if flood[py, px] == 0:
            cv2.floodFill(flood, floodfill_mask, (px, py), 255)

    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_255, flood_inv)
    return (filled > 0).astype(np.uint8)


def preserve_internal_openings_from_context(
        focus_mask: Image.Image,
        context_mask: Image.Image,
        min_void_area: int = 20,
        max_void_area_ratio: float = 0.08,
        min_context_overlap_ratio: float = 0.35,
) -> Image.Image:
    focus_binary = (np.array(focus_mask.convert("L")) > 0).astype(np.uint8)
    context_binary = (np.array(context_mask.convert("L")) > 0).astype(np.uint8)

    filled_focus = _fill_binary_holes(focus_binary)

    candidate_voids = ((filled_focus == 1) & (focus_binary == 0)).astype(np.uint8)
    if np.count_nonzero(candidate_voids) == 0:
        return focus_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_voids, connectivity=8)

    out_voids = np.zeros_like(candidate_voids, dtype=np.uint8)
    focus_area = int(np.count_nonzero(focus_binary))
    max_void_area = max(20, int(focus_area * max_void_area_ratio))

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_void_area or area > max_void_area:
            continue

        component_mask = (labels == label_id).astype(np.uint8)
        overlap = int(np.count_nonzero((component_mask == 1) & (context_binary == 1)))
        overlap_ratio = overlap / max(1, area)

        if overlap_ratio >= min_context_overlap_ratio:
            out_voids[component_mask > 0] = 1

    refined = focus_binary.copy()
    refined[out_voids > 0] = 0

    return Image.fromarray((refined * 255).astype(np.uint8), mode="L")


def subtract_context_supported_regions_from_focus(
        focus_mask: Image.Image,
        context_mask: Image.Image,
        min_region_area: int = 20,
        max_region_area_ratio: float = 0.08,
        final_open_kernel: int = 0,
) -> Image.Image:
    """
    Remove internal regions from the focus mask where context support overlaps focus.

    This helps in cases where openings/voids are currently swallowed by the focus mask
    instead of being absent and recoverable later.
    """
    focus_binary = (np.array(focus_mask.convert("L")) > 0).astype(np.uint8)
    context_binary = (np.array(context_mask.convert("L")) > 0).astype(np.uint8)

    overlap = ((focus_binary == 1) & (context_binary == 1)).astype(np.uint8)
    if np.count_nonzero(overlap) == 0:
        return focus_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(overlap, connectivity=8)

    out_cut = np.zeros_like(overlap, dtype=np.uint8)
    focus_area = int(np.count_nonzero(focus_binary))
    max_region_area = max(20, int(focus_area * max_region_area_ratio))

    h, w = overlap.shape

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_region_area or area > max_region_area:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        rw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        rh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        region_mask = (labels == label_id).astype(np.uint8)

        # Skip regions touching the image border — these are unlikely to be interior openings
        if x == 0 or y == 0 or (x + rw) >= w or (y + rh) >= h:
            continue

        # Light enclosure check:
        # dilate the region and see if it's mostly surrounded by focus
        kernel = np.ones((3, 3), np.uint8)
        ring = cv2.dilate(region_mask, kernel, iterations=1) - region_mask

        ring_pixels = int(np.count_nonzero(ring))
        if ring_pixels == 0:
            continue

        focus_ring = int(np.count_nonzero((ring == 1) & (focus_binary == 1)))
        enclosure_ratio = focus_ring / ring_pixels

        if enclosure_ratio < 0.45:
            continue

        out_cut[region_mask > 0] = 1

    refined = focus_binary.copy()
    refined[out_cut > 0] = 0

    if final_open_kernel and final_open_kernel > 1:
        kernel = np.ones((final_open_kernel, final_open_kernel), np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

    return Image.fromarray((refined * 255).astype(np.uint8), mode="L")


def subtract_non_focus_supported_regions(
        image: Image.Image,
        focus_mask: Image.Image,
        hue_tolerance: int,
        min_saturation: int,
        min_value: int,
        min_region_area: int = 20,
        max_region_area_ratio: float = 0.08,
        enclosure_ratio_threshold: float = 0.55,
) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    focus_binary = (np.array(focus_mask.convert("L")) > 0).astype(np.uint8)

    # Rebuild a stricter magenta support mask
    # Use slightly stricter thresholds than the main focus mask
    magenta_mask = create_color_mask(
        image=image,
        mode="hsv",
        target_name="magenta",
        hue_tolerance=max(6, int(hue_tolerance * 0.7)),
        min_saturation=min(60, max(min_saturation + 10, min_saturation)),
        min_value=max(min_value, 25),
    )
    magenta_binary = (np.array(magenta_mask.convert("L")) > 0).astype(np.uint8)

    # Candidate cutouts = currently in focus, but not supported by stricter focus evidence
    candidates = ((focus_binary == 1) & (magenta_binary == 0)).astype(np.uint8)

    if np.count_nonzero(candidates) == 0:
        return focus_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)

    out_cut = np.zeros_like(candidates, dtype=np.uint8)
    focus_area = int(np.count_nonzero(focus_binary))
    max_region_area = max(20, int(focus_area * max_region_area_ratio))
    h, w = candidates.shape

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_region_area or area > max_region_area:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        rw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        rh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        region_mask = (labels == label_id).astype(np.uint8)

        # Skip anything touching image border
        if x == 0 or y == 0 or (x + rw) >= w or (y + rh) >= h:
            continue

        kernel = np.ones((3, 3), np.uint8)
        ring = cv2.dilate(region_mask, kernel, iterations=1) - region_mask

        ring_pixels = int(np.count_nonzero(ring))
        if ring_pixels == 0:
            continue

        focus_ring = int(np.count_nonzero((ring == 1) & (focus_binary == 1)))
        enclosure_ratio = focus_ring / ring_pixels

        if enclosure_ratio < enclosure_ratio_threshold:
            continue

        out_cut[region_mask > 0] = 1

    refined = focus_binary.copy()
    refined[out_cut > 0] = 0

    return Image.fromarray((refined * 255).astype(np.uint8), mode="L")


def subtract_focus_overlap_from_context(
        context_mask,
        focus_mask,
        erosion_kernel=3,
):
    kernel = np.ones((erosion_kernel, erosion_kernel), np.uint8)

    if isinstance(focus_mask, Image.Image):
        focus_mask = np.array(focus_mask)

    if isinstance(context_mask, Image.Image):
        context_mask = np.array(context_mask)

    focus_mask = (focus_mask > 0).astype(np.uint8)
    context_mask = (context_mask > 0).astype(np.uint8)

    safe_focus = cv2.erode(focus_mask, kernel)

    refined_context = context_mask.copy()
    refined_context[safe_focus > 0] = 0

    return Image.fromarray((refined_context * 255).astype(np.uint8), mode="L")