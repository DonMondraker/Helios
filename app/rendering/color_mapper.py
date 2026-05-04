import numpy as np
import cv2
from PIL import Image


MAIN_FOCUS = np.array([255, 160, 70], dtype=np.float32)
DARK_GREY = np.array([90, 90, 90], dtype=np.float32)
LIGHT_GREY = np.array([180, 180, 180], dtype=np.float32)
WHITE = np.array([255, 255, 255], dtype=np.float32)


def pil_to_np(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB")).astype(np.float32)


def np_to_pil(image_np: np.ndarray) -> Image.Image:
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)


def compute_luminance(image_np: np.ndarray) -> np.ndarray:
    return (
            0.299 * image_np[:, :, 0]
            + 0.587 * image_np[:, :, 1]
            + 0.114 * image_np[:, :, 2]
    )


def compute_shading_map(
        luminance: np.ndarray,
        gamma: float = 0.82,
) -> np.ndarray:
    """
    Build a compressed-but-not-flat shading map from luminance.

    Lower gamma (<1) preserves more shadow depth.
    Output is normalized to 0..1.
    """
    lum = luminance.astype(np.float32)

    lum_min = np.min(lum)
    lum_max = np.max(lum)

    if lum_max - lum_min < 1e-5:
        return np.ones_like(lum, dtype=np.float32)

    norm = (lum - lum_min) / (lum_max - lum_min)
    norm = np.clip(norm, 0.0, 1.0)

    shaded = np.power(norm, gamma)
    return shaded


def create_object_mask(
        image_np: np.ndarray,
        background_threshold: int = 245
) -> np.ndarray:
    """
    Detect visible object region using border-connected background logic.

    Background is defined as regions that are:
    - bright enough
    - low in color variation
    - connected to the image border

    This helps avoid classifying internal reflections as background.
    """
    luminance = (
            0.299 * image_np[:, :, 0]
            + 0.587 * image_np[:, :, 1]
            + 0.114 * image_np[:, :, 2]
    )

    bright_mask = luminance > background_threshold
    color_var = np.std(image_np, axis=2)
    low_variance = color_var < 10

    candidate_background = bright_mask & low_variance
    candidate_u8 = candidate_background.astype(np.uint8)

    h, w = candidate_u8.shape
    visited = np.zeros_like(candidate_u8, dtype=bool)
    background = np.zeros_like(candidate_u8, dtype=bool)

    from collections import deque
    queue = deque()

    for x in range(w):
        if candidate_u8[0, x]:
            queue.append((0, x))
        if candidate_u8[h - 1, x]:
            queue.append((h - 1, x))

    for y in range(h):
        if candidate_u8[y, 0]:
            queue.append((y, 0))
        if candidate_u8[y, w - 1]:
            queue.append((y, w - 1))

    while queue:
        y, x = queue.popleft()

        if visited[y, x]:
            continue
        visited[y, x] = True

        if not candidate_u8[y, x]:
            continue

        background[y, x] = True

        if y > 0 and not visited[y - 1, x]:
            queue.append((y - 1, x))
        if y < h - 1 and not visited[y + 1, x]:
            queue.append((y + 1, x))
        if x > 0 and not visited[y, x - 1]:
            queue.append((y, x - 1))
        if x < w - 1 and not visited[y, x + 1]:
            queue.append((y, x + 1))

    object_mask = ~background

    # Small silhouette cleanup
    kernel = np.ones((3, 3), np.uint8)
    object_mask_u8 = object_mask.astype(np.uint8) * 255
    object_mask_u8 = cv2.morphologyEx(object_mask_u8, cv2.MORPH_CLOSE, kernel)
    object_mask = object_mask_u8 > 0

    return object_mask


def enhance_focus_edges(
        base_result: np.ndarray,
        original_image: np.ndarray,
        focus_mask: np.ndarray,
        edge_strength: float = 0.18
) -> np.ndarray:
    """
    Add edge emphasis only inside the focus region.
    """
    focus_mask = focus_mask.astype(bool)
    result = base_result.copy()

    gray = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140).astype(np.float32) / 255.0

    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(
        (edges * 255).astype(np.uint8),
        kernel,
        iterations=1
    ).astype(np.float32) / 255.0

    focus_edges = edges * focus_mask.astype(np.float32)

    for c in range(3):
        channel = result[:, :, c]
        channel[focus_mask] = channel[focus_mask] * (
                1.0 - edge_strength * focus_edges[focus_mask]
        )
        result[:, :, c] = channel

    return result


def create_silhouette_mask(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """
    Create a silhouette/outline mask from a binary object mask.
    """
    mask_u8 = (mask.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    outline = np.zeros_like(mask_u8)
    cv2.drawContours(outline, contours, -1, 255, thickness=thickness)

    return outline.astype(np.float32) / 255.0


def apply_outline_layer(
        image_np: np.ndarray,
        outline_mask: np.ndarray,
        strength: float = 0.24
) -> np.ndarray:
    result = image_np.copy()

    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1.0 - strength * outline_mask)

    return result


def enhance_contours(
        base_result: np.ndarray,
        focus_mask: np.ndarray,
        object_mask: np.ndarray,
        focus_outline_strength: float = 0.28,
        object_outline_strength: float = 0.14,
        focus_outline_thickness: int = 2,
        object_outline_thickness: int = 1
) -> np.ndarray:
    result = base_result.copy()

    focus_outline = create_silhouette_mask(
        focus_mask.astype(bool),
        thickness=focus_outline_thickness
    )

    object_outline = create_silhouette_mask(
        object_mask.astype(bool),
        thickness=object_outline_thickness
    )

    result = apply_outline_layer(
        result,
        object_outline,
        strength=object_outline_strength
    )

    result = apply_outline_layer(
        result,
        focus_outline,
        strength=focus_outline_strength
    )

    return result


def apply_region_shading(
        base_color: np.ndarray,
        shading_map: np.ndarray,
        min_factor: float,
        max_factor: float,
) -> np.ndarray:
    """
    Convert shading_map (0..1) into a multiplier range.
    Lower min_factor = deeper shadows.
    """
    factor = min_factor + shading_map * (max_factor - min_factor)
    return base_color.reshape(1, 1, 3) * factor[:, :, None]


def apply_palette_with_shading(
        image_np: np.ndarray,
        focus_mask: np.ndarray,
        context_mask: np.ndarray,
        object_mask: np.ndarray,
        enhance_focus: bool = True,
        enhance_contour: bool = True
) -> np.ndarray:
    """
    Apply final palette:
    - focus_mask -> orange
    - context_mask -> dark grey
    - remaining object -> light grey
    - background -> white

    With stronger 3D-shaded feel than earlier flatter versions.
    """
    result = np.ones_like(image_np, dtype=np.float32) * WHITE

    luminance = compute_luminance(image_np)
    shading_map = compute_shading_map(luminance, gamma=0.82)

    focus_mask = focus_mask.astype(bool)
    context_mask = context_mask.astype(bool)
    object_mask = object_mask.astype(bool)

    context_mask = context_mask & (~focus_mask)
    remaining_object_mask = object_mask & (~focus_mask) & (~context_mask)
    background_mask = ~object_mask

    # Region-specific shading strength
    focus_shaded = apply_region_shading(
        base_color=MAIN_FOCUS,
        shading_map=shading_map,
        min_factor=0.55, # Focus region shading
        max_factor=1.05,
    )

    context_shaded = apply_region_shading(
        base_color=DARK_GREY,
        shading_map=shading_map,
        min_factor=0.55, # Context region shading
        max_factor=1.00,
    )

    remaining_shaded = apply_region_shading(
        base_color=LIGHT_GREY,
        shading_map=shading_map,
        min_factor=0.55, # remaining region shading
        max_factor=1.02,
    )

    for c in range(3):
        result[:, :, c][focus_mask] = focus_shaded[:, :, c][focus_mask]
        result[:, :, c][context_mask] = context_shaded[:, :, c][context_mask]
        result[:, :, c][remaining_object_mask] = remaining_shaded[:, :, c][remaining_object_mask]
        result[:, :, c][background_mask] = WHITE[c]

    if enhance_focus:
        result = enhance_focus_edges(
            base_result=result,
            original_image=image_np,
            focus_mask=focus_mask,
            edge_strength=0.18
        )

    if enhance_contour:
        result = enhance_contours(
            base_result=result,
            focus_mask=focus_mask,
            object_mask=object_mask,
            focus_outline_strength=0.28,
            object_outline_strength=0.14,
            focus_outline_thickness=2,
            object_outline_thickness=1
        )

    return result


def render_standardized_illustration(
        image: Image.Image,
        focus_mask: Image.Image,
        context_mask: Image.Image,
        background_threshold: int = 245,
        enhance_focus: bool = True,
        enhance_contour: bool = True
) -> Image.Image:
    image_np = pil_to_np(image)

    focus_mask_np = np.array(focus_mask.convert("L")) > 0
    context_mask_np = np.array(context_mask.convert("L")) > 0

    object_mask = create_object_mask(
        image_np,
        background_threshold=background_threshold
    )

    result_np = apply_palette_with_shading(
        image_np=image_np,
        focus_mask=focus_mask_np,
        context_mask=context_mask_np,
        object_mask=object_mask,
        enhance_focus=enhance_focus,
        enhance_contour=enhance_contour
    )

    return np_to_pil(result_np)