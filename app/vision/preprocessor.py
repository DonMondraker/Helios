import cv2
import numpy as np
from PIL import Image


def normalize_creoview_image(
        image: Image.Image,
        bilateral_d: int = 7,
        bilateral_sigma_color: int = 40,
        bilateral_sigma_space: int = 40,
        highlight_clip_percentile: float = 98.5,
        saturation_boost: float = 1.2,
        value_gamma: float = 0.9,
) -> Image.Image:
    """
    Normalize Creo View screenshots to reduce reflections and improve mask extraction.
    """
    img = np.array(image.convert("RGB")).astype(np.uint8)

    # 1) Edge-preserving smoothing
    img = cv2.bilateralFilter(
        img,
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    )

    # 2) Convert to HSV for value/saturation control
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # 3) Clip extreme highlights
    clip_value = np.percentile(v, highlight_clip_percentile)
    v = np.minimum(v, clip_value)

    # 4) Normalize clipped value channel back to 0-255
    if v.max() > v.min():
        v = (v - v.min()) / (v.max() - v.min()) * 255.0

    # 5) Mild gamma compression for bright regions
    v = 255.0 * np.power(v / 255.0, value_gamma)

    # 6) Slight saturation boost to help magenta/cyan survive
    s = np.clip(s * saturation_boost, 0, 255)

    hsv_norm = cv2.merge([h, s, v]).astype(np.uint8)
    rgb_norm = cv2.cvtColor(hsv_norm, cv2.COLOR_HSV2RGB)

    return Image.fromarray(rgb_norm)