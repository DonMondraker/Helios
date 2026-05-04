import numpy as np
import cv2
from PIL import Image


def remove_with_mask(image, mask):
    """
    Removes masked area by filling with background color.
    """

    image_np = np.array(image)
    mask_np = np.array(mask)

    output = image_np.copy()

    # Create white background where mask is active
    output[mask_np > 0] = [255, 255, 255]

    return Image.fromarray(output)


def overlay_mask_debug(image, mask):
    """
    Debug overlay (red highlight)
    """

    image_np = np.array(image)
    mask_np = np.array(mask)

    overlay = image_np.copy()

    overlay[mask_np > 0] = [255, 0, 0]

    return Image.fromarray(overlay)