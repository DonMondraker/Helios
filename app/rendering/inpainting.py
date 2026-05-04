import cv2
import numpy as np
from PIL import Image


def inpaint_image(image, mask):
    """
    Removes objects using OpenCV inpainting (smooth fill).
    """

    image_np = np.array(image)
    mask_np = np.array(mask)

    # OpenCV expects single-channel mask
    mask_np = (mask_np > 0).astype(np.uint8) * 255

    # Inpainting (THIS is the key improvement)
    inpainted = cv2.inpaint(
        image_np,
        mask_np,
        inpaintRadius=7,
        flags=cv2.INPAINT_TELEA
    )

    return Image.fromarray(inpainted)