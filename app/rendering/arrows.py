import cv2
import numpy as np


ARROW_COLOR = (0, 0, 0)  # RGB(0,0,0)


def draw_motion_arrow(image, start, end, thickness=2):
    """
    Draws a motion/rotation arrow (allowed type in your rules).
    """

    img = np.array(image).copy()

    cv2.arrowedLine(
        img,
        start,
        end,
        ARROW_COLOR,
        thickness,
        tipLength=0.05
    )

    return img