import cv2
import numpy as np
from PIL import Image


def pil_to_np(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def np_to_pil(image_np: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))


def draw_callouts(
        image: Image.Image,
        callouts: list[dict],
        radius: int = 18,
        line_thickness: int = 2,
        font_scale: float = 0.65,
        text_thickness: int = 2,
        halo: bool = True,
        halo_color: tuple[int, int, int] = (255, 255, 255),
        halo_extra_thickness: int = 3,
):
    """
    Draw callouts with optional white halo.
    """
    img = pil_to_np(image).copy()

    for callout in callouts:
        label = str(callout["label"]).strip().upper()
        cx = int(callout["circle_x"])
        cy = int(callout["circle_y"])
        ex = int(callout["end_x"])
        ey = int(callout["end_y"])

        # Leader line halo
        if halo:
            cv2.line(
                img,
                (cx, cy),
                (ex, ey),
                halo_color,
                line_thickness + halo_extra_thickness,
                lineType=cv2.LINE_AA,
                )

        # Leader line black
        cv2.line(
            img,
            (cx, cy),
            (ex, ey),
            (0, 0, 0),
            line_thickness,
            lineType=cv2.LINE_AA,
        )

        # Circle halo
        if halo:
            cv2.circle(
                img,
                (cx, cy),
                radius + max(1, halo_extra_thickness // 2),
                halo_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
                )

        # Circle fill and outline
        cv2.circle(
            img,
            (cx, cy),
            radius,
            (255, 255, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        if halo:
            cv2.circle(
                img,
                (cx, cy),
                radius,
                halo_color,
                thickness=line_thickness + halo_extra_thickness,
                lineType=cv2.LINE_AA,
            )

        cv2.circle(
            img,
            (cx, cy),
            radius,
            (0, 0, 0),
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

        # Text
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )

        text_x = int(cx - text_w / 2)
        text_y = int(cy + text_h / 2) - 2

        if halo:
            cv2.putText(
                img,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                halo_color,
                text_thickness + halo_extra_thickness,
                lineType=cv2.LINE_AA,
                )

        cv2.putText(
            img,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            text_thickness,
            lineType=cv2.LINE_AA,
        )

    return np_to_pil(img)