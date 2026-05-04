from PIL import Image
import cv2
import numpy as np


def _fit_image_to_box(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height), Image.LANCZOS)


def _leader_start_from_box(placement: dict, target_end: list[int]) -> list[int]:
    x = int(placement["x"])
    y = int(placement["y"])
    w = int(placement["width"])
    h = int(placement["height"])

    candidates = [
        [x, y + h // 2],
        [x + w, y + h // 2],
        [x + w // 2, y],
        [x + w // 2, y + h],
    ]

    tx, ty = int(target_end[0]), int(target_end[1])
    return min(candidates, key=lambda p: (p[0] - tx) ** 2 + (p[1] - ty) ** 2)


def _pil_to_np(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def _np_to_pil(image_np: np.ndarray) -> Image.Image:
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)


def _draw_halo_line_cv2(
        img: np.ndarray,
        start: list[int],
        end: list[int],
        line_width: int = 3,
        halo_extra: int = 4,
):
    x1, y1 = int(start[0]), int(start[1])
    x2, y2 = int(end[0]), int(end[1])

    cv2.line(
        img,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        line_width + halo_extra,
        lineType=cv2.LINE_AA,
        )

    cv2.line(
        img,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        line_width,
        lineType=cv2.LINE_AA,
    )


def _draw_halo_rectangle_cv2(
        img: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        border_width: int = 3,
        halo_extra: int = 4,
):
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (255, 255, 255),
        border_width + halo_extra,
        lineType=cv2.LINE_AA,
        )

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 0, 0),
        border_width,
        lineType=cv2.LINE_AA,
    )


def _draw_label_with_halo_cv2(
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        font_scale: float = 0.7,
        text_thickness: int = 2,
):
    cv2.putText(
        img,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness + 3,
        lineType=cv2.LINE_AA,
        )

    cv2.putText(
        img,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        text_thickness,
        lineType=cv2.LINE_AA,
    )


from PIL import Image
import cv2
import numpy as np


def _fit_image_to_box(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height), Image.LANCZOS)


def _leader_start_from_box(placement: dict, target_end: list[int]) -> list[int]:
    x = int(placement["x"])
    y = int(placement["y"])
    w = int(placement["width"])
    h = int(placement["height"])

    candidates = [
        [x, y + h // 2],
        [x + w, y + h // 2],
        [x + w // 2, y],
        [x + w // 2, y + h],
    ]

    tx, ty = int(target_end[0]), int(target_end[1])
    return min(candidates, key=lambda p: (p[0] - tx) ** 2 + (p[1] - ty) ** 2)


def _pil_to_np(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def _np_to_pil(image_np: np.ndarray) -> Image.Image:
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)


def _draw_halo_line_cv2(
        img: np.ndarray,
        start: list[int],
        end: list[int],
        line_width: int = 2,
        halo_extra: int = 3,
):
    x1, y1 = int(start[0]), int(start[1])
    x2, y2 = int(end[0]), int(end[1])

    cv2.line(
        img,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        line_width + halo_extra,
        lineType=cv2.LINE_AA,
        )

    cv2.line(
        img,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        line_width,
        lineType=cv2.LINE_AA,
    )


def _draw_halo_rectangle_cv2(
        img: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        border_width: int = 4,
        halo_extra: int = 4,
):
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (255, 255, 255),
        border_width + halo_extra,
        lineType=cv2.LINE_AA,
        )

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 0, 0),
        border_width,
        lineType=cv2.LINE_AA,
    )


def _draw_label_with_halo_cv2(
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        font_scale: float = 0.7,
        text_thickness: int = 2,
):
    cv2.putText(
        img,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness + 3,
        lineType=cv2.LINE_AA,
        )

    cv2.putText(
        img,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        text_thickness,
        lineType=cv2.LINE_AA,
    )


def draw_insets_on_image(
        image: Image.Image,
        insets: list[dict],
        inset_assets: dict,
) -> Image.Image:
    output = image.convert("RGB").copy()
    img = _pil_to_np(output)

    img_h, img_w = img.shape[:2]

    for inset in insets:
        if not inset.get("visible", True):
            continue

        asset = inset_assets.get(inset["asset_id"])
        if not asset:
            continue

        rendered = asset.get("rendered_image")
        if rendered is None:
            continue

        placement = inset["placement"]
        x = int(placement["x"])
        y = int(placement["y"])
        w = int(placement["width"])
        h = int(placement["height"])

        if w <= 0 or h <= 0:
            continue

        fitted = _fit_image_to_box(rendered.convert("RGB"), w, h)
        fitted_np = _pil_to_np(fitted)

        # Clip inset box to image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        # Skip if completely outside image
        if x1 >= x2 or y1 >= y2:
            continue

        # Corresponding crop region inside inset image
        src_x1 = x1 - x
        src_y1 = y1 - y
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)

        img[y1:y2, x1:x2] = fitted_np[src_y1:src_y2, src_x1:src_x2]

        border = inset.get("border_style", {})
        border_enabled = border.get("enabled", True)
        border_width = int(border.get("width", 3))
        halo_enabled = bool(border.get("halo", True))

        if border_enabled:
            if halo_enabled:
                _draw_halo_rectangle_cv2(
                    img=img,
                    x=x1,
                    y=y1,
                    w=max(1, x2 - x1),
                    h=max(1, y2 - y1),
                    border_width=2,
                    halo_extra=3,
                )
            else:
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 0),
                    border_width,
                    lineType=cv2.LINE_AA,
                )

        leader = inset.get("leader", {})
        if leader.get("enabled", True):
            end = leader.get("end", [x - 40, y + h // 2])
            start = _leader_start_from_box(
                {
                    "x": x1,
                    "y": y1,
                    "width": max(1, x2 - x1),
                    "height": max(1, y2 - y1),
                },
                end,
            )

            if leader.get("halo", True):
                _draw_halo_line_cv2(
                    img=img,
                    start=start,
                    end=end,
                    line_width=2,
                    halo_extra=3,
                )
            else:
                cv2.line(
                    img,
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                    (0, 0, 0),
                    3,
                    lineType=cv2.LINE_AA,
                )

        label = inset.get("label", {})
        if label.get("enabled") and label.get("text"):
            lx, ly = label.get("position", [x, y - 20])
            _draw_label_with_halo_cv2(
                img=img,
                text=str(label["text"]),
                x=int(lx),
                y=int(ly),
                font_scale=0.7,
                text_thickness=2,
            )

    return _np_to_pil(img)