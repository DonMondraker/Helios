from PIL import Image, ImageDraw, ImageFont


LINE_HALO_WIDTH = 8
LINE_WIDTH = 3

CALLOUT_HALO_WIDTH = 8
CALLOUT_WIDTH = 3
CALLOUT_RADIUS = 18
CALLOUT_HALO_RADIUS = 22
CALLOUT_FONT_SIZE = 18

INSET_HALO_WIDTH = 8
INSET_BORDER_WIDTH = 2


def draw_react_editor_state(base: Image.Image, state: dict | None) -> Image.Image:
    if not state:
        return base

    image = base.copy().convert("RGB")
    draw = ImageDraw.Draw(image)

    draw_focus_halos(draw, state)
    draw_detail_views(image, draw, state)
    # draw_inset_images(image, draw, state)
    draw_lines(draw, state)
    draw_callouts(draw, state)

    return image


def draw_lines(draw: ImageDraw.ImageDraw, state: dict) -> None:
    for line in state.get("lines", []):
        points = [
            (int(line["x1"]), int(line["y1"])),
            (int(line["x2"]), int(line["y2"])),
        ]

        draw.line(points, fill="white", width=LINE_HALO_WIDTH)
        draw.line(points, fill="black", width=LINE_WIDTH)


def draw_callouts(draw: ImageDraw.ImageDraw, state: dict) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", CALLOUT_FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    for callout in state.get("callouts", []):
        cx = int(callout["circleX"])
        cy = int(callout["circleY"])
        ax = int(callout["anchorX"])
        ay = int(callout["anchorY"])
        label = str(callout.get("label", ""))

        # React: halo line first
        draw.line(
            [(cx, cy), (ax, ay)],
            fill="white",
            width=CALLOUT_HALO_WIDTH,
        )

        # React: black main line
        draw.line(
            [(cx, cy), (ax, ay)],
            fill="black",
            width=CALLOUT_WIDTH,
        )

        # React: white halo circle
        draw.ellipse(
            [
                cx - CALLOUT_HALO_RADIUS,
                cy - CALLOUT_HALO_RADIUS,
                cx + CALLOUT_HALO_RADIUS,
                cy + CALLOUT_HALO_RADIUS,
                ],
            fill="white",
        )

        # React: white circle with black stroke
        draw.ellipse(
            [
                cx - CALLOUT_RADIUS,
                cy - CALLOUT_RADIUS,
                cx + CALLOUT_RADIUS,
                cy + CALLOUT_RADIUS,
                ],
            fill="white",
            outline="black",
            width=2,
        )

        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.text(
            (cx - tw / 2, cy - th / 2 - 1),
            label,
            fill="black",
            font=font,
        )


def draw_focus_halos(draw: ImageDraw.ImageDraw, state: dict) -> None:
    for obj in state.get("focusObjects", []):
        if not obj.get("haloEnabled"):
            continue

        polygon = obj.get("polygon", [])
        if len(polygon) < 3:
            continue

        points = [(int(p["x"]), int(p["y"])) for p in polygon]

        draw.line(points + [points[0]], fill="white", width=3)
        draw.line(points + [points[0]], fill="white", width=2)


def draw_detail_views(image: Image.Image, draw: ImageDraw.ImageDraw, state: dict) -> None:
    for inset in state.get("detailViews", []):
        x = int(inset["x"])
        y = int(inset["y"])
        w = int(inset["width"])
        h = int(inset["height"])

        sx = int(inset["sourceX"])
        sy = int(inset["sourceY"])
        sw = int(inset["sourceWidth"])
        sh = int(inset["sourceHeight"])

        if inset.get("showLeader", True):
            anchor_x = int(inset["leaderAnchorX"])
            anchor_y = int(inset["leaderAnchorY"])
            center_x = x + w // 2
            center_y = y + h // 2

            draw.line(
                [(anchor_x, anchor_y), (center_x, center_y)],
                fill="white",
                width=LINE_HALO_WIDTH,
            )
            draw.line(
                [(anchor_x, anchor_y), (center_x, center_y)],
                fill="black",
                width=LINE_WIDTH,
            )

        crop = image.crop((sx, sy, sx + sw, sy + sh))
        crop = crop.resize((w, h))
        image.paste(crop, (x, y))

        draw.rectangle(
            [x, y, x + w, y + h],
            outline="white",
            width=INSET_HALO_WIDTH,
        )
        draw.rectangle(
            [x, y, x + w, y + h],
            outline="black",
            width=INSET_BORDER_WIDTH,
        )