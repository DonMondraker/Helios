from PIL import ImageDraw, Image
import base64
from io import BytesIO

def draw_focus_halos(image, react_state):
    focus_objects = react_state.get("focusObjects", [])
    draw = ImageDraw.Draw(image)

    for obj in focus_objects:
        if not obj.get("haloEnabled"):
            continue

        polygon = obj.get("polygon", [])
        if len(polygon) < 3:
            continue

        points = [(p["x"], p["y"]) for p in polygon]

        # Outer soft halo
        draw.line(points + [points[0]], fill=(255, 255, 255), width=6)

        # Inner sharper edge
        draw.line(points + [points[0]], fill=(255, 255, 255), width=3)

    return image


def draw_react_inset_images(image, react_state):
    inset_images = react_state.get("insetImages", [])
    draw = ImageDraw.Draw(image)

    for inset in inset_images:
        image_src = inset.get("imageSrc")
        if not image_src or not image_src.startswith("data:image"):
            continue

        # Decode base64
        header, encoded = image_src.split(",", 1)
        inset_img = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")

        # Draw border (halo + stroke)
        x, y = int(inset["x"]), int(inset["y"])
        w, h = int(inset["width"]), int(inset["height"])

        # Leader line
        if inset.get("showLeader"):
            cx = x + w // 2
            cy = y + h // 2

            lx = int(inset.get("leaderAnchorX", cx))
            ly = int(inset.get("leaderAnchorY", cy))

            # Halo
            draw.line([lx, ly, cx, cy], fill=(255,255,255), width=6)

            # Main line
            draw.line([lx, ly, cx, cy], fill=(0,0,0), width=2)

        # Resize
        inset_img = inset_img.resize(
            (int(inset["width"]), int(inset["height"]))
        )

        # Paste
        image.paste(inset_img, (int(inset["x"]), int(inset["y"])))

        # Halo border
        draw.rectangle([x, y, x + w, y + h], outline=(255,255,255), width=6)

        # Main border
        draw.rectangle([x, y, x + w, y + h], outline=(0,0,0), width=2)

    return image