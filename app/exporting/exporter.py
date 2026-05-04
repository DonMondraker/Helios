import io
from PIL import Image
import zipfile

from exporting.export_config import EXPORT_PRESETS, EXPORT_BUNDLES


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white_bg, image)
        return image.convert("RGB")

    return image.convert("RGB")


def resize_to_width(image: Image.Image, target_width: int) -> Image.Image:
    width, height = image.size

    if width == target_width:
        return image.copy()

    scale = target_width / width
    new_height = int(round(height * scale))

    return image.resize((target_width, new_height), Image.LANCZOS)


def pad_to_square(image: Image.Image, background=(255, 255, 255)) -> Image.Image:
    width, height = image.size
    size = max(width, height)

    canvas = Image.new("RGB", (size, size), background)
    x = (size - width) // 2
    y = (size - height) // 2
    canvas.paste(image, (x, y))

    return canvas


def make_icon(image: Image.Image, icon_size: int) -> Image.Image:
    image = ensure_rgb(image)
    image = pad_to_square(image, background=(255, 255, 255))
    return image.resize((icon_size, icon_size), Image.LANCZOS)


def prepare_image_for_export(image: Image.Image, preset_key: str) -> tuple[Image.Image, dict]:
    if preset_key not in EXPORT_PRESETS:
        raise ValueError(f"Unknown export preset: {preset_key}")

    preset = EXPORT_PRESETS[preset_key]
    image = ensure_rgb(image)

    if "icon_size" in preset:
        processed = make_icon(image, preset["icon_size"])
    elif "target_width" in preset:
        processed = resize_to_width(image, preset["target_width"])
    else:
        processed = image.copy()

    return processed, preset


def export_image_bytes(image: Image.Image, preset_key: str) -> tuple[bytes, str, str]:
    processed, preset = prepare_image_for_export(image, preset_key)

    output = io.BytesIO()
    fmt = preset["format"]
    dpi = preset.get("dpi")

    if fmt == "PNG":
        processed.save(output, format="PNG")
        mime = "image/png"
        extension = "png"

    elif fmt == "JPEG":
        processed.save(
            output,
            format="JPEG",
            quality=preset.get("quality", 95),
            dpi=dpi,
            optimize=False,
            progressive=False,
            subsampling="4:2:2",
        )
        mime = "image/jpeg"
        extension = "jpg"

    elif fmt == "TIFF":
        save_kwargs = {
            "format": "TIFF",
        }

        if dpi:
            save_kwargs["dpi"] = dpi

        if "compression" in preset:
            save_kwargs["compression"] = preset["compression"]

        processed.save(output, **save_kwargs)
        mime = "image/tiff"
        extension = "tiff"

    else:
        raise ValueError(f"Unsupported export format: {fmt}")

    output.seek(0)
    return output.getvalue(), mime, extension


def build_export_filename(base_name: str, preset_key: str, extension: str) -> str:
    safe_base = base_name.strip().replace(" ", "_") or "illustration"
    return f"{safe_base}_{preset_key}.{extension}"


def get_export_dimensions(image: Image.Image, preset_key: str) -> tuple[int, int]:
    processed, _ = prepare_image_for_export(image, preset_key)
    return processed.size


def export_bundle_zip(
        final_image: Image.Image,
        raw_image: Image.Image,
        project_state_bytes: bytes,
        base_name: str,
        bundle_key: str,
) -> bytes:
    zip_buffer = io.BytesIO()
    bundle = EXPORT_BUNDLES[bundle_key]

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        folder = base_name

        for preset_key in bundle["presets"]:
            file_bytes, mime, extension = export_image_bytes(final_image, preset_key)
            zipf.writestr(
                f"{folder}/{base_name}_{preset_key}.{extension}",
                file_bytes,
            )

        # cad_buffer = io.BytesIO()
        # raw_image.save(cad_buffer, format="PNG")
        # zipf.writestr(
        #     f"{folder}/{base_name}_original_cad.png",
        #     cad_buffer.getvalue(),
        # )

        zipf.writestr(
            f"{folder}/{base_name}_project.json",
            project_state_bytes,
        )

    zip_buffer.seek(0)
    return zip_buffer.getvalue()