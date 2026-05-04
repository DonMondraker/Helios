EXPORT_PRESETS = {
    "regular_png": {
        "label": "Regular PNG",
        "format": "PNG",
    },
    "regular_jpg": {
        "label": "Regular JPG",
        "format": "JPEG",
        "dpi": (300, 300),
        "quality": 95,
    },

    # GRIPS JPEG
    "grips_jpeg_600": {
        "label": "GRIPS JPEG - 600 px",
        "format": "JPEG",
        "target_width": 600,
        "dpi": (206, 206),
        "quality": 85,  # practical Pillow approximation
    },
    "grips_jpeg_800": {
        "label": "GRIPS JPEG - 800 px",
        "format": "JPEG",
        "target_width": 800,
        "dpi": (206, 206),
        "quality": 85,
    },
    "grips_jpeg_icon": {
        "label": "GRIPS JPEG - 140 x 140 icon",
        "format": "JPEG",
        "icon_size": 140,
        "dpi": (206, 206),
        "quality": 85,
    },

    # GRIPS TIFF
    "grips_tiff_600": {
        "label": "GRIPS TIFF - 600 px",
        "format": "TIFF",
        "target_width": 600,
        "dpi": (206, 206),
        "compression": "tiff_lzw",
    },
    "grips_tiff_800": {
        "label": "GRIPS TIFF - 800 px",
        "format": "TIFF",
        "target_width": 800,
        "dpi": (206, 206),
        "compression": "tiff_lzw",
    },
    "grips_tiff_icon": {
        "label": "GRIPS TIFF - 140 x 140 icon",
        "format": "TIFF",
        "icon_size": 140,
        "dpi": (206, 206),
        "compression": "tiff_lzw",
    },

    # SID JPEG = same as GRIPS JPEG
    "sid_jpeg_600": {
        "label": "SID JPEG - 600 px",
        "format": "JPEG",
        "target_width": 600,
        "dpi": (206, 206),
        "quality": 85,
    },
    "sid_jpeg_800": {
        "label": "SID JPEG - 800 px",
        "format": "JPEG",
        "target_width": 800,
        "dpi": (206, 206),
        "quality": 85,
    },
    "sid_jpeg_icon": {
        "label": "SID JPEG - 140 x 140 icon",
        "format": "JPEG",
        "icon_size": 140,
        "dpi": (206, 206),
        "quality": 85,
    },

    # SID TIFF
    "sid_tiff_1538": {
        "label": "SID TIFF - 1538 px",
        "format": "TIFF",
        "target_width": 1538,
        "dpi": (300, 300),
        "compression": "tiff_lzw",
    },
}

EXPORT_BUNDLES = {
    "GRIPS": {
        "label": "GRIPS Export Package",
        "presets": [
            "grips_jpeg_600",
            # "grips_jpeg_800",
            # "grips_jpeg_icon",
            "grips_tiff_600",
            # "grips_tiff_800",
            # "grips_tiff_icon",
        ],
    },
    "SID": {
        "label": "SID Export Package",
        "presets": [
            "sid_jpeg_600",
            # "sid_jpeg_800",
            # "sid_jpeg_icon",
            "sid_tiff_1538",
        ],
    },
}