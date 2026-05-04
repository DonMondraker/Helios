def get_illustration_rules():
    """
    Centralized illustration constraints for all AI prompts.
    """

    return {
        "style": [
            "3D shaded technical illustration",
            "clean engineering visualization",
            "no photography",
            "no realistic textures"
        ],

        "geometry_rules": [
            "keep original structure intact",
            "do not invent new components",
            "do not distort CAD proportions"
        ],

        "removal_logic": [
            "removed components must disappear cleanly",
            "no partial ghost parts unless required",
            "use clean cut removal"
        ],

        "color_system": [
            "main focus RGB(255,160,70)",
            "secondary RGB(90,90,90)",
            "light grey RGB(180,180,180)",
            "background white"
        ],

        "prohibited": [
            "text labels",
            "brand logos",
            "part numbers",
            "decorative symbols"
        ],

        "clarity_rules": [
            "minimal background",
            "high readability",
            "simple composition",
            "avoid visual clutter"
        ]
    }