# from rendering.arrows import draw_motion_arrow
# from rendering.callouts import draw_callout


def apply_annotations(image):
    """
    Example annotation pipeline.
    In real use, this will be driven by CAD metadata later.
    """

    img = image

    # Example arrow (you will later replace with CAD-driven coordinates)
    img = draw_motion_arrow(img, (100, 200), (250, 200))

    # Example callout
    img = draw_callout(img, (300, 150), "A")

    return img