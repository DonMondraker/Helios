import streamlit as st
from PIL import Image

def show_image_preview(uploaded_file):
    """
    Displays uploaded image preview.

    Args:
        uploaded_file: Streamlit uploaded file object
    """

    if uploaded_file is None:
        return None

    # Convert uploaded file to PIL Image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image in UI
    st.image(
        image,
        caption="Uploaded CAD Image",
        use_container_width=True
    )

    return image