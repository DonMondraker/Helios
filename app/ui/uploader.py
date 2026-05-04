import streamlit as st

def upload_image():
    """
    Handles image upload from the user.

    Returns:
        uploaded_file (UploadedFile or None)
    """

    uploaded_file = st.file_uploader(
        label="Upload CAD Screenshot",
        type=["png", "jpg", "jpeg"],
        help="Upload a screenshot exported from your CAD system"
    )

    return uploaded_file