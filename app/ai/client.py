import os
import streamlit as st
from openai import OpenAI


def get_openai_client() -> OpenAI:
    api_key = st.secrets.get(
        "OPENAI_API_KEY",
        os.getenv("OPENAI_API_KEY"),
    )

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured")

    return OpenAI(api_key=api_key)

# from openai import OpenAI
#
#
# def get_client():
#     """
#     Create and return OpenAI client.
#     Requires OPENAI_API_KEY in environment variables.
#     """
#     return OpenAI()