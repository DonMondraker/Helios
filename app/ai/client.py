from openai import OpenAI


def get_client():
    """
    Create and return OpenAI client.
    Requires OPENAI_API_KEY in environment variables.
    """
    return OpenAI()