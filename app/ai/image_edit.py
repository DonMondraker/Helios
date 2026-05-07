import base64
import io
from PIL import Image
# from openai import OpenAI
from core.prompt_builder import build_prompt
from ai.client import get_openai_client

# client = OpenAI()
client = get_openai_client()


def pil_to_bytes(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def edit_image_with_mask(image, mask, instruction):
    client = OpenAI()

    import base64
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    prompt = build_prompt(instruction)

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024"
    )

    image_base64 = result.data[0].b64_json
    image_data = base64.b64decode(image_base64)

    return Image.open(io.BytesIO(image_data))