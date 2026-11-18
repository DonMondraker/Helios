from datetime import datetime, timezone
from uuid import uuid4
from copy import deepcopy
import json
import base64
import io
from PIL import Image


PROJECT_VERSION = "2.0"


def create_empty_project(name: str = "Untitled Project") -> dict:
    return {
        "project_version": PROJECT_VERSION,
        "project_id": str(uuid4()),
        "project_name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "illustrations": [],
    }
    
    
def add_illustration(
    project: dict,
    name: str,
    illustration_state: dict,
) -> str:
    """
    Add an illustration to a Helios project.

    Returns the unique ID of the new illustration.
    """
    illustration_id = str(uuid4())

    illustration = {
        "illustration_id": illustration_id,
        "name": name,
        "state": deepcopy(illustration_state),
    }

    project["illustrations"].append(illustration)

    return illustration_id
    
    
def get_illustration(
    project: dict,
    illustration_id: str,
) -> dict | None:
    """
    Find an illustration by its unique ID.

    Returns the illustration dictionary,
    or None if the ID does not exist.
    """
    for illustration in project["illustrations"]:
        if illustration["illustration_id"] == illustration_id:
            return illustration

    return None
    
    
def remove_illustration(
    project: dict,
    illustration_id: str,
) -> bool:
    """
    Remove an illustration from the project.

    Returns True if an illustration was removed,
    or False if the ID was not found.
    """
    for index, illustration in enumerate(project["illustrations"]):
        if illustration["illustration_id"] == illustration_id:
            project["illustrations"].pop(index)
            return True

    return False


def update_illustration_state(
    project: dict,
    illustration_id: str,
    new_state: dict,
) -> bool:
    """
    Replace an illustration's state with new data.

    Returns True if the illustration was updated,
    or False if the ID was not found.
    """
    illustration = get_illustration(project, illustration_id)

    if illustration is None:
        return False

    illustration["state"] = deepcopy(new_state)

    return True
    
    
def serialize_project(project: dict) -> bytes:
    """
    Convert a Helios 2.0 project into UTF-8 JSON bytes.

    Raises TypeError if the project contains data
    that cannot be represented as JSON.
    """
    return json.dumps(
        project,
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")
    
    
def deserialize_project(data: bytes) -> dict:
    """
    Load a Helios 2.0 project from JSON bytes.

    Raises ValueError when the project file
    is invalid or unsupported.
    """
    try:
        project = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "The selected file is not valid JSON."
        ) from exc

    if not isinstance(project, dict):
        raise ValueError(
            "Invalid Helios project: expected a JSON object."
        )

    if project.get("project_version") != PROJECT_VERSION:
        raise ValueError(
            "Unsupported Helios project version."
        )

    if not isinstance(project.get("illustrations"), list):
        raise ValueError(
            "Invalid Helios project: illustrations must be a list."
        )


    # Every illustration must have a valid, unique identifier.
    seen_illustration_ids = set()

    for illustration in project["illustrations"]:
        if not isinstance(illustration, dict):
            raise ValueError(
                "Invalid Helios project: illustration must be an object."
            )

        illustration_id = illustration.get("illustration_id")

        if not isinstance(illustration_id, str) or not illustration_id.strip():
            raise ValueError(
                "Invalid Helios project: illustration ID is missing."
            )

        if illustration_id in seen_illustration_ids:
            raise ValueError(
                "Invalid Helios project: duplicate illustration ID."
            )

        seen_illustration_ids.add(illustration_id)


        illustration_name = illustration.get("name")

        if not isinstance(illustration_name, str) or not illustration_name.strip():
            raise ValueError(
                "Invalid Helios project: illustration name is missing."
            )

        if not isinstance(illustration.get("state"), dict):
            raise ValueError(
                "Invalid Helios project: illustration state must be an object."
            )



    return project
    
    
def encode_image(image: Image.Image | None) -> str | None:
    """
    Convert a PIL image into a base64-encoded PNG string.
    """
    if image is None:
        return None

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def decode_image(
    encoded: str | None,
    mode: str | None = None,
) -> Image.Image | None:
    """
    Restore a PIL image from a base64-encoded PNG string.
    """
    if encoded is None:
        return None

    raw = base64.b64decode(encoded, validate=True)

    with Image.open(io.BytesIO(raw)) as image:
        image.load()

        if mode is not None:
            return image.convert(mode)

        return image.copy()

def update_illustration_fields(
        project: dict,
        illustration_id: str,
        fields: dict,
) -> bool:
    """
    Update selected top-level fields of an illustration's state
    without replacing the entire state.
    """
    illustration = get_illustration(project, illustration_id)

    if illustration is None:
        return False

    for key, value in fields.items():
        illustration["state"][key] = deepcopy(value)

    return True
