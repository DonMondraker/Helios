
import io
import zipfile

from project_state.project_model import (
    serialize_project,
    deserialize_project,
)


def save_project_file(project: dict) -> bytes:
    """Package a Helios project into a .helios archive."""
    project_json = serialize_project(project)

    buffer = io.BytesIO()

    with zipfile.ZipFile(
            buffer,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("project.json", project_json)

    return buffer.getvalue()


def load_project_file(data: bytes) -> dict:
    """Restore a Helios project from a .helios archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
            project_json = archive.read("project.json")

    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise ValueError(
            "Invalid Helios project file."
        ) from exc

    return deserialize_project(project_json)
