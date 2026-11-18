
from app.project_state.project_model import (
    create_empty_project,
    add_illustration,
    encode_image,
    decode_image,
    get_illustration,
)
from app.project_state.project_file import (
    save_project_file,
    load_project_file,
)

from PIL import Image
import io
import zipfile


def test_native_project_round_trip():
    project = create_empty_project("Helios Native Test")

    source_image = Image.new(
        "RGB", (40, 30), color=(200, 100, 50)
    )
    focus_mask = Image.new(
        "L", (40, 30), color=0
    )
    focus_mask.putpixel((10, 10), 255)

    id_a = add_illustration(
        project,
        "Illustration A",
        {
            "source": {
                "filename": "source_a.png",
                "image_b64": encode_image(source_image),
            },
            "masks": {
                "focus_mask_b64": encode_image(focus_mask),
                "context_mask_b64": None,
            },
            "annotations": {
                "manual_lines": [{"id": "line-a"}],
            },
        },
    )

    id_b = add_illustration(
        project,
        "Illustration B",
        {
            "source": {
                "filename": "source_b.png",
                "image_b64": encode_image(
                    Image.new("RGB", (20, 20), color="white")
                ),
            },
            "annotations": {
                "manual_lines": [],
            },
        },
    )

    # Save the complete project
    project_bytes = save_project_file(project)

    assert isinstance(project_bytes, bytes)

    # Verify the native file is a ZIP archive
    with zipfile.ZipFile(io.BytesIO(project_bytes)) as archive:
        assert "project.json" in archive.namelist()

    # Load the saved project
    loaded_project = load_project_file(project_bytes)

    assert loaded_project["project_id"] == project["project_id"]
    assert loaded_project["project_name"] == "Helios Native Test"
    assert len(loaded_project["illustrations"]) == 2

    # Verify illustration A
    loaded_a = get_illustration(loaded_project, id_a)
    assert loaded_a["name"] == "Illustration A"
    assert loaded_a["state"]["annotations"]["manual_lines"] == [
        {"id": "line-a"}
    ]

    restored_source = decode_image(
        loaded_a["state"]["source"]["image_b64"],
        mode="RGB",
    )
    assert restored_source.size == (40, 30)
    assert restored_source.getpixel((0, 0)) == (200, 100, 50)

    restored_mask = decode_image(
        loaded_a["state"]["masks"]["focus_mask_b64"],
        mode="L",
    )
    assert restored_mask.getpixel((10, 10)) == 255

    # Verify illustration B remains independent
    loaded_b = get_illustration(loaded_project, id_b)
    assert loaded_b["name"] == "Illustration B"
    assert loaded_b["state"]["annotations"]["manual_lines"] == []


def test_invalid_native_project_file():
    try:
        load_project_file(b"not a valid archive")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid archive was accepted")


def test_native_project_rejects_missing_illustration_id():
    project = create_empty_project("Invalid Project")

    project["illustrations"].append({
        "name": "Broken Illustration",
        "state": {},
    })

    project_bytes = save_project_file(project)

    try:
        load_project_file(project_bytes)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Project with missing illustration ID was accepted"
        )

def test_native_project_rejects_duplicate_illustration_ids():
    project = create_empty_project("Duplicate ID Test")

    illustration_id = add_illustration(
        project,
        "Illustration A",
        {"annotations": {"manual_lines": []}},
    )

    add_illustration(
        project,
        "Illustration B",
        {"annotations": {"manual_lines": []}},
    )

    # Simulate a corrupted project containing duplicate IDs
    project["illustrations"][1]["illustration_id"] = illustration_id

    project_bytes = save_project_file(project)

    try:
        load_project_file(project_bytes)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Project with duplicate illustration IDs was accepted"
        )


def test_native_project_rejects_missing_illustration_name():
    project = create_empty_project("Invalid Name Test")

    add_illustration(
        project,
        "Illustration A",
        {"annotations": {"manual_lines": []}},
    )

    # Simulate a corrupted illustration without a name.
    del project["illustrations"][0]["name"]

    project_bytes = save_project_file(project)

    try:
        load_project_file(project_bytes)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Project with missing illustration name was accepted"
        )


def test_native_project_rejects_invalid_illustration_state():
    project = create_empty_project("Invalid State Test")

    add_illustration(
        project,
        "Illustration A",
        {"annotations": {"manual_lines": []}},
    )

    # Simulate a corrupted illustration with an invalid state.
    project["illustrations"][0]["state"] = None

    project_bytes = save_project_file(project)

    try:
        load_project_file(project_bytes)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Project with invalid illustration state was accepted"
        )
