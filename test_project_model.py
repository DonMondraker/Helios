from app.project_state.project_model import (
    create_empty_project,
    add_illustration,
    get_illustration,
    remove_illustration,
    update_illustration_state,
    serialize_project,
    deserialize_project,
    encode_image,
    decode_image,
    update_illustration_fields,
)

from PIL import Image


def test_project_model():
    # Create a new project
    project = create_empty_project("Helios Test Project")

    # Create two illustrations with different settings
    illustration_a = {
        "render_settings": {
            "enhance_focus": True,
        },
        "annotations": {
            "manual_lines": [],
        },
    }

    illustration_b = {
        "render_settings": {
            "enhance_focus": False,
        },
        "annotations": {
            "manual_lines": [],
        },
    }

    # Add both illustrations
    id_a = add_illustration(
        project,
        "Illustration A",
        illustration_a,
    )

    id_b = add_illustration(
        project,
        "Illustration B",
        illustration_b,
    )

    # Verify project structure
    assert project["project_version"] == "2.0"
    assert len(project["illustrations"]) == 2
    assert id_a != id_b

    # Verify independent illustration states
    assert project["illustrations"][0]["state"]["render_settings"]["enhance_focus"] is True
    assert project["illustrations"][1]["state"]["render_settings"]["enhance_focus"] is False

    # Modify illustration A
    project["illustrations"][0]["state"]["render_settings"]["enhance_focus"] = False

    # Illustration B should remain unchanged
    assert project["illustrations"][1]["state"]["render_settings"]["enhance_focus"] is False
    
        # Retrieve illustrations by ID
    retrieved_a = get_illustration(project, id_a)
    retrieved_b = get_illustration(project, id_b)

    assert retrieved_a is not None
    assert retrieved_b is not None

    assert retrieved_a["name"] == "Illustration A"
    assert retrieved_b["name"] == "Illustration B"

    # Unknown IDs should return None
    assert get_illustration(project, "nonexistent-id") is None
    
        # Update illustration A
    updated_state = {
        "render_settings": {
            "enhance_focus": True,
        },
        "annotations": {
            "manual_lines": [
                {"id": "line-001", "length": 100}
            ],
        },
    }

    assert update_illustration_state(
        project,
        id_a,
        updated_state,
    ) is True

    # Verify the updated state was saved
    retrieved_a = get_illustration(project, id_a)

    assert retrieved_a["state"]["render_settings"]["enhance_focus"] is True
    assert len(retrieved_a["state"]["annotations"]["manual_lines"]) == 1

    # Verify illustration B was not affected
    retrieved_b = get_illustration(project, id_b)

    assert retrieved_b["state"]["annotations"]["manual_lines"] == []

    # Verify the state is independent of the original dictionary
    updated_state["annotations"]["manual_lines"].clear()

    assert len(retrieved_a["state"]["annotations"]["manual_lines"]) == 1

    # Updating an unknown illustration should fail safely
    assert update_illustration_state(
        project,
        "nonexistent-id",
        updated_state,
    ) is False
    
    
        # Verify that adding an illustration copies its state
    shared_state = {
        "annotations": {
            "manual_lines": [],
        },
    }

    id_c = add_illustration(
        project,
        "Illustration C",
        shared_state,
    )

    # Modify the original state after adding it
    shared_state["annotations"]["manual_lines"].append(
        {"id": "line-999"}
    )

    # Illustration C must remain unchanged
    retrieved_c = get_illustration(project, id_c)

    assert retrieved_c["state"]["annotations"]["manual_lines"] == []
    
        # Verify that a complete project can be serialized
    project_bytes = serialize_project(project)

    assert isinstance(project_bytes, bytes)

    # Verify the JSON contains our project information
    import json

    saved_project = json.loads(project_bytes.decode("utf-8"))

    assert saved_project["project_version"] == "2.0"
    assert saved_project["project_name"] == "Helios Test Project"
    assert len(saved_project["illustrations"]) == 3

    # Verify illustration data survives serialization
    saved_a = saved_project["illustrations"][0]

    assert saved_a["name"] == "Illustration A"
    assert saved_a["state"]["render_settings"]["enhance_focus"] is True
    
        # Load the serialized project
    loaded_project = deserialize_project(project_bytes)

    assert loaded_project["project_id"] == project["project_id"]
    assert loaded_project["project_name"] == project["project_name"]
    assert len(loaded_project["illustrations"]) == 3

    # Verify that illustration data survived saving and loading
    loaded_a = get_illustration(loaded_project, id_a)

    assert loaded_a is not None
    assert loaded_a["name"] == "Illustration A"
    assert loaded_a["state"]["render_settings"]["enhance_focus"] is True

    # Invalid JSON should produce a clear error
    try:
        deserialize_project(b"not valid JSON")
        assert False, "Invalid JSON was incorrectly accepted"
    except ValueError:
        pass

    # Unsupported project versions should be rejected
    invalid_version = {
        "project_version": "99.0",
        "illustrations": [],
    }

    try:
        deserialize_project(
            json.dumps(invalid_version).encode("utf-8")
        )
        assert False, "Unsupported version was incorrectly accepted"
    except ValueError:
        pass
    
        # Test image encoding and decoding
    original_image = Image.new(
        "RGB",
        (100, 80),
        color=(255, 160, 70),
    )

    encoded_image = encode_image(original_image)

    assert isinstance(encoded_image, str)

    restored_image = decode_image(
        encoded_image,
        mode="RGB",
    )

    assert restored_image is not None
    assert restored_image.size == (100, 80)
    assert restored_image.getpixel((0, 0)) == (255, 160, 70)

    # Test grayscale mask encoding
    original_mask = Image.new(
        "L",
        (100, 80),
        color=255,
    )

    encoded_mask = encode_image(original_mask)
    restored_mask = decode_image(encoded_mask, mode="L")

    assert restored_mask is not None
    assert restored_mask.mode == "L"
    assert restored_mask.getpixel((0, 0)) == 255

    # Test optional images
    assert encode_image(None) is None
    assert decode_image(None) is None
    
        # Test saving and loading a project containing an image
    image_project = create_empty_project("Image Persistence Test")

    source_image = Image.new(
        "RGB",
        (100, 80),
        color=(255, 160, 70),
    )

    focus_mask = Image.new(
        "L",
        (100, 80),
        color=255,
    )

    image_state = {
        "source": {
            "filename": "test_illustration.png",
            "image_b64": encode_image(source_image),
        },
        "masks": {
            "focus_mask_b64": encode_image(focus_mask),
        },
        "annotations": {
            "react_editor_state": None,
        },
    }

    image_id = add_illustration(
        image_project,
        "Test Illustration",
        image_state,
    )

    # Serialize and deserialize the complete project
    image_project_bytes = serialize_project(image_project)
    restored_project = deserialize_project(image_project_bytes)

    # Retrieve the illustration
    restored_illustration = get_illustration(
        restored_project,
        image_id,
    )

    assert restored_illustration is not None

    restored_state = restored_illustration["state"]

    # Restore the source image
    restored_source = decode_image(
        restored_state["source"]["image_b64"],
        mode="RGB",
    )

    assert restored_source is not None
    assert restored_source.size == (100, 80)
    assert restored_source.getpixel((0, 0)) == (255, 160, 70)

    # Restore the focus mask
    restored_focus_mask = decode_image(
        restored_state["masks"]["focus_mask_b64"],
        mode="L",
    )

    assert restored_focus_mask is not None
    assert restored_focus_mask.size == (100, 80)
    assert restored_focus_mask.getpixel((0, 0)) == 255

    # Test partial illustration state updates
    partial_project = create_empty_project("Partial Update Test")

    partial_id = add_illustration(
        partial_project,
        "Partial Illustration",
        {
            "source": {"filename": "original.png"},
            "masks": {"focus_mask_b64": None},
            "annotations": {"react_editor_state": None},
        },
    )

    updated = update_illustration_fields(
        partial_project,
        partial_id,
        {
            "masks": {"focus_mask_b64": "encoded-mask-data"},
            "annotations": {
                "react_editor_state": {"lines": [{"id": "line-1"}]},
            },
        },
    )

    assert updated is True

    partial_state = get_illustration(
        partial_project,
        partial_id,
    )["state"]

    # Original source information must remain intact
    assert partial_state["source"]["filename"] == "original.png"

    # Updated fields must be saved
    assert partial_state["masks"]["focus_mask_b64"] == "encoded-mask-data"
    assert partial_state["annotations"]["react_editor_state"]["lines"][0]["id"] == "line-1"

    # Updating an unknown illustration must fail safely
    assert update_illustration_fields(
        partial_project,
        "nonexistent-id",
        {"masks": {}},
    ) is False
    
        # Remove illustration A
    assert remove_illustration(project, id_a) is True

    # Verify illustration A is gone
    assert get_illustration(project, id_a) is None

    # Verify illustration B remains
    assert get_illustration(project, id_b) is not None
    assert len(project["illustrations"]) == 2

    # Removing an unknown illustration should fail safely
    assert remove_illustration(project, "nonexistent-id") is False

    print("Helios 2.0 project model test passed!")


if __name__ == "__main__":
    test_project_model()
