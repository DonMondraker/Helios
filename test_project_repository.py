
from app.project_state.project_model import (
    create_empty_project,
    add_illustration,
)
from app.project_state.project_repository import (
    save_project,
    load_project,
    list_projects,
    delete_project,
)


def test_project_persistence(tmp_path):
    database_path = tmp_path / "helios_test.db"

    # Create a project containing two illustrations.
    project = create_empty_project("Persistent Project")

    add_illustration(
        project,
        "Illustration A",
        {"annotations": {"manual_lines": [{"id": "line-1"}]}},
    )

    add_illustration(
        project,
        "Illustration B",
        {"annotations": {"manual_lines": []}},
    )

    project_id = project["project_id"]

    # Save the project to SQLite.
    save_project(project, database_path)

    # Load it using a separate database connection.
    restored = load_project(project_id, database_path)

    assert restored is not None
    assert restored["project_id"] == project_id
    assert restored["project_name"] == "Persistent Project"
    assert len(restored["illustrations"]) == 2

    assert (
            restored["illustrations"][0]["state"]
            ["annotations"]["manual_lines"][0]["id"]
            == "line-1"
    )

    # Modify the project and save it again.
    restored["project_name"] = "Updated Project"

    restored["illustrations"][1]["state"][
        "annotations"
    ]["manual_lines"].append({"id": "line-2"})

    save_project(restored, database_path)

    # Confirm the updated data survives another load.
    reloaded = load_project(project_id, database_path)

    assert reloaded["project_name"] == "Updated Project"
    assert (
            reloaded["illustrations"][1]["state"]
            ["annotations"]["manual_lines"][0]["id"]
            == "line-2"
    )


def test_loading_unknown_project(tmp_path):
    database_path = tmp_path / "helios_test.db"

    assert load_project("nonexistent-project", database_path) is None


def test_list_projects(tmp_path):
    database_path = tmp_path / "helios_test.db"

    # An empty database should return an empty list.
    assert list_projects(database_path) == []

    project_a = create_empty_project("Engine Assembly")
    project_b = create_empty_project("Hydraulic System")

    save_project(project_a, database_path)
    save_project(project_b, database_path)

    projects = list_projects(database_path)

    assert len(projects) == 2

    project_ids = {item["project_id"] for item in projects}

    assert project_a["project_id"] in project_ids
    assert project_b["project_id"] in project_ids

    project_names = {item["project_name"] for item in projects}

    assert project_names == {
        "Engine Assembly",
        "Hydraulic System",
    }

    # The listing should contain metadata, not image data.
    for item in projects:
        assert "created_at" in item
        assert "updated_at" in item
        assert "project_data" not in item

    # Saving an existing project should not create a duplicate.
    project_a["project_name"] = "Updated Engine Assembly"
    save_project(project_a, database_path)

    updated_projects = list_projects(database_path)

    assert len(updated_projects) == 2

    updated_a = next(
        item
        for item in updated_projects
        if item["project_id"] == project_a["project_id"]
    )

    assert updated_a["project_name"] == "Updated Engine Assembly"


def test_delete_project(tmp_path):
    database_path = tmp_path / "helios_test.db"

    project_a = create_empty_project("Engine Assembly")
    project_b = create_empty_project("Hydraulic System")

    save_project(project_a, database_path)
    save_project(project_b, database_path)

    # Delete only the first project.
    assert delete_project(
        project_a["project_id"],
        database_path,
    ) is True

    # The deleted project must no longer exist.
    assert load_project(
        project_a["project_id"],
        database_path,
    ) is None

    # The other project must remain intact.
    remaining = load_project(
        project_b["project_id"],
        database_path,
    )

    assert remaining is not None
    assert remaining["project_name"] == "Hydraulic System"

    # Deleting the same project again should return False.
    assert delete_project(
        project_a["project_id"],
        database_path,
    ) is False

    # Verify that only one project remains.
    projects = list_projects(database_path)

    assert len(projects) == 1
    assert projects[0]["project_id"] == project_b["project_id"]
