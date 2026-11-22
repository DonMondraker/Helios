import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from .project_model import (
    serialize_project,
    deserialize_project,
)


DEFAULT_DATABASE_PATH = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "helios_projects.db"
)


def initialize_database(
        database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    """Create the database and project table if needed."""

    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                project_name TEXT NOT NULL,
                project_data BLOB NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def save_project(
        project: dict,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    """Insert or update a Helios project in SQLite."""

    initialize_database(database_path)

    now = datetime.now(timezone.utc).isoformat()

    project_id = project["project_id"]
    project_name = project["project_name"]
    created_at = project["created_at"]
    project_data = serialize_project(project)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO projects (
                project_id,
                project_name,
                project_data,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                project_name = excluded.project_name,
                project_data = excluded.project_data,
                updated_at = excluded.updated_at
            """,
            (
                project_id,
                project_name,
                project_data,
                created_at,
                now,
            ),
        )
        connection.commit()


def load_project(
        project_id: str,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict | None:
    """Load a Helios project by its unique ID."""

    initialize_database(database_path)

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT project_data
            FROM projects
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()

    if row is None:
        return None

    return deserialize_project(row[0])


def list_projects(
        database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict]:
    """List saved projects, most recently updated first."""

    initialize_database(database_path)

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                project_id,
                project_name,
                created_at,
                updated_at
            FROM projects
            ORDER BY updated_at DESC, project_id ASC
            """
        ).fetchall()

    return [
        {
            "project_id": row[0],
            "project_name": row[1],
            "created_at": row[2],
            "updated_at": row[3],
        }
        for row in rows
    ]


def delete_project(
        project_id: str,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> bool:
    """Delete a saved project by ID.

    Returns True if the project existed and was deleted.
    Returns False if no matching project was found.
    """

    initialize_database(database_path)

    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            DELETE FROM projects
            WHERE project_id = ?
            """,
            (project_id,),
        )
        connection.commit()

        return cursor.rowcount > 0
