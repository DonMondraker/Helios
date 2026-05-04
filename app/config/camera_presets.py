from __future__ import annotations

import math


def _normalize_vec(dx: float, dy: float) -> tuple[float, float]:
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return 0.0, 0.0
    return dx / length, dy / length


CAMERA_PRESETS = {
    "TRUCK_ISO1": {
        "label": "Truck ISO 1",
        "remove_vector": (-0.94, 0.34),
    },
    "TRUCK_ISO2": {
        "label": "Truck ISO 2",
        "remove_vector": (0.94, 0.34),
    },
    "UP": {
        "label": "Up",
        "direction": "up",
    },
    "DOWN": {
        "label": "Down",
        "direction": "down",
    },
    "LEFT": {
        "label": "Left",
        "direction": "left",
    },
    "RIGHT": {
        "label": "Right",
        "direction": "right",
    },
    "UP_LEFT": {
        "label": "Up-left",
        "direction": "up-left",
    },
    "UP_RIGHT": {
        "label": "Up-right",
        "direction": "up-right",
    },
    "DOWN_LEFT": {
        "label": "Down-left",
        "direction": "down-left",
    },
    "DOWN_RIGHT": {
        "label": "Down-right",
        "direction": "down-right",
    },
}


def resolve_direction(camera_preset: str, operation_type: str) -> str:
    preset = CAMERA_PRESETS.get(camera_preset, {})
    base_direction = preset.get("direction", "down-left")

    reverse_map = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
        "up-right": "down-left",
        "up-left": "down-right",
        "down-right": "up-left",
        "down-left": "up-right",
        "outward": "inward",
        "inward": "outward",
        "none": "none",
    }

    if operation_type == "Install":
        return reverse_map.get(base_direction, base_direction)

    return base_direction


def resolve_direction_vector(camera_preset: str, operation_type: str) -> tuple[float, float] | None:
    preset = CAMERA_PRESETS.get(camera_preset, {})
    vector = preset.get("remove_vector")

    if vector is None:
        return None

    dx, dy = vector

    if operation_type == "Install":
        dx, dy = -dx, -dy

    return _normalize_vec(dx, dy)