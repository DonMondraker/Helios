import json
from typing import Any, Dict, Optional

# from ai.client import get_client
from ai.client import get_openai_client

def build_suggestion_system_prompt() -> str:
    return """
You are assisting with automotive service illustrations.

Your task is to choose an annotation strategy for a technical illustration.

Do NOT generate a new image.
Do NOT place exact final annotations by artistic guesswork.
Instead, choose a strategy that deterministic code can apply.

Rules:
- Return JSON only.
- Prefer concise and practical strategies.
- Use movement lines for removal/installation, not arrows.
- Keep suggestions minimal and clear.
- For repeated small parts, prefer grouped logic unless the task clearly requires separate emphasis.
- Callout labels must be short such as "A", "B", "1", "2".
- Do not use words as callout labels.

Return JSON in this exact structure:
{
  "task_type": "removal | installation | fastener_operation | reference | measurement | unknown",
  "target_mode": "single_large_part | multiple_repeated_small_parts | multiple_mixed_parts | sparse_small_parts | unknown",
  "suggested_direction": "up | down | left | right | up-right | up-left | down-right | down-left | outward | inward | none",
  "movement_line_strategy": "none | short_parallel_near_focus | two_lines_per_large_part | grouped_repeated_targets",
  "callout_strategy": "none | single_main_callout | grouped_callout | separate_main_parts",
  "max_callouts": 0,
  "notes": [
    "short suggestion"
  ]
}

Guidance:
- If the task involves removal, installation, or movement, choose a non-none suggested_direction when possible.
- If the structure is repeated small parts, prefer grouped logic.
- If uncertain, return conservative strategies and explain uncertainty in notes.
""".strip()


def _summarize_components_for_prompt(components: list[dict], max_components: int = 8) -> str:
    if not components:
        return "No components detected."

    lines = []
    for comp in components[:max_components]:
        lines.append(
            f'- id={comp["component_id"]}, '
            f'area={comp["area"]}, '
            f'centroid={tuple(comp["centroid"])}, '
            f'bbox={tuple(comp["bbox"])}'
        )
    return "\n".join(lines)


def build_suggestion_user_prompt(
        template_instruction: str,
        image_width: int,
        image_height: int,
        focus_bbox: Optional[tuple[int, int, int, int]],
        context_bbox: Optional[tuple[int, int, int, int]],
        focus_analysis: Optional[dict[str, Any]],
        extra_details: str = "",
) -> str:
    if focus_analysis is None:
        focus_analysis_text = "No focus analysis available."
        component_summary = "No components detected."
    else:
        focus_analysis_text = (
            f'component_count={focus_analysis.get("component_count")}\n'
            f'total_focus_area={focus_analysis.get("total_focus_area")}\n'
            f'largest_component_area={focus_analysis.get("largest_component_area")}\n'
            f'median_component_area={focus_analysis.get("median_component_area")}\n'
            f'repeated_small_parts={focus_analysis.get("repeated_small_parts")}\n'
            f'dominant_structure={focus_analysis.get("dominant_structure")}\n'
            f'focus_bbox={focus_analysis.get("focus_bbox")}'
        )
        component_summary = _summarize_components_for_prompt(
            focus_analysis.get("components", []),
            max_components=8,
        )

    return f"""
Illustration size:
- width: {image_width}
- height: {image_height}

Task:
{template_instruction}

Additional details:
{extra_details}

Focus bounding box:
{focus_bbox}

Context bounding box:
{context_bbox}

Focus analysis summary:
{focus_analysis_text}

Top detected focus components:
{component_summary}

Choose an annotation strategy suitable for deterministic placement.
Return JSON only.
""".strip()


def request_ai_suggestions(
        template_instruction: str,
        image_width: int,
        image_height: int,
        focus_bbox: Optional[tuple[int, int, int, int]],
        context_bbox: Optional[tuple[int, int, int, int]],
        focus_analysis: Optional[dict[str, Any]] = None,
        extra_details: str = "",
        model: str = "gpt-5.4-mini",
) -> Dict[str, Any]:
    # client = get_client()
    client = get_openai_client()

    system_prompt = build_suggestion_system_prompt()
    user_prompt = build_suggestion_user_prompt(
        template_instruction=template_instruction,
        image_width=image_width,
        image_height=image_height,
        focus_bbox=focus_bbox,
        context_bbox=context_bbox,
        focus_analysis=focus_analysis,
        extra_details=extra_details,
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = response.output_text.strip()

    try:
        parsed = json.loads(text)
        parsed.setdefault("task_type", "unknown")
        parsed.setdefault("target_mode", "unknown")
        parsed.setdefault("suggested_direction", "none")
        parsed.setdefault("movement_line_strategy", "none")
        parsed.setdefault("callout_strategy", "none")
        parsed.setdefault("max_callouts", 0)
        parsed.setdefault("notes", [])
        return parsed
    except json.JSONDecodeError:
        return {
            "task_type": "unknown",
            "target_mode": "unknown",
            "suggested_direction": "none",
            "movement_line_strategy": "none",
            "callout_strategy": "none",
            "max_callouts": 0,
            "notes": [f"Could not parse AI response as JSON. Raw response: {text}"],
        }