from core.rules_engine import get_illustration_rules


def build_prompt(instruction: str):
    rules = get_illustration_rules()

    def flatten(category):
        return "\n".join([f"- {r}" for r in category])

    prompt = f"""
You are a professional automotive service illustration generator.

TASK:
{instruction}

GLOBAL RULES:

STYLE:
{flatten(rules["style"])}

GEOMETRY:
{flatten(rules["geometry_rules"])}

REMOVAL LOGIC:
{flatten(rules["removal_logic"])}

COLOR SYSTEM:
{flatten(rules["color_system"])}

PROHIBITED ELEMENTS:
{flatten(rules["prohibited"])}

CLARITY:
{flatten(rules["clarity_rules"])}

CRITICAL INSTRUCTION:
Follow ALL rules strictly. Do not improvise.
Return a clean service illustration suitable for technical documentation.
"""

    return prompt