"""Parsing utilities for ReAct steps."""

import json
import re


def _parse_react_step(text: str) -> tuple[str | None, str | None, dict, str | None]:
    """
    Parse one ReAct step from model output.

    Returns:
        (thought, action, action_input_dict, final_answer)
        where final_answer is non-None only when the agent is done.
    """
    final_match = re.search(r"Final Answer\s*:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        return None, None, {}, final_match.group(1).strip()

    thought_match = re.search(r"Thought\s*:\s*(.+?)(?=Action\s*:|$)", text, re.DOTALL | re.IGNORECASE)
    action_match = re.search(r"Action\s*:\s*(\w+)", text, re.IGNORECASE)
    input_match = re.search(
        r"Action Input\s*:\s*(\{.*?\}|\S.*?)(?=\nObservation|\nThought|\nAction|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.group(1).strip() if action_match else ""

    try:
        action_input = json.loads(input_match.group(1).strip()) if input_match else {}
    except (json.JSONDecodeError, AttributeError):
        action_input = {}

    return thought, action, action_input, None
