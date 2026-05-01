# utils/json_utils.py

import json
import re


def extract_json(text):
    """
    Extract the FIRST valid JSON object from text.
    Non-greedy + safer parsing.
    """

    if text is None:
        return None

    matches = re.findall(r"\{.*?\}", text, re.DOTALL)

    for match in matches:
        try:
            json.loads(match)
            return match
        except:
            continue

    return None


def parse_json_safe(text):
    """
    Safely extract and parse JSON
    """

    try:
        cleaned = extract_json(text)

        if cleaned is None:
            return None

        parsed = json.loads(cleaned)

        # Ensure it's a dictionary
        if not isinstance(parsed, dict):
            return None

        return normalize_keys(parsed)

    except:
        return None


def normalize_keys(obj):
    """
    Normalize keys to lowercase for consistency
    """

    return {k.lower(): v for k, v in obj.items()}


def validate_schema(task_type, obj):
    """
    Flexible schema validation (important for hybrid sampling)
    """

    if not isinstance(obj, dict):
        return False

    # -------- EXTRACTION --------
    if task_type == "extraction":
        # Allow flexible keys but require at least 2 fields
        return len(obj) >= 2

    # -------- CLASSIFICATION --------
    elif task_type == "classification":
        label = obj.get("label") or obj.get("category")

        if not isinstance(label, str):
            return False

        return label.lower() in ["supported", "refuted", "neutral"]

    # -------- TOOL --------
    elif task_type == "tool":
        # Must contain at least 1 argument
        return len(obj) >= 1

    # -------- REPAIR --------
    elif task_type == "repair":
        return isinstance(obj, dict)

    # -------- SCHEMA --------
    elif task_type == "schema":
        # Must contain structured fields
        return len(obj) >= 2

    return False