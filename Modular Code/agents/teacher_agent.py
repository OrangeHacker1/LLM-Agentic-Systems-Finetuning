# agents/teacher_agent.py

import json
from utils.json_utils import parse_json_safe, validate_schema

"""
class TeacherAgent:

    def __init__(self, llm, prompts, max_retries=3):
        self.llm = llm
        self.prompts = prompts
        self.max_retries = max_retries

    def generate(self, task_type, input_text):
        " ""
        Main entry point
        task_type: extraction | classification | schema | repair | tool
        " ""

        prompt_template = self.prompts[task_type]

        prompt = prompt_template.replace("{input}", input_text)

        return self._get_valid_response(prompt)

    def _get_valid_response(self, prompt):
        " ""
        Ensures valid JSON output with retries
        " ""

        for attempt in range(self.max_retries):

            raw_output = self.llm.query(prompt)

            parsed = parse_json_safe(raw_output)

            if parsed is not None:
                return parsed  # RETURN STRUCTURED JSON

        return None  # failed after retries
        """

class TeacherAgent:

    def __init__(self, llm, prompts, config):
        self.llm = llm
        self.prompts = prompts
        self.max_retries = config["generation"]["retries"]

    """
    def generate(self, task_type, input_text):

        prompt_template = self.prompts[task_type]
        prompt = prompt_template.replace("{input}", input_text)

        return self._get_valid_response(prompt)
    """
    def generate(self, task_type, input_text, failure_log):

        prompt_template = self.prompts[task_type]
        prompt = prompt_template.replace("{input}", input_text)

        for _ in range(self.max_retries):

            raw_output = self.llm.query(prompt)
            print(raw_output)
            if raw_output is None:
                failure_log["none_output"] += 1
                print("ERROR: NO OUTPUT")
                continue

            parsed = parse_json_safe(raw_output)
            print(f"RAW = == = == = = =\n\n\n\n{raw_output}")
            print(f"parsed:\n\n{parsed}\n\n\n\n\n\n=======")
            if parsed is None:
                failure_log["invalid_json"] += 1
                continue

            if not validate_schema(task_type, parsed):
                failure_log["schema_fail"] += 1

                # OPTIONAL: still keep it if it's valid JSON
                # helps diversity for hybrid sampling
                return parsed

            return parsed

        return None

    def _get_valid_response(self, prompt):

        for _ in range(self.max_retries):

            raw_output = self.llm.query(prompt)

            parsed = parse_json_safe(raw_output)

            if parsed is not None:
                return parsed

        return None