import json
import random

from config.config_loader import load_config
from config.env_loader import load_environment
from agents.client_factory import create_client
from agents.base_llm import BaseLLM
from judge.prompts import JUDGE_PROMPT


class LLMJudge:
    def __init__(self):
        config = load_config()
        env = load_environment()

        client = create_client(env)
        self.llm = BaseLLM(env, config, client)

    def _safe_parse(self, raw):
        try:
            return json.loads(raw)
        except:
            pass

        start = raw.find("{")
        end = raw.rfind("}") + 1

        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except:
                return None

        return None

    def evaluate(self, sample, resp1, resp2):
        swap = random.choice([True, False])

        a = resp2 if swap else resp1
        b = resp1 if swap else resp2

        prompt = JUDGE_PROMPT.format(
            instruction=sample["instruction"],
            input_text=sample.get("input", ""),
            response_a=a,
            response_b=b
        )

        raw = self.llm.query(prompt)
        parsed = self._safe_parse(raw or "")

        if parsed is None:
            return {
                "winner": "Tie",
                "error": "judge_parse_failed",
                "raw_output": raw
            }

        winner = parsed.get("winner", "Tie")

        if swap:
            if winner == "A":
                winner = "B"
            elif winner == "B":
                winner = "A"

        parsed["winner"] = winner
        return parsed