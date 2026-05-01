JUDGE_PROMPT = """
You are an expert evaluator comparing two model responses.

Evaluate Response A and Response B for the user prompt below.

Be strict and unbiased. Ignore response order.

User Prompt:
{instruction}

Optional Input:
{input_text}

Response A:
{response_a}

Response B:
{response_b}

Score each response from 1 to 5 on:
1. instruction_following
2. correctness
3. clarity
4. completeness
5. structured_output_validity
6. hallucination_risk

Then choose a winner:
- "A"
- "B"
- "Tie"

Return ONLY valid JSON in this schema:

{
  "response_a_scores": {
    "instruction_following": int,
    "correctness": int,
    "clarity": int,
    "completeness": int,
    "structured_output_validity": int,
    "hallucination_risk": int
  },
  "response_b_scores": {
    "instruction_following": int,
    "correctness": int,
    "clarity": int,
    "completeness": int,
    "structured_output_validity": int,
    "hallucination_risk": int
  },
  "winner": "A|B|Tie",
  "justification": "brief reason"
}
"""


PAIRWISE_PROMPT = '''You are an impartial evaluator.
Task Prompt:\n{prompt}\n\nResponse A:\n{resp_a}\n\nResponse B:\n{resp_b}\n\nScore each response from 1-5 for:
- instruction_following
- correctness
- clarity
- completeness
- structured_output_validity
- hallucination_risk

Return ONLY JSON:
{{
  "response_a_scores": {{...}},
  "response_b_scores": {{...}},
  "winner": "A|B|Tie",
  "justification": "short reason"
}}
'''