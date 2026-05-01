# prompts/teacher_prompts.py

TEACHER_PROMPTS = {

    "extraction": """
Extract structured information as JSON.

Rules:
- Output ONLY valid JSON
- No explanations
- Use double quotes
- Start with a "{" and end with a "}"

Schema:
{
  "name": string,
  "location": string,
  "date": string
}

Text:
{input}
""",

    "classification": """
Classify the following text.

Return ONLY JSON:
{
  "label": "supported" | "refuted" | "neutral"
}

Text:
{input}
""",

    "schema": """
Convert the text into structured JSON.

Output ONLY JSON with meaningful keys.

Text:
{input}
""",

    "repair": """
Fix the following broken JSON.

Return ONLY valid JSON.

Input:
{input}
""",

    "tool": """
Generate function arguments in JSON.

Function:
get_weather(city: string, date: string)

Input:
{input}

Return ONLY JSON.
"""
}