import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("TEACHER_BASE_URL")
MODEL = os.getenv("TEACHER_MODEL")
API_KEY = os.getenv("TEACHER_API_KEY")


def call_teacher(prompt, temperature=0):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]