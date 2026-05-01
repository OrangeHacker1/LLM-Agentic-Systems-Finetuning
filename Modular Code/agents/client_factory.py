# models/client_factory.py

from openai import OpenAI


def create_client(env):
    """
    Creates an OpenAI-compatible client using environment variables.
    """

    client = OpenAI(
        api_key=env["teacher_key"],
        base_url=env["teacher_base"]  # works for OpenAI OR OpenAI-compatible APIs
    )

    return client