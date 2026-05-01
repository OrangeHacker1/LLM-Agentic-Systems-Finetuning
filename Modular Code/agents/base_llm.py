"""# models/base_llm.py

import time

class BaseLLM:

    def __init__(self, client, model_name="gpt-4", temperature=0.2, max_tokens=512):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def query(self, prompt, retries=3, backoff=1.5):

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a strict JSON generator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                return response.choices[0].message.content

            except Exception as e:
                time.sleep(backoff ** attempt)

        return None
        """

# models/base_llm.py

import time

class BaseLLM:

    def __init__(self, env, config, client):
        self.client = client

        self.base_url = env["teacher_base"]
        self.model_name = env["teacher_model"]
        self.api_key = env["teacher_key"]

        self.temperature = config["generation"]["temperature"]
        self.max_tokens = config["generation"]["max_tokens"]


    def query(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            print("\n🚨 LLM ERROR:")
            print(e)
            return None
    
    
    def query2(self, prompt, retries=3, backoff=1.5):

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a strict JSON generator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                return response.choices[0].message.content

            except Exception:
                time.sleep(backoff ** attempt)

        return None