# Let's implement a simple GPT chat bot that has conversation history for its api calls.
from __future__ import annotations

import json
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from responses.action_response import ActionResponse

class GPTClient:
    """Wrapper around the OpenAI client for vision-based prompts."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None, system_prompt: str | None = None) -> None:
        load_dotenv()
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name or "gpt-4.1"
        self.system_prompt = system_prompt

    def upload_image(self, path: str) -> str:
        with open(path, "rb") as f:
            result = self.client.files.create(file=f, purpose="vision")
        return result.id

    def chat(self, messages: List[dict], previous_response_id: int | None = None) -> str:
        if previous_response_id is not None:
            response = self.client.responses.create(
                model=self.model_name,
                instructions=self.system_prompt,
                input=messages,
                previous_response_id=previous_response_id,
            )
        else:
            response = self.client.responses.create(
                model=self.model_name,
                instructions=self.system_prompt,
                input=messages,
            )
        return response.output_text, response.id
    
    def parse(self, messages: List[dict], previous_response_id: int | None = None, text_format: type = None) -> ActionResponse | str:
        if previous_response_id is None:
            response = self.client.responses.parse(
                model=self.model_name,
                instructions=self.system_prompt,
                input=messages,
                text_format=text_format,
            )
        else:
            response = self.client.responses.parse(
                model=self.model_name,
                instructions=self.system_prompt,
                input=messages,
                previous_response_id=previous_response_id,
                text_format=text_format,
            )
        return response.output_parsed, response.id

if __name__ == "__main__":
    # Example usage
    client = GPTClient(api_key=None, model_name="gpt-4.1", system_prompt="You are a helpful assistant.")
    for i in range(10):
        input_text = input("Enter your message: ")
        if i == 0:
            response_text, response_id = client.chat([{"role": "user", 
                "content": [
                    {"type": "input_text", "text": input_text}
                ]
            }])
        else:
            response_text, response_id = client.chat([{"role": "user", 
                "content": [
                    {"type": "input_text", "text": input_text}
                ]}], 
                previous_response_id=response_id
                )
        print(f"Response: {response_text}")
