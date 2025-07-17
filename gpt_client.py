from __future__ import annotations

import json
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI


class GPTClient:
    """Wrapper around the OpenAI client for vision-based prompts."""

    def __init__(self, api_key: str | None = None) -> None:
        load_dotenv()
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def upload_image(self, path: str) -> str:
        with open(path, "rb") as f:
            result = self.client.files.create(file=f, purpose="vision")
        return result.id

    def chat(self, system_prompt: str, messages: List[dict], model_name: str = "gpt-4.1") -> str:
        response = self.client.responses.create(
            model=model_name,
            instructions=system_prompt,
            input=messages,
        )
        return response.output_text

    def save_json(self, text: str, out_path: str) -> None:
        try:
            data = json.loads(text)
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved response to {out_path}")
        except json.JSONDecodeError:
            print("Response was not valid JSON")
            