# core/groq_llm.py

import os
from dotenv import load_dotenv
from langchain.llms.base import LLM
from typing import Optional, List
from pydantic import PrivateAttr, Field
# Assuming you have a Groq client library
from groq import Groq

load_dotenv()

class GROQLLM(LLM):
    api_key: str = Field(os.getenv("GROQ_API_KEY"), description="The API key for the GROQ service")
    model: str = Field(os.getenv("MODEL_NAME"), description="The model to use for GROQ")
    _client: Groq = PrivateAttr()

    def __init__(self, api_key: str, model: str):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self._client = Groq(api_key=self.api_key)

    @property
    def _llm_type(self) -> str:
        return "GROQLLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            chat_completion = self._client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
            )
            return chat_completion.choices[0].message.content if chat_completion.choices else ""
        except Exception as e:
            return f"Error: {e}"
