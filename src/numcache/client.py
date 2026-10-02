"""OpenRouter chat completions with retries; records tokens, provider and wall-clock time including retries."""

import os
import time
from dataclasses import dataclass

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
RETRIES = 8


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    provider: str
    seconds: float


class OpenRouter:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {os.environ['OPENROUTER_API_KEY']}"

    def chat(self, model: str, messages: list[dict], max_tokens: int, provider: str) -> Reply:
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "provider": {"order": [provider], "allow_fallbacks": False},
        }
        start = time.perf_counter()
        for attempt in range(RETRIES):
            try:
                response = self.session.post(URL, json=body, timeout=120)
                data = response.json()
            except (requests.RequestException, ValueError):
                time.sleep(min(2**attempt, 60))
                continue
            if response.status_code == 200 and data.get("choices"):
                usage = data.get("usage") or {}
                return Reply(
                    data["choices"][0]["message"].get("content") or "",
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    data.get("provider", ""),
                    time.perf_counter() - start,
                )
            time.sleep(min(2**attempt, 60))
        raise RuntimeError(f"{model}: no reply after {RETRIES} attempts")
