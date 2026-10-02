"""Ollama embeddings and chat with log-probabilities of the first generated token."""

import math
import time

import requests

OLLAMA = "http://localhost:11434"


def embed(model: str, texts: list[str]) -> tuple[list[list[float]], float]:
    start = time.perf_counter()
    response = requests.post(f"{OLLAMA}/api/embed", json={"model": model, "input": texts}, timeout=600)
    response.raise_for_status()
    return response.json()["embeddings"], time.perf_counter() - start


def yes_probability(model: str, messages: list[dict]) -> tuple[float, str, float]:
    """Probability mass of 'yes' among the top first tokens, normalised by yes + no; reply text; seconds."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "logprobs": True,
        "top_logprobs": 10,
        "options": {"temperature": 0, "num_predict": 1, "num_ctx": 4096},
    }
    start = time.perf_counter()
    response = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=600)
    response.raise_for_status()
    seconds = time.perf_counter() - start
    data = response.json()
    yes = no = 0.0
    for t in data["logprobs"][0]["top_logprobs"]:
        word = t["token"].strip().lower()
        if word == "yes":
            yes += math.exp(t["logprob"])
        elif word == "no":
            no += math.exp(t["logprob"])
    p = yes / (yes + no) if yes + no > 0 else 0.5
    return p, data["message"]["content"], seconds
