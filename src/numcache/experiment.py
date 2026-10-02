"""Measurement stages: remote answers for every question, local embeddings, local verifier decisions."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import backend, local, verifier
from .client import OpenRouter
from .data import Question
from .store import Store

BACKEND = "qwen/qwen-2.5-72b-instruct"
BACKEND_PROVIDER = "deepinfra"
BACKEND_TOKENS = 1024
WORKERS = 32
EMBED_BATCH = 32


def answer_all(questions: list[Question], store: Store) -> None:
    client = OpenRouter()

    def job(q: Question) -> None:
        try:
            reply = client.chat(BACKEND, backend.messages(q.text), BACKEND_TOKENS, BACKEND_PROVIDER)
        except RuntimeError:
            return
        row = {"stage": "backend", "model": BACKEND, "id": q.id, **asdict(reply)}
        store.add(row | {"answer": backend.final_answer(reply.text)})

    todo = [q for q in questions if not store.has("backend", BACKEND, q.id)]
    while todo:
        with ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(job, todo))
        left = [q for q in todo if not store.has("backend", BACKEND, q.id)]
        if len(left) == len(todo):
            raise RuntimeError(f"{len(left)} calls failed repeatedly")
        todo = left


def embed_all(model: str, questions: list[Question], out: Path) -> None:
    if out.exists():
        return
    vectors, seconds = [], 0.0
    for k in range(0, len(questions), EMBED_BATCH):
        batch, t = local.embed(model, [q.text for q in questions[k : k + EMBED_BATCH]])
        vectors.extend(batch)
        seconds += t
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    np.savez_compressed(out, ids=np.array([q.id for q in questions]), vectors=matrix, seconds=seconds)


def judgement(store: Store, model: str, query: Question, cached: Question, answer: float) -> dict | None:
    """The stored verifier judgement for this request if it was made for this stored question and answer.

    Early records lack the "shown" field; they were made with the answer formatted as f"{answer:g}", taken from
    their "cached_answer" field or, in the first records, from the answer stored with the remote reply."""
    if not store.has("verify", model, query.id):
        return None
    row = store.get("verify", model, query.id)
    legacy = row.get("cached_answer", store.get("backend", BACKEND, cached.id)["answer"])
    shown = row.get("shown", None if legacy is None else f"{legacy:g}")
    return row if row["cached"] == cached.id and shown == verifier.shown(answer) else None


def verify_all(model: str, pairs: list[tuple[Question, Question, float]], store: Store) -> None:
    for query, cached, cached_answer in pairs:
        if judgement(store, model, query, cached, cached_answer):
            continue
        p, text, seconds = local.yes_probability(model, verifier.messages(query.text, cached.text, cached_answer))
        store.add({"stage": "verify", "model": model, "id": query.id, "cached": cached.id,
                   "shown": verifier.shown(cached_answer), "p_yes": p, "text": text, "seconds": seconds})
