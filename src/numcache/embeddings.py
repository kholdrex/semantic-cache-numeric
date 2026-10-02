"""Stored question embeddings and the nearest stored question of every request."""

from pathlib import Path

import numpy as np

from .data import Question


def path(results: Path, model: str) -> Path:
    return results / f"embeddings_{model.replace(':', '_')}.npz"


def nearest(file: Path, seeds: list[Question], queries: list[Question]) -> tuple[np.ndarray, np.ndarray, float]:
    """Index of the most similar stored question, its cosine similarity, and the mean embedding time per text."""
    saved = np.load(file)
    index = {i: k for k, i in enumerate(saved["ids"])}
    s = saved["vectors"][[index[q.id] for q in seeds]]
    v = saved["vectors"][[index[q.id] for q in queries]]
    sims = v @ s.T
    return sims.argmax(axis=1), sims.max(axis=1), float(saved["seconds"]) / len(saved["ids"])
