"""GSM-Plus: GSM8K test questions (cache entries) and their eight kinds of variation (incoming requests)."""

import json
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import requests

URL = (
    "https://huggingface.co/datasets/qintongli/GSM-Plus/resolve/"
    "3b708db57b96a16e8e3368ed2956990c0809440e/data/test-00000-of-00001.jsonl"
)


@dataclass(frozen=True)
class Question:
    id: str
    seed: int
    kind: str
    text: str
    answer: float | None


def number(text: str) -> float | None:
    """Numerical value of an answer such as "18", "1,250.5" or "3/4"; None for anything else (e.g. "None")."""
    text = str(text).strip().replace(",", "").lstrip("$").rstrip("%")
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None


def download(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(URL, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def load(path: Path) -> tuple[list[Question], list[Question]]:
    """Seed questions (one per GSM8K test item) and all variations, linked by the seed index."""
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    index: dict[str, int] = {}
    seeds, variations = [], []
    for r in rows:
        if r["seed_question"] not in index:
            index[r["seed_question"]] = len(index)
            k = len(seeds)
            seeds.append(Question(f"seed/{k}", k, "seed", r["seed_question"], number(r["seed_answer"])))
        s = index[r["seed_question"]]
        kind = r["perturbation_type"]
        variations.append(Question(f"var/{s}/{kind}", s, kind, r["question"], number(r["answer"])))
    return seeds, variations


def split(n_seeds: int, dev_share: float, seed: int) -> set[int]:
    """Seed indices used for choosing thresholds; all other seeds (and their variations) form the test set."""
    rng = random.Random(seed)
    return set(rng.sample(range(n_seeds), round(n_seeds * dev_share)))
