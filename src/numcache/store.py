"""Append-only JSONL store of call results keyed by (stage, model, question id)."""

import json
import threading
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.rows: dict[tuple[str, str, str], dict] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with path.open() as f:
                for row in map(json.loads, f):
                    self.rows[(row["stage"], row["model"], row["id"])] = row

    def has(self, stage: str, model: str, id: str) -> bool:
        return (stage, model, id) in self.rows

    def get(self, stage: str, model: str, id: str) -> dict:
        return self.rows[(stage, model, id)]

    def add(self, row: dict) -> None:
        with self.lock:
            self.rows[(row["stage"], row["model"], row["id"])] = row
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
