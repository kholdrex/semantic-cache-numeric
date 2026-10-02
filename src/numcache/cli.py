import argparse
from pathlib import Path

from . import data, embeddings, experiment
from .backend import final_answer
from .store import Store

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 2026
DEV_SHARE = 0.2
EMBEDDERS = ["all-minilm", "bge-m3"]
VERIFIERS = ["qwen2.5:3b-instruct", "qwen2.5:7b-instruct"]
VERIFIER_EMBEDDER = "bge-m3"


def setup():
    seeds, variations = data.load(data.download(DATA / "gsmplus" / "test.jsonl"))
    return seeds, variations, data.split(len(seeds), DEV_SHARE, SEED)


def store() -> Store:
    return Store(RESULTS / "calls.jsonl")


def cmd_backend(_):
    seeds, variations, _ = setup()
    experiment.answer_all(seeds + variations, store())


def cmd_embed(args):
    seeds, variations, _ = setup()
    RESULTS.mkdir(exist_ok=True)
    for model in args.models:
        experiment.embed_all(model, seeds + variations, embeddings.path(RESULTS, model))


def cmd_verify(args):
    seeds, variations, _ = setup()
    s = store()
    nearest, _, _ = embeddings.nearest(embeddings.path(RESULTS, VERIFIER_EMBEDDER), seeds, variations)
    pairs = []
    for v, n in zip(variations, nearest, strict=True):
        answer = final_answer(s.get("backend", experiment.BACKEND, seeds[n].id)["text"])
        if answer is not None:
            pairs.append((v, seeds[n], answer))
    for model in args.models:
        experiment.verify_all(model, pairs, s)


def cmd_analyze(_):
    from . import analysis

    seeds, variations, dev = setup()
    analysis.main(seeds, variations, dev, store(), RESULTS, EMBEDDERS, VERIFIERS, VERIFIER_EMBEDDER)


def main():
    parser = argparse.ArgumentParser(prog="numcache")
    sub = parser.add_subparsers(required=True)
    sub.add_parser("backend").set_defaults(func=cmd_backend)
    embed = sub.add_parser("embed")
    embed.add_argument("--models", nargs="+", default=EMBEDDERS, choices=EMBEDDERS)
    embed.set_defaults(func=cmd_embed)
    verify = sub.add_parser("verify")
    verify.add_argument("--models", nargs="+", default=VERIFIERS, choices=VERIFIERS)
    verify.set_defaults(func=cmd_verify)
    sub.add_parser("analyze").set_defaults(func=cmd_analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
