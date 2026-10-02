"""Reuse decisions, thresholds chosen on the development seeds, test metrics, costs and figures."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import embeddings
from .backend import final_answer
from .experiment import BACKEND, judgement
from .numbers import same_numbers
from .store import Store

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 13})

SEED = 2026
RESAMPLES = 2000
TARGETS = (0.005, 0.01, 0.02)
MAIN_TARGET = 0.01
VERIFIER_GRID = np.linspace(0, 1, 41)
TOLERANCE = 1e-6
NAMES = {"all-minilm": "all-MiniLM", "bge-m3": "BGE-M3", "qwen2.5:3b-instruct": "3B", "qwen2.5:7b-instruct": "7B"}
KINDS = (
    "problem understanding",
    "distraction insertion",
    "integer-decimal-fraction conversion",
    "numerical substitution",
    "digit expansion",
    "adding operation",
    "reversing operation",
    "critical thinking",
)


def same(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) < TOLERANCE


def frame(seeds, variations, store: Store, results: Path, embedders: list[str], verifiers: list[str],
          verifier_embedder: str) -> dict:
    """Per-request arrays. Answers are re-read from the stored reply texts, so parser fixes apply to old runs."""
    remote = {q.id: store.get("backend", BACKEND, q.id) for q in seeds + variations}
    answer = {i: final_answer(row["text"]) for i, row in remote.items()}
    answerable = np.array([v.answer is not None for v in variations])
    f = {
        "seed": np.array([v.seed for v in variations]),
        "kind": np.array([v.kind for v in variations]),
        "answerable": answerable,
        "miss_correct": np.array([same(answer[v.id], v.answer) for v in variations]),
        "seed_correct": np.array([same(answer[s.id], s.answer) for s in seeds]),
        "remote_seconds": np.array([remote[v.id]["seconds"] for v in variations]),
        "remote_tokens": np.array([remote[v.id]["prompt_tokens"] + remote[v.id]["completion_tokens"]
                                   for v in variations]),
        "embedders": {},
        "verifiers": {},
    }
    for name in embedders:
        nearest, sim, per_text = embeddings.nearest(embeddings.path(results, name), seeds, variations)
        pairs = list(zip(nearest, variations, strict=True))
        f["embedders"][name] = {
            "nearest": nearest,
            "sim": sim,
            "reuse_correct": np.array([same(answer[seeds[n].id], v.answer) for n, v in pairs]),
            "exact": np.array([" ".join(v.text.split()) == " ".join(seeds[n].text.split()) for n, v in pairs]),
            "numbers": np.array([same_numbers(v.text, seeds[n].text) for n, v in pairs]),
            "seconds": per_text,
        }
    nearest = f["embedders"][verifier_embedder]["nearest"]
    for model in verifiers:
        p = np.full(len(variations), np.nan)
        seconds = []
        for k, v in enumerate(variations):
            cached = seeds[nearest[k]]
            if answer[cached.id] is None:
                continue
            row = judgement(store, model, v, cached, answer[cached.id])
            if row is None:
                raise ValueError(f"{model}: no verifier judgement for {v.id} with its current stored answer; "
                                 "run `numcache verify`")
            p[k] = row["p_yes"]
            seconds.append(row["seconds"])
        f["verifiers"][model] = {"p": p, "seconds": float(np.mean(seconds)),
                                 "p95_seconds": float(np.percentile(seconds, 95))}
    return f


def extra_check(f: dict, policy: str, v: float, embedder: str, verifier: str | None) -> np.ndarray:
    """Conditions of a rule besides the similarity threshold."""
    ok = np.ones(len(f["seed"]), dtype=bool)
    if policy in ("numbers", "numbers+verifier"):
        ok &= f["embedders"][embedder]["numbers"]
    if policy in ("verifier", "numbers+verifier"):
        ok &= f["verifiers"][verifier]["p"] >= v
    return ok


def decisions(f: dict, policy: str, tau: float, v: float, embedder: str, verifier: str | None) -> np.ndarray:
    e = f["embedders"][embedder]
    if policy == "exact":
        return e["exact"].copy()
    return (e["sim"] >= tau) & extra_check(f, policy, v, embedder, verifier)


def rates(hit: np.ndarray, f: dict, embedder: str, mask: np.ndarray) -> dict:
    """Hit rate and wrong reuse over all requests; precision over hits; numerical accuracy over answerable requests."""
    good = f["embedders"][embedder]["reuse_correct"][mask]
    h = hit[mask]
    answerable = f["answerable"][mask]
    final = np.where(h, good, f["miss_correct"][mask])
    return {
        "hit_rate": float(h.mean()),
        "wrong_reuse": float((h & ~good).mean()),
        "precision": float(good[h].mean()) if h.any() else None,
        "accuracy": float(final[answerable].mean()) if answerable.any() else None,
    }


def sweep(f: dict, policy: str, v: float, embedder: str, verifier: str | None, mask: np.ndarray):
    """Hit rate and wrong reuse for every attainable similarity threshold, from the strictest to the loosest."""
    e = f["embedders"][embedder]
    idx = np.flatnonzero(mask)
    order = idx[np.argsort(-e["sim"][idx], kind="mergesort")]
    sims = e["sim"][order]
    accepted = extra_check(f, policy, v, embedder, verifier)[order]
    hits = np.cumsum(accepted)
    wrong = np.cumsum(accepted & ~e["reuse_correct"][order])
    last = np.r_[sims[1:] != sims[:-1], True]
    n = len(idx)
    return sims[last], hits[last] / n, wrong[last] / n


def choose(f: dict, policy: str, embedder: str, verifier: str | None, mask: np.ndarray, target: float) -> tuple:
    """Thresholds with the highest development hit rate whose wrong reuse does not exceed the target.

    Every similarity value observed on the development requests is a candidate τ; v runs over VERIFIER_GRID.
    Ties in hit rate are broken by the lower wrong reuse."""
    best = (-1.0, np.inf, None, None)
    for v in VERIFIER_GRID if "verifier" in policy else [0.0]:
        taus, hit, wrong = sweep(f, policy, v, embedder, verifier, mask)
        feasible = np.flatnonzero(wrong <= target)
        if len(feasible) == 0:
            continue
        k = feasible[np.lexsort((wrong[feasible], -hit[feasible]))[0]]
        if (hit[k], -wrong[k]) > (best[0], -best[1]):
            best = (hit[k], wrong[k], float(taus[k]), float(v))
    return best[2], best[3]


def cluster_interval(values_fn, seeds_test: np.ndarray, f: dict, mask: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    idx_by_seed = {s: np.flatnonzero(mask & (f["seed"] == s)) for s in seeds_test}
    out = []
    for _ in range(RESAMPLES):
        chosen = rng.choice(seeds_test, len(seeds_test))
        idx = np.concatenate([idx_by_seed[s] for s in chosen])
        out.append(values_fn(idx))
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]


def policies(f: dict, verifier_embedder: str) -> list[tuple[str, str, str | None]]:
    out = [("exact", verifier_embedder, None)]
    for e in f["embedders"]:
        out += [("similarity", e, None), ("numbers", e, None)]
    for m in f["verifiers"]:
        out += [("verifier", verifier_embedder, m), ("numbers+verifier", verifier_embedder, m)]
    return out


def latency(f: dict, hit: np.ndarray, tau: float, policy: str, embedder: str, verifier: str | None,
            mask: np.ndarray) -> float:
    """Modelled mean seconds per request: embedding, verifier calls above the similarity threshold, remote call on
    a miss, each taken from the measured mean or per-request times."""
    e = f["embedders"][embedder]
    total = np.full(mask.sum(), e["seconds"] if policy != "exact" else 0.0)
    if verifier is not None:
        checked = e["sim"][mask] >= tau
        if policy == "numbers+verifier":
            checked &= e["numbers"][mask]
        total += checked * f["verifiers"][verifier]["seconds"]
    total += ~hit[mask] * f["remote_seconds"][mask]
    return float(total.mean())


def finite(value):
    """JSON-safe copy: NaN and infinities become null."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def kind_summary(f: dict, embedder: str, mask: np.ndarray) -> dict:
    e = f["embedders"][embedder]
    return {
        "reuse_correct": float(e["reuse_correct"][mask].mean()),
        "median_similarity": float(np.median(e["sim"][mask])),
        "remote_correct": rates(np.zeros(len(mask), dtype=bool), f, embedder, mask)["accuracy"],
    }


def main(seeds, variations, dev_seeds: set[int], store: Store, results: Path, embedders: list[str],
         verifiers: list[str], verifier_embedder: str) -> None:
    f = frame(seeds, variations, store, results, embedders, verifiers, verifier_embedder)
    dev = np.isin(f["seed"], list(dev_seeds))
    test = ~dev
    test_seeds = np.unique(f["seed"][test])
    no_cache = np.zeros(len(test), dtype=bool)
    summary = {
        "n_test": int(test.sum()),
        "n_test_answerable": int((test & f["answerable"]).sum()),
        "n_dev": int(dev.sum()),
        "no_cache": rates(no_cache, f, verifier_embedder, test) | {
            "seconds": float(f["remote_seconds"][test].mean()),
            "tokens": float(f["remote_tokens"][test].mean()),
        },
        "seed_accuracy_test": float(f["seed_correct"][test_seeds].mean()),
        "reusable_test": float(f["embedders"][verifier_embedder]["reuse_correct"][test].mean()),
        "embedding_seconds": {e: f["embedders"][e]["seconds"] for e in f["embedders"]},
        "verifier_seconds": {m: {"mean": x["seconds"], "p95": x["p95_seconds"]} for m, x in f["verifiers"].items()},
        "by_kind": {k: kind_summary(f, verifier_embedder, test & (f["kind"] == k)) for k in KINDS},
        "policies": [],
    }
    for target in TARGETS:
        for policy, embedder, verifier in policies(f, verifier_embedder):
            tau, v = (None, None) if policy == "exact" else choose(f, policy, embedder, verifier, dev, target)
            if policy != "exact" and tau is None:
                summary["policies"].append({"policy": policy, "embedder": embedder, "verifier": verifier,
                                            "target": target, "feasible": False})
                continue
            hit = decisions(f, policy, tau or 0.0, v or 0.0, embedder, verifier)
            entry = {"policy": policy, "embedder": embedder, "verifier": verifier, "target": target,
                     "feasible": True, "tau": tau, "v": v, "dev": rates(hit, f, embedder, dev),
                     "test": rates(hit, f, embedder, test),
                     "hits_test": int(hit[test].sum()),
                     "latency": latency(f, hit, tau or 0.0, policy, embedder, verifier, test),
                     "remote_calls": float((~hit[test]).mean()),
                     "remote_tokens": float((~hit[test] * f["remote_tokens"][test]).mean())}
            if target == MAIN_TARGET:
                ok = f["embedders"][embedder]["reuse_correct"]
                entry["ci"] = {
                    "hit_rate": cluster_interval(lambda i, hit=hit: hit[i].mean(), test_seeds, f, test),
                    "wrong_reuse": cluster_interval(lambda i, hit=hit, ok=ok: (hit[i] & ~ok[i]).mean(),
                                                    test_seeds, f, test),
                }
                entry["by_kind"] = {k: rates(hit, f, embedder, test & (f["kind"] == k)) for k in KINDS}
            summary["policies"].append(entry)
    (results / "summary.json").write_text(json.dumps(finite(summary), indent=1, allow_nan=False) + "\n")
    figures = results.parent / "figures"
    figures.mkdir(exist_ok=True)
    similarity_by_kind(f, verifier_embedder, test, figures / "similarity_by_kind.png")
    tradeoff(f, verifier_embedder, test, figures / "tradeoff.png")


def similarity_by_kind(f: dict, embedder: str, mask: np.ndarray, path: Path) -> None:
    """Cosine similarity of each request to its nearest stored question, by kind of variation."""
    e = f["embedders"][embedder]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    data = [e["sim"][mask & (f["kind"] == k)] for k in KINDS]
    kept = [e["reuse_correct"][mask & (f["kind"] == k)].mean() for k in KINDS]
    box = ax.boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True,
                     medianprops={"color": "black"})
    for patch, keep in zip(box["boxes"], kept, strict=True):
        patch.set_facecolor("white" if keep > 0.9 else "0.85" if keep > 0.1 else "0.6")
    ax.set_yticks(range(1, len(KINDS) + 1), [k.replace("integer-decimal-fraction", "integer–decimal–fraction")
                                             for k in KINDS])
    ax.invert_yaxis()
    ax.set_xlabel(f"Cosine similarity, {NAMES[embedder]}")
    ax.grid(axis="x", color="0.9")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def tradeoff(f: dict, embedder: str, mask: np.ndarray, path: Path) -> None:
    """Descriptive frontier on the given requests: the lowest wrong reuse reachable at each hit rate."""
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    name = NAMES[embedder]
    curves = [("similarity", e, None, f"{NAMES[e]} similarity", "black" if e == embedder else "0.6", "--")
              for e in f["embedders"]]
    curves.append(("numbers", embedder, None, f"{name} + number check", "black", "-"))
    styles = (("0.55", "-."), ("black", ":"))
    for m, (color, ls) in zip(f["verifiers"], styles, strict=False):
        curves.append(("verifier", embedder, m, f"{name} + verifier {NAMES[m]}", color, ls))
    for policy, embedder, verifier, label, color, ls in curves:
        points = []
        for v in VERIFIER_GRID if verifier else [0.0]:
            _, hit, wrong = sweep(f, policy, v, embedder, verifier, mask)
            points += list(zip(hit, wrong, strict=True))
        frontier = pareto(points)
        ax.plot([p[0] * 100 for p in frontier], [p[1] * 100 for p in frontier], color=color, ls=ls, lw=1.5,
                label=label)
    ax.axhline(MAIN_TARGET * 100, color="0.6", lw=0.8)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 5)
    ax.set_xlabel("Answered from the cache, %")
    ax.set_ylabel("Wrong answers, %")
    ax.grid(color="0.9")
    ax.legend(fontsize=12, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """For each hit rate, the lowest wrong-reuse rate among settings that reach at least that hit rate."""
    best: dict[float, float] = {}
    for h, w in points:
        best[h] = min(w, best.get(h, np.inf))
    out, low = [], np.inf
    for h in sorted(best, reverse=True):
        low = min(low, best[h])
        out.append((h, low))
    return sorted(out)
