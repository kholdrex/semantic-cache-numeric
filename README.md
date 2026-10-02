# Correctness of semantic caching for numerical questions

Code, prompts, all model answers and analysis for the paper *Correctness of semantic caching of language-model
answers to numerical questions* (O. Kholodniak).

An information system answers math word problems with a remote model and may return a stored answer when a new
question is close to a stored one. The cache holds the 1319 GSM8K test problems with the answers of
Qwen2.5-72B-Instruct; the requests are the 10552 GSM-Plus variations of the same problems. Seven of the eight kinds
of variation have a numerical answer, so each reuse decision can be scored exactly; returning a stored number to a
critical-thinking variant (annotated as unanswerable) counts as wrong reuse.

Acceptance rules compared (`analysis.py`):

| Rule | Return the stored answer when |
|---|---|
| exact | the texts are identical |
| similarity | cosine similarity of embeddings (all-MiniLM or BGE-M3) ≥ τ |
| numbers | similarity ≥ τ and both questions mention the same multiset of numbers (`numbers.py`, a lexical heuristic) |
| verifier | similarity ≥ τ and a local model (Qwen2.5-3B or 7B) gives the stored answer a score ≥ v: P(yes) / (P(yes) + P(no)) over the ten most likely first tokens (`verifier.py`, `local.py`) |
| numbers+verifier | both checks pass |

Thresholds are chosen on 20% of the stored problems (seed 2026) for at most 0.5%, 1% or 2% wrong answers among all
requests (every observed development similarity is a candidate τ; v in steps of 0.025) and applied unchanged to the
remaining problems; intervals are bootstrapped over stored problems.

## Data and models

- [GSM-Plus](https://huggingface.co/datasets/qintongli/GSM-Plus), revision `3b708db`, CC BY-SA 4.0 (downloaded on
  first use).
- Remote model: `qwen/qwen-2.5-72b-instruct` on OpenRouter, provider DeepInfra, temperature 0.
- Local (Ollama 0.34.3, Apple M1 Pro): embeddings `all-minilm` (digest 1b226e2802db) and `bge-m3` (790764642607),
  F16; verifiers `qwen2.5:3b-instruct` (357c53fb659c) and `qwen2.5:7b-instruct` (845dbda0ea48), Q4_K_M.

## Layout

```
src/numcache/   data, remote and local clients, call store, number check, verifier prompt, runner, analysis
results/        calls.jsonl (remote replies and verifier scores with times and tokens),
                embeddings_*.npz (normalised embeddings of all questions), summary.json
figures/        figures of the paper (similarity_by_kind.png, tradeoff.png)
tests/          unit tests
PROTOCOL.md     analysis plan written before the measurements; protocol_freeze.sha256 holds the hashes at that time
CHANGELOG.md    changes after the plan
```

## Usage

Python 3.10+ and [uv](https://docs.astral.sh/uv/). Install from the checkout (editable), because the commands read
`data/` and `results/` relative to it. `requirements.lock.txt` lists the exact package versions used.

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest
uv run numcache analyze      # results/summary.json and figures/ from the stored calls
```

New measurements: `uv run numcache backend` (needs `OPENROUTER_API_KEY`), then `uv run numcache embed` and
`uv run numcache verify` (need Ollama with the four models). Each step resumes from `results/`; to start a fresh run,
move `results/` aside first.

## License

Code: MIT. The stored calls contain question ids and model replies; the GSM-Plus questions are downloaded from the
dataset (CC BY-SA 4.0). See `THIRD_PARTY_NOTICES.md`.
