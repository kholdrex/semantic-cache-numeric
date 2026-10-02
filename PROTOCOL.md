# Analysis plan

Fixed on 2026-10-02 before any measurement except a one-question smoke test of the remote model.

## Question
When an information system answers numerical questions (school word problems) with a remote LLM and reuses
stored answers for similar new questions, how often does reuse return a wrong answer, and how much do cheap checks
(an embedding-similarity threshold, a check that the numbers in both questions agree, a small local LLM verifier)
reduce wrong reuse at a given share of avoided remote calls?

## Data
GSM-Plus (revision 3b708db, CC BY-SA 4.0): 1319 GSM8K test questions form the cache; their 10552 variations of
eight kinds are the incoming requests. Reusing a stored answer is correct when the remote model's answer to the
stored question equals the gold answer of the new question (numerical equality, tolerance 1e-6). 20% of the stored
questions (264, seed 2026) and their variations are used to choose thresholds; the rest are the test set.

## Remote model and local components
- Remote: qwen/qwen-2.5-72b-instruct (OpenRouter, provider DeepInfra), temperature 0, answers every stored question
  and every variation (the latter is the answer on a cache miss).
- Embeddings: all-minilm and bge-m3 (Ollama), cosine similarity to the nearest stored question.
- Number check: the multisets of numbers (digits, decimals, fractions, number words up to twenty, tens, hundred,
  thousand, half, twice) extracted from both questions are equal.
- Verifier: qwen2.5:3b-instruct and qwen2.5:7b-instruct (Ollama) see the stored question, the stored answer and
  the new question; score = P(yes)/(P(yes)+P(no)) of the first token. Pairs use the bge-m3 nearest neighbour.

## Policies (decision per request: reuse the nearest stored answer or call the remote model)
P0 no cache; P1 exact text match; P2 similarity ≥ τ; P3 similarity ≥ τ and number check; P4 similarity ≥ τ and
verifier score ≥ v; P5 = P3 and P4 together.

## Thresholds
On the development set: τ and v are chosen to maximise the hit rate subject to wrong reuse ≤ 1% of all requests
(and, as a sensitivity analysis, ≤ 0.5% and ≤ 2%).

## Outcomes on the test set
Hit rate; wrong reuse (wrong reused answers / all requests); precision of reuse; end-to-end accuracy (reused answer
on a hit, remote answer on a miss); by kind of variation; remote calls, tokens and measured latency saved versus
the local cost of embedding and verification; 95% bootstrap intervals over stored questions (clusters).
