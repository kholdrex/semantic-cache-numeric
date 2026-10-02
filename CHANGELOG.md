# Changes after the analysis plan was fixed

The plan (`PROTOCOL.md`) was written on 2026-10-02 before any measurement except a one-question test of the remote
model. The stored calls were not changed. Changes to the code afterwards:

- The runner retries calls that failed after all attempts in further passes and runs 32 requests in parallel.
- The OpenRouter key is read from the `OPENROUTER_API_KEY` environment variable.
- Added a descriptive figure of similarity by kind of variation; the trade-off figure shows the range up to 20% of
  requests answered from the cache, in greyscale.
- `numbers.py` (the number check) was written after the plan; besides the words listed there it reads "double",
  "triple" and "dozen".
- After review: answers are re-read from the stored reply texts with a parser that handles fractions and reads the
  number after the last `####` marker (no answer if that marker is not followed by a number); gold answers such as
  `3/4` are parsed as fractions. Compared with the answers stored at measurement time, this changed 18 parsed
  answers: one cached answer (`seed/1001`) and 17 remote answers to variations. Accuracy is reported for requests
  with a numerical answer only.
- `seed/1001` asks for a clock time; the remote reply ends with `#### 2:00 PM` and then `#### 1400`, so its answer is
  read as 1400 and scored as wrong against the gold answer 2. Clock times are not normalised.
- Verifier prompts showed the stored answer with six significant digits (`f"{a:g}"`), which shortened one answer
  (`seed/119`, 99076.92). Prompts now use ten significant digits, verifier records store the text they showed
  (`shown`), and judgements whose shown answer differs from the current one are repeated: 24 pairs for `seed/119`
  and 8 for `seed/1001`, for each verifier. Analysis refuses to run if a judgement is missing or stale.
- The threshold search covers every similarity value observed on the development requests instead of 101 quantiles;
  ties in hit rate are broken by the lower wrong reuse. Fig. 2 is a descriptive frontier on the test set, with one
  point per hit rate.
- Remote call times now include failed attempts and waiting between retries (new runs only; stored times cover the
  successful attempt).
- Models are passed from `cli.py` to the analysis; nearest-neighbour search lives in `embeddings.py`; policies report
  the mean number of remote tokens per request.
- `data.py` and `verifier.py` were reformatted for line length after the freeze. The repository has no history from
  before the freeze, so `protocol_freeze.sha256` documents the hashes at that time but the frozen files themselves
  are not included.
