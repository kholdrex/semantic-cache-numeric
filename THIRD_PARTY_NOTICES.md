# Third-party data

The questions and gold answers come from GSM-Plus (https://huggingface.co/datasets/qintongli/GSM-Plus, revision
3b708db), licensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), which builds on the GSM8K
test set (https://github.com/openai/grade-school-math, MIT). They are downloaded on first use and are not part of this
repository. `results/calls.jsonl` stores question ids and the replies of Qwen2.5-72B-Instruct and the Qwen2.5
verifiers; `results/embeddings_*.npz` stores embeddings of the questions. The MIT licence of this repository covers
the code.
