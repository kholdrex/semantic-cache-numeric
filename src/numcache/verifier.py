"""Prompt of the local model that decides whether a cached answer can be returned for a new question."""

SYSTEM = (
    "A tutoring system keeps answers to questions it has already solved. Decide whether the stored answer is also "
    "the correct answer to the new question. Questions that differ in any number, quantity or operation usually "
    "need a different answer. Reply with yes or no only."
)


def shown(answer: float) -> str:
    """The stored answer as written in the prompt."""
    return f"{answer:.10g}"


def messages(new: str, cached: str, cached_answer: float) -> list[dict]:
    user = (
        f"Stored question: {cached}\nStored answer: {shown(cached_answer)}\n\n"
        f"New question: {new}\n\nCan the stored answer be returned?"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
