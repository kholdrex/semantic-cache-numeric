"""The remote model that answers requests on a cache miss: step-by-step solution ending with '#### <number>'."""

import re

from .data import number

NUMBER = r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?|-?\.\d+"
FINAL_RE = re.compile(rf"####\s*\$?\s*({NUMBER})")
NUMBER_RE = re.compile(NUMBER)

SYSTEM = (
    "Solve the math word problem step by step. End your reply with a final line of the form '#### <number>' "
    "that contains only the numerical answer."
)


def messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def final_answer(reply: str) -> float | None:
    """Number that follows the last '####' marker (None if it is not a number), or the last number in the reply if
    there is no marker."""
    if "####" in reply:
        match = FINAL_RE.match("####" + reply.rsplit("####", 1)[1])
        return number(match.group(1)) if match else None
    found = NUMBER_RE.findall(reply)
    return number(found[-1]) if found else None
