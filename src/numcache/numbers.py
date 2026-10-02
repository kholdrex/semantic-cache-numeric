"""Numbers mentioned in a question, as a multiset, for the check that two questions use the same quantities."""

import re
from collections import Counter
from fractions import Fraction

WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000, "half": Fraction(1, 2),
    "twice": 2, "double": 2, "triple": 3, "dozen": 12,
}
TOKEN_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?|[a-z]+")


def numbers(text: str) -> Counter:
    found = Counter()
    for token in TOKEN_RE.findall(text.lower()):
        if token[0].isdigit():
            found[Fraction(token.replace(",", ""))] += 1
        elif token in WORDS:
            found[Fraction(WORDS[token])] += 1
    return found


def same_numbers(a: str, b: str) -> bool:
    return numbers(a) == numbers(b)
