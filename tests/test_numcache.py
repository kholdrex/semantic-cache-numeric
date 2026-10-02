from collections import Counter
from fractions import Fraction

from numcache.analysis import pareto
from numcache.backend import final_answer
from numcache.numbers import numbers, same_numbers


def test_numbers_reads_digits_words_and_fractions():
    assert numbers("She eats three eggs and sells 1,200 for $1.6 or 3/4 of a dozen") == Counter(
        {Fraction(3): 1, Fraction(1200): 1, Fraction(8, 5): 1, Fraction(3, 4): 1, Fraction(12): 1}
    )


def test_same_numbers_ignores_wording():
    assert same_numbers("Tom has 3 apples and buys two more.", "After buying 2 more, Tom who had three apples")
    assert not same_numbers("Tom has 3 apples.", "Tom has 4 apples.")


def test_final_answer():
    assert final_answer("so the total is 18.\n#### 18") == 18.0
    assert final_answer("#### $1,250.50") == 1250.5
    assert final_answer("The answer is 42") == 42.0
    assert final_answer("no number") is None
    assert final_answer("#### 2:00 PM\n#### 1400") == 1400.0
    assert final_answer("#### 2\n#### None") is None
    assert final_answer("#### x - 10") is None
    assert final_answer("so #### 3/4") == 0.75


def test_pareto_keeps_lowest_error_for_each_hit_rate():
    points = [(0.5, 0.02), (0.4, 0.01), (0.4, 0.03), (0.2, 0.005), (0.1, 0.02)]
    assert pareto(points) == [(0.1, 0.005), (0.2, 0.005), (0.4, 0.01), (0.5, 0.02)]


def test_verifier_shows_the_exact_answer():
    from numcache.verifier import messages, shown

    assert shown(99076.92) == "99076.92"
    assert "Stored answer: 0.75" in messages("new", "old", 0.75)[1]["content"]
