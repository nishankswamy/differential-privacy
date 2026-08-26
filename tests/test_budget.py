"""The privacy budget: composition, and the hard stop that makes it real."""

import math

import pytest

from privacy.budget import (
    BudgetExceeded,
    PrivacyBudget,
    advanced_composition,
    basic_composition,
)


def test_spends_and_tracks():
    b = PrivacyBudget(total_epsilon=1.0)
    b.spend(0.3)
    b.spend(0.2)
    assert abs(b.spent_epsilon - 0.5) < 1e-9
    assert abs(b.remaining_epsilon - 0.5) < 1e-9


def test_refuses_to_overspend():
    """A budget you can overspend isn't a budget."""
    b = PrivacyBudget(total_epsilon=1.0)
    b.spend(0.8)
    with pytest.raises(BudgetExceeded):
        b.spend(0.5)


def test_rejected_query_costs_nothing():
    b = PrivacyBudget(total_epsilon=1.0)
    b.spend(0.8)
    try:
        b.spend(0.5)
    except BudgetExceeded:
        pass
    assert abs(b.spent_epsilon - 0.8) < 1e-9   # unchanged


def test_exact_budget_is_allowed():
    b = PrivacyBudget(total_epsilon=1.0)
    b.spend(1.0)
    assert b.is_exhausted


def test_history_is_logged():
    b = PrivacyBudget(total_epsilon=1.0)
    b.spend(0.3, label="count")
    b.spend(0.3, label="mean")
    labels = [entry[0] for entry in b.history()]
    assert labels == ["count", "mean"]


def test_basic_composition_sums_epsilons():
    assert basic_composition([0.1, 0.2, 0.3]) == pytest.approx(0.6)


def test_advanced_composition_beats_basic_at_scale():
    """Advanced composition grows ~sqrt(k), so for many queries it permits a far
    smaller total epsilon than basic — the reason it exists."""
    k = 1000
    basic = basic_composition([0.1] * k)
    advanced = advanced_composition(0.1, k, delta_prime=1e-6)
    assert advanced < basic / 3


def test_advanced_composition_overhead_at_small_k():
    """Honest: advanced composition has overhead and isn't always tighter — at
    small k basic can win."""
    basic = basic_composition([0.1] * 10)
    advanced = advanced_composition(0.1, 10, delta_prime=1e-6)
    assert advanced > basic   # basic is tighter here
