"""The privacy budget — the part people forget, which quietly voids the guarantee.

Every DP query spends some epsilon. The catch: **privacy loss composes**. Ask
the same ε-DP query twice and you've spent 2ε, because an attacker can average
the two noisy answers and halve the noise. Ask it a thousand times and the noise
averages away entirely — the earlier demo showed a DP mean converging to the true
value over 2000 queries. Without a budget that *stops* answering, DP is
security theatre.

Two composition rules, both here:

**Basic (sequential) composition:** total privacy loss is the sum of the epsilons.
Simple, always valid, pessimistic. k queries at ε each = kε.

**Advanced composition:** over many queries the loss grows more like √k·ε rather
than k·ε (the noise adds in quadrature). This lets you answer far more queries for
the same total budget, at the cost of a small δ. It's why real systems track an
(ε, δ) budget, not just ε.

The tracker refuses any query that would exceed the budget — a hard stop, because
a budget you can overspend isn't a budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    pass


@dataclass
class PrivacyBudget:
    """Tracks epsilon spent against a fixed total. Refuses to overspend."""
    total_epsilon: float
    total_delta: float = 0.0
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    _log: list = field(default_factory=list)

    def spend(self, epsilon: float, delta: float = 0.0, label: str = "") -> None:
        """Charge a query against the budget (basic composition). Raises before
        spending if it would exceed the total — so a rejected query costs
        nothing."""
        if self.spent_epsilon + epsilon > self.total_epsilon + 1e-12:
            raise BudgetExceeded(
                f"query needs ε={epsilon}, only "
                f"{self.remaining_epsilon:.4f} of {self.total_epsilon} left")
        if self.spent_delta + delta > self.total_delta + 1e-12:
            raise BudgetExceeded(f"query needs δ={delta}, budget exhausted")
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self._log.append((label or f"query {len(self._log)+1}", epsilon, delta))

    @property
    def remaining_epsilon(self) -> float:
        return self.total_epsilon - self.spent_epsilon

    @property
    def is_exhausted(self) -> bool:
        return self.remaining_epsilon <= 1e-12

    def history(self) -> list:
        return list(self._log)


def basic_composition(epsilons: list[float]) -> float:
    """Total ε under sequential composition: just the sum."""
    return sum(epsilons)


def advanced_composition(epsilon: float, k: int, delta_prime: float) -> float:
    """Total ε for k queries each ε-DP, under advanced composition, at the cost
    of an extra δ'. Grows ~√k instead of k — the whole reason advanced
    composition is worth the added δ.

    (Dwork–Roth theorem 3.20 form.)"""
    if k <= 0:
        return 0.0
    return (math.sqrt(2 * k * math.log(1 / delta_prime)) * epsilon
            + k * epsilon * (math.exp(epsilon) - 1))
