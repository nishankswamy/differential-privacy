"""Differential privacy: the Laplace and Gaussian mechanisms, done correctly.

Differential privacy makes a promise k-anonymity can't: the output of a query is
almost the same whether or not *any one individual* is in the dataset. So no
attacker — however much side information they hold — can learn much about a
specific person from the result. It's a property of the *mechanism*, not of the
data, which is why it survives linkage, homogeneity, and composition.

The mechanism: add calibrated random noise to the true answer. The amount of
noise is set by two things, and getting either wrong silently breaks the whole
guarantee:

**Sensitivity (Δf)** — the most the query's answer can change if one person is
added or removed. A count has sensitivity 1 (one person changes it by 1). A sum
of values in [0, C] has sensitivity C (one person can move it by at most C). Get
sensitivity too low and you add too little noise and the guarantee is a lie. This
is the single most common way real DP implementations are broken.

**Epsilon (ε)** — the privacy budget. Smaller ε = more noise = stronger privacy.
ε is not a probability or a percentage; it bounds a *ratio of likelihoods*, which
is what makes it composable and what makes it hard to explain (see the README).

    Laplace mechanism:  answer + Laplace(scale = Δf / ε)          gives ε-DP
    Gaussian mechanism: answer + Normal(σ from Δf, ε, δ)          gives (ε,δ)-DP
"""

from __future__ import annotations

import math

import numpy as np


class PrivacyError(Exception):
    pass


def laplace_mechanism(true_value: float, sensitivity: float, epsilon: float,
                      rng: np.random.Generator | None = None) -> float:
    """ε-differentially private answer via additive Laplace noise.

    scale = sensitivity / epsilon. Larger sensitivity or smaller epsilon both
    mean more noise. The noise is symmetric and unbiased, so averaging many DP
    answers converges to the truth — which is exactly why a privacy *budget* is
    needed to stop an attacker doing that."""
    if epsilon <= 0:
        raise PrivacyError("epsilon must be positive")
    if sensitivity < 0:
        raise PrivacyError("sensitivity must be non-negative")
    rng = rng or np.random.default_rng()
    scale = sensitivity / epsilon
    return true_value + rng.laplace(0, scale)


def gaussian_mechanism(true_value: float, sensitivity: float, epsilon: float,
                       delta: float, rng: np.random.Generator | None = None) -> float:
    """(ε,δ)-differentially private answer via additive Gaussian noise.

    δ is the small probability the ε guarantee fails outright — it must be much
    smaller than 1/n (or an attacker exploits the failure case), so this refuses
    a δ that's too loose. Gaussian noise composes better than Laplace over many
    queries (the noise adds in quadrature, not linearly), which is why it's
    preferred when a budget is spread across a large query workload."""
    if epsilon <= 0:
        raise PrivacyError("epsilon must be positive")
    if not (0 < delta < 1):
        raise PrivacyError("delta must be in (0, 1)")
    rng = rng or np.random.default_rng()
    # Classic Gaussian mechanism calibration (Dwork & Roth).
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    return true_value + rng.normal(0, sigma)


# --- sensitivity helpers: the part that's easy to get wrong ----------------

def count_sensitivity() -> float:
    """A count changes by exactly 1 when one person is added or removed."""
    return 1.0


def sum_sensitivity(lower: float, upper: float) -> float:
    """A sum of values clamped to [lower, upper] changes by at most (upper-lower)
    when one person is added or removed. This is why DP sums REQUIRE bounds —
    without a clamp, one person with an arbitrarily large value gives unbounded
    sensitivity and no finite noise is enough."""
    if upper < lower:
        raise PrivacyError("upper must be >= lower")
    return float(upper - lower)


def mean_sensitivity(lower: float, upper: float, n: int) -> float:
    """A mean's sensitivity is the value range divided by n — one person shifts
    the mean by at most (range)/n. Note it depends on n, which must itself be
    public or separately privatised."""
    if n <= 0:
        raise PrivacyError("n must be positive")
    return (upper - lower) / n


# --- private query wrappers ------------------------------------------------

def private_count(data: np.ndarray, predicate, epsilon: float,
                  rng=None) -> float:
    """DP count of rows satisfying `predicate`."""
    true_count = int(np.sum(predicate(data)))
    return laplace_mechanism(true_count, count_sensitivity(), epsilon, rng)


def private_sum(values: np.ndarray, lower: float, upper: float, epsilon: float,
                rng=None) -> float:
    """DP sum. Values are CLAMPED to [lower, upper] first — the clamp is not
    optional, it's what makes the sensitivity finite."""
    clamped = np.clip(values, lower, upper)
    return laplace_mechanism(float(clamped.sum()),
                             sum_sensitivity(lower, upper), epsilon, rng)


def private_mean(values: np.ndarray, lower: float, upper: float, epsilon: float,
                 rng=None) -> float:
    clamped = np.clip(values, lower, upper)
    return laplace_mechanism(float(clamped.mean()),
                             mean_sensitivity(lower, upper, len(values)), epsilon, rng)
