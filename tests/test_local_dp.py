"""Local DP (randomised response): unbiased after correction, and measurably
noisier than central DP."""

import numpy as np

from privacy.local_dp import (
    estimate_rate,
    local_vs_central_error,
    randomised_response,
)


def test_randomised_response_is_unbiased_after_correction():
    rng = np.random.default_rng(0)
    true_bits = (rng.random(20000) < 0.3).astype(int)
    responses = randomised_response(true_bits, epsilon=1.0, rng=rng)
    assert abs(estimate_rate(responses, 1.0) - 0.3) < 0.02


def test_higher_epsilon_less_noise():
    rng = np.random.default_rng(0)
    true_bits = (rng.random(20000) < 0.4).astype(int)

    errs = {}
    for eps in (0.5, 3.0):
        estimates = [estimate_rate(randomised_response(true_bits, eps, rng), eps)
                     for _ in range(50)]
        errs[eps] = np.std(estimates)
    assert errs[0.5] > errs[3.0]


def test_satisfies_the_dp_guarantee():
    """The actual privacy property, measured: the probability of reporting 'yes'
    given a true 'yes' vs given a true 'no' differs by at most a factor of e^ε.
    That bounded likelihood ratio IS local ε-DP — no attacker can distinguish
    the two cases by more than that factor, whatever they already know."""
    import math

    rng = np.random.default_rng(0)
    epsilon = 1.0
    n = 100000

    yes_reports = randomised_response(np.ones(n, dtype=int), epsilon, rng).mean()
    no_reports = randomised_response(np.zeros(n, dtype=int), epsilon, rng).mean()

    ratio = yes_reports / no_reports
    assert ratio <= math.exp(epsilon) + 0.05   # within the ε bound (plus noise)


def test_local_is_noisier_than_central():
    """The measured cost of trusting no curator."""
    rng = np.random.default_rng(0)
    true_bits = (rng.random(2000) < 0.1).astype(int)
    result = local_vs_central_error(true_bits, epsilon=1.0, rng=rng)
    assert result["ratio"] > 3    # local several times worse at the same epsilon
