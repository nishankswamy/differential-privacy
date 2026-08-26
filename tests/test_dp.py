"""The DP mechanisms: unbiased, correctly calibrated, and refusing bad inputs."""

import numpy as np
import pytest

from privacy.dp import (
    PrivacyError,
    gaussian_mechanism,
    laplace_mechanism,
    private_count,
    private_mean,
    private_sum,
    sum_sensitivity,
)


def test_laplace_is_unbiased():
    """Averaged over many runs, DP noise cancels — the estimate converges to the
    truth. (Which is exactly why a budget is needed.)"""
    rng = np.random.default_rng(0)
    est = [laplace_mechanism(100, 1.0, 1.0, rng) for _ in range(20000)]
    assert abs(np.mean(est) - 100) < 0.5


def test_smaller_epsilon_means_more_noise():
    rng = np.random.default_rng(0)
    tight = np.std([laplace_mechanism(100, 1, 0.1, rng) for _ in range(5000)])
    loose = np.std([laplace_mechanism(100, 1, 10, rng) for _ in range(5000)])
    assert tight > loose * 10       # 100x epsilon ratio -> ~100x noise ratio


def test_larger_sensitivity_means_more_noise():
    rng = np.random.default_rng(0)
    low = np.std([laplace_mechanism(0, 1, 1, rng) for _ in range(5000)])
    high = np.std([laplace_mechanism(0, 10, 1, rng) for _ in range(5000)])
    assert high > low * 5


def test_epsilon_must_be_positive():
    with pytest.raises(PrivacyError):
        laplace_mechanism(1, 1, 0)
    with pytest.raises(PrivacyError):
        laplace_mechanism(1, 1, -1)


def test_gaussian_requires_valid_delta():
    with pytest.raises(PrivacyError):
        gaussian_mechanism(1, 1, 1, delta=0)
    with pytest.raises(PrivacyError):
        gaussian_mechanism(1, 1, 1, delta=1)


def test_sum_sensitivity_is_the_value_range():
    assert sum_sensitivity(0, 100) == 100
    with pytest.raises(PrivacyError):
        sum_sensitivity(100, 0)     # upper < lower


def test_private_sum_clamps():
    """The clamp is the guarantee: one outlier can't blow up sensitivity."""
    rng = np.random.default_rng(0)
    values = np.array([50.0] * 100 + [1e9])   # one huge outlier
    # With clamping to [0,100], the outlier contributes at most 100.
    est = np.mean([private_sum(values, 0, 100, 1.0, rng) for _ in range(200)])
    assert est < 100 * 101 * 1.5    # bounded, not dominated by 1e9


def test_private_count_matches_truth_on_average():
    rng = np.random.default_rng(0)
    data = np.arange(1000)
    est = np.mean([private_count(data, lambda d: d < 300, 1.0, rng) for _ in range(500)])
    assert abs(est - 300) < 5


def test_private_mean_matches_truth_on_average():
    rng = np.random.default_rng(0)
    values = np.random.default_rng(1).uniform(20, 80, 1000)
    est = np.mean([private_mean(values, 20, 80, 1.0, rng) for _ in range(500)])
    assert abs(est - values.mean()) < 1.0
