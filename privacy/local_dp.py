"""Local differential privacy: randomised response.

Central DP (the Laplace/Gaussian mechanisms) trusts a curator to hold the raw
data and add noise to query answers. Local DP trusts no one: each person
randomises their *own* answer before it ever leaves their device, so the
collector never sees a true value. This is what Apple and Google deploy on
phones — the server learns aggregate statistics but no individual's real answer.

The classic mechanism is **randomised response** (Warner, 1965), predating DP by
decades: to survey a sensitive yes/no question, each respondent tells the truth
with probability p and reports the *opposite* of their true answer with
probability 1-p. The collector can't know whether any single answer was truthful
or flipped — that's the plausible deniability — but the *fraction* of yeses can
be corrected back to the true rate, because p is known.

Calibrating p = e^ε / (e^ε + 1) makes this exactly ε-locally-DP: the ratio of
P(report yes | true yes) to P(report yes | true no) is p/(1-p) = e^ε, the DP
bound. (The 'flip the bit' variant matters: a 'randomise to a uniform coin'
variant gives a different, weaker ratio for the same p — a subtle calibration
trap.)

The price is steep, and measuring it is the point: local DP needs far more people
for the same accuracy, because every single answer is noisy, not just the
aggregate. Central DP adds noise once; local DP adds it n times.
"""

from __future__ import annotations

import math

import numpy as np


def randomised_response(true_bits: np.ndarray, epsilon: float,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Each person reports their bit truthfully with probability p, flipped with
    probability 1-p. ε-locally-DP with p = e^ε / (e^ε + 1).

    At ε→∞, p→1 (everyone truthful, no privacy); at ε=0, p=0.5 (a coin flip,
    perfect privacy, zero signal). The likelihood ratio P(yes|1)/P(yes|0) =
    p/(1-p) = e^ε is the DP guarantee."""
    rng = rng or np.random.default_rng()
    p_truth = math.exp(epsilon) / (math.exp(epsilon) + 1)

    tell_truth = rng.random(len(true_bits)) < p_truth
    flipped = 1 - true_bits
    return np.where(tell_truth, true_bits, flipped)


def estimate_rate(responses: np.ndarray, epsilon: float) -> float:
    """Correct the observed 'yes' rate back to the true rate, inverting the known
    randomisation. Without this the estimate is biased toward 0.5."""
    p_truth = math.exp(epsilon) / (math.exp(epsilon) + 1)
    observed = responses.mean()
    # observed = p*true_rate + (1-p)*(1-true_rate); solve for true_rate.
    denom = 2 * p_truth - 1
    if abs(denom) < 1e-12:
        return 0.5   # epsilon ~ 0: no signal recoverable
    return (observed - (1 - p_truth)) / denom


def local_vs_central_error(true_bits: np.ndarray, epsilon: float, trials: int = 200,
                           rng: np.random.Generator | None = None) -> dict:
    """Compare the estimation error of local RR against central Laplace on the
    same count query — the measurement that shows local DP's cost."""
    from .dp import laplace_mechanism

    rng = rng or np.random.default_rng()
    n = len(true_bits)
    true_rate = true_bits.mean()

    local_errors, central_errors = [], []
    for _ in range(trials):
        # local
        responses = randomised_response(true_bits, epsilon, rng)
        local_errors.append(abs(estimate_rate(responses, epsilon) - true_rate))
        # central: DP count / n
        dp_count = laplace_mechanism(true_bits.sum(), 1.0, epsilon, rng)
        central_errors.append(abs(dp_count / n - true_rate))

    return {
        "epsilon": epsilon,
        "local_error": float(np.mean(local_errors)),
        "central_error": float(np.mean(central_errors)),
        "ratio": float(np.mean(local_errors) / np.mean(central_errors)),
    }
