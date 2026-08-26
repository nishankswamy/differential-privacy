"""Day 22: the utility/privacy tradeoff, measured, with a plain-language
recommendation.

    python evaluate_main.py
"""

from __future__ import annotations

import numpy as np

from privacy.dp import private_count, private_mean, private_sum
from privacy.generate import generate_population
from privacy.local_dp import local_vs_central_error


def utility_curve(pop, epsilons, trials=500, rng=None) -> None:
    """Relative error of three query types across a range of epsilon. The curve
    every DP deployment has to reason about: how much accuracy does a given
    privacy level cost?"""
    rng = rng or np.random.default_rng(0)
    ages = (2025 - pop.birth_year).to_numpy().astype(float)

    true_count = int((pop.condition == "cancer").sum())
    true_mean = ages.mean()
    true_sum = ages.sum()

    print(f"{'epsilon':>8} {'count err%':>11} {'mean err%':>11} {'sum err%':>11}")
    print("-" * 44)
    for eps in epsilons:
        counts = [private_count(pop, lambda d: (d.condition == "cancer").values, eps, rng)
                  for _ in range(trials)]
        means = [private_mean(ages, 20, 90, eps, rng) for _ in range(trials)]
        sums = [private_sum(ages, 20, 90, eps, rng) for _ in range(trials)]

        ce = np.mean(np.abs(np.array(counts) - true_count)) / true_count * 100
        me = np.mean(np.abs(np.array(means) - true_mean)) / true_mean * 100
        se = np.mean(np.abs(np.array(sums) - true_sum)) / true_sum * 100
        print(f"{eps:>8} {ce:>10.1f}% {me:>10.2f}% {se:>10.3f}%")


def main() -> None:
    pop = generate_population()
    rng = np.random.default_rng(0)

    print("Utility vs privacy — relative error by query type\n")
    utility_curve(pop, [0.01, 0.1, 0.5, 1.0, 5.0], rng=rng)

    print("""
  Reading the curve: at ε=0.1 (strong privacy) a count is off by ~6%, a mean
  by under 1%, and a sum by even less. Same noise, wildly
  different relative impact — because a count of 159 is small and a sum over
  2,000 ages is large, the same absolute Laplace noise is a big fraction of one
  and a rounding error on the other. The lesson: sensitivity and the true value's
  scale together decide utility, not epsilon alone.
""")

    print("=" * 60)
    print("Local vs central DP — the cost of trusting no curator")
    print("=" * 60 + "\n")
    has_cancer = (pop.condition == "cancer").to_numpy().astype(int)
    print(f"{'epsilon':>8} {'local err':>11} {'central err':>13} {'local penalty':>14}")
    print("-" * 48)
    for eps in (0.5, 1.0, 2.0, 5.0):
        r = local_vs_central_error(has_cancer, eps, rng=rng)
        print(f"{eps:>8} {r['local_error']:>11.4f} {r['central_error']:>13.4f} "
              f"{r['ratio']:>12.0f}x")

    print("""
  Local DP (randomised response) is 10-20x noisier than central DP at the same
  epsilon, because it perturbs every individual answer instead of the aggregate.
  You pay that penalty to avoid trusting a curator with raw data — which is why
  Apple/Google use it on-device but a hospital with a secure warehouse uses
  central DP.
""")

    print("=" * 60)
    print("The recommendation, in plain language")
    print("=" * 60)
    print("""
  - ε ≤ 1 is meaningful privacy; ε ≥ 10 is privacy theatre (little noise).
    Pick ε from what you're protecting, not from what keeps the data useful.
  - Budget it. Every query spends ε; when it's gone, stop answering. A DP
    system that answers unlimited queries provides no privacy — the noise
    averages out (this repo's own DP-mean-over-2000-queries demo proves it).
  - Use central DP if you can secure the raw data; local DP only when you
    genuinely cannot trust the collector, and accept the 10-20x accuracy hit.
  - Clamp before you sum. Unbounded values = unbounded sensitivity = no valid
    noise. The clamp is the guarantee, not an optimisation.
""")


if __name__ == "__main__":
    main()
