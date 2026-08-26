"""Synthetic personal data, generated so we can safely attack it.

The whole project attacks a dataset, so it must be one nobody real is in. This
builds a population with the quasi-identifiers that matter for re-identification
— ZIP, birth date, sex — plus a sensitive attribute (a medical condition) that
the attacker wants to learn.

The famous result being reproduced is Latanya Sweeney's: 87% of Americans are
uniquely identified by the combination of ZIP + birth date + sex. None of those
is identifying alone; together they are a fingerprint. We build that structure in
deliberately so the linkage attack on Day 20 works on data we made, not on
anyone real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_population(n: int = 2000, seed: int = 20) -> pd.DataFrame:
    """A 'hospital' dataset: quasi-identifiers + a sensitive condition.

    ZIPs are drawn from a modest set and birth dates span decades, so that the
    (zip, birthdate, sex) triple is unique for most rows — reproducing the
    real-world uniqueness that makes quasi-identifiers dangerous.
    """
    rng = np.random.default_rng(seed)

    zips = [f"021{d:02d}" for d in range(20)]  # 20 ZIP codes
    people = pd.DataFrame({
        "name": [f"person_{i}" for i in range(n)],  # the identity to protect
        "zip": rng.choice(zips, n),
        # Birth date as an integer yyyymmdd across ~50 years — high cardinality.
        "birth_year": rng.integers(1950, 2005, n),
        "birth_month": rng.integers(1, 13, n),
        "birth_day": rng.integers(1, 29, n),
        "sex": rng.choice(["M", "F"], n),
    })
    people["birthdate"] = (people.birth_year * 10000
                           + people.birth_month * 100 + people.birth_day)

    # The sensitive attribute CORRELATES with the quasi-identifiers, as real
    # medical data does — older people get more chronic conditions. This is what
    # makes k-anonymity's homogeneity hole real: a group that's k-anonymous on
    # identity can still be uniform on the condition, leaking it.
    age = 2025 - people["birth_year"]
    condition = []
    for a in age:
        if a > 65:
            p_cond = [0.15, 0.30, 0.35, 0.05, 0.15]   # older: chronic-heavy
        elif a > 45:
            p_cond = [0.45, 0.20, 0.20, 0.08, 0.07]
        else:
            p_cond = [0.75, 0.05, 0.05, 0.13, 0.02]   # younger: mostly none/asthma
        condition.append(rng.choice(
            ["none", "diabetes", "hypertension", "asthma", "cancer"], p=p_cond))
    people["condition"] = condition

    # A specialty-clinic cohort: a cluster of people sharing one ZIP and a narrow
    # age band who ALL have the same diagnosis (an oncology clinic's patients).
    # This is how homogeneity leaks arise in reality — and it survives
    # k-anonymity, because the group is large enough to be k-anonymous yet
    # uniform on the sensitive value.
    cohort = rng.choice(people.index, size=18, replace=False)
    people.loc[cohort, "zip"] = "02199"          # a ZIP only this clinic uses
    people.loc[cohort, "birth_year"] = 1950      # same age band
    people.loc[cohort, "sex"] = "F"
    people.loc[cohort, "condition"] = "cancer"   # all the same diagnosis
    people.loc[cohort, "birthdate"] = (1950 * 10000
                                       + people.loc[cohort, "birth_month"] * 100
                                       + people.loc[cohort, "birth_day"])
    return people


def public_voter_roll(population: pd.DataFrame, coverage: float = 0.6,
                      seed: int = 21) -> pd.DataFrame:
    """A second, public dataset the attacker already has.

    In Sweeney's attack this was a voter registration roll bought for $20: it
    carries names *and* the same quasi-identifiers, but not the sensitive
    condition. Joining it to the 'anonymised' hospital data on the shared
    quasi-identifiers is what re-attaches names to conditions.
    """
    rng = np.random.default_rng(seed)
    sample = population.sample(frac=coverage, random_state=seed).copy()
    # The voter roll knows who you are and your quasi-identifiers, not your
    # medical condition.
    return sample[["name", "zip", "birthdate", "sex"]].reset_index(drop=True)


QUASI_IDENTIFIERS = ["zip", "birthdate", "sex"]
SENSITIVE = "condition"


if __name__ == "__main__":
    pop = generate_population()
    combo = pop.groupby(QUASI_IDENTIFIERS).size()
    unique = (combo == 1).sum()
    print(f"{len(pop)} people")
    print(f"{unique} of {len(combo)} quasi-identifier combinations are unique "
          f"({unique / len(pop) * 100:.0f}% of people uniquely identified)")
