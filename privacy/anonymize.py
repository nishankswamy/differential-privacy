"""k-anonymity, and the demonstration that it isn't enough.

k-anonymity says: generalise the quasi-identifiers until every record is
indistinguishable from at least k-1 others. If everyone in a group shares the
same generalised ZIP/age/sex, an attacker linking on those fields can't pin a
record to one person — only to a group of k.

It's the intuitive first answer to anonymisation, and it has two well-known
holes this module demonstrates:

1. **It's fragile.** Achieve k-anonymity by generalising too little and unique
   records remain, re-identifiable by linkage. The Day 20 attack shows this.
2. **Homogeneity attack.** Even at k-anonymity, if every record in a group shares
   the same *sensitive* value, you learn it without identifying the individual.
   k-anonymity protects identity, not the secret — which is usually what you
   actually wanted to protect. (This is what l-diversity was invented to fix.)

The point of building it is to show *why* differential privacy exists: not because
k-anonymity is stupid, but because it protects the wrong thing and breaks under
composition.
"""

from __future__ import annotations

import pandas as pd

from .generate import QUASI_IDENTIFIERS, SENSITIVE


def generalise(df: pd.DataFrame, zip_digits: int = 5, age_bucket: int = 1) -> pd.DataFrame:
    """Blur the quasi-identifiers. Fewer ZIP digits and wider birth-year buckets
    mean coarser groups — more privacy, less utility."""
    out = df.copy()
    out["zip"] = out["zip"].str[:zip_digits].str.ljust(5, "*")
    out["birth_year"] = (out["birthdate"] // 10000)
    out["birth_year"] = (out["birth_year"] // age_bucket) * age_bucket
    # Drop the fine-grained birthdate; keep only the bucketed year.
    out = out.drop(columns=["birthdate", "birth_month", "birth_day"], errors="ignore")
    return out


def k_anonymity_level(df: pd.DataFrame, quasi=None) -> int:
    """The smallest equivalence-class size = the k actually achieved. k=1 means
    at least one record is unique and re-identifiable."""
    quasi = quasi or ["zip", "birth_year", "sex"]
    return int(df.groupby(quasi).size().min())


def enforce_k_anonymity(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Generalise progressively until every group has >= k records, suppressing
    whatever still can't reach k.

    This is deliberately the *naive* enforcement — coarsen ZIP, then widen age
    buckets — so the tradeoff (privacy costs a lot of utility) is visible."""
    for zip_digits, age_bucket in [(5, 1), (4, 1), (4, 5), (4, 10), (3, 10), (2, 20)]:
        generalised = generalise(df, zip_digits, age_bucket)
        quasi = ["zip", "birth_year", "sex"]
        sizes = generalised.groupby(quasi).size()
        if sizes.min() >= k:
            return generalised
        # Suppress groups still below k, keep the rest at this generalisation.
        good_groups = sizes[sizes >= k].index
        kept = generalised.set_index(quasi).loc[
            generalised.set_index(quasi).index.isin(good_groups)
        ].reset_index()
        if len(kept) >= 0.5 * len(df) and k_anonymity_level(kept) >= k:
            return kept
    return generalise(df, 2, 20)  # maximal generalisation as a last resort


def homogeneity_leak(df: pd.DataFrame, quasi=None) -> pd.DataFrame:
    """Find k-anonymous groups where the sensitive value is identical for
    everyone — you learn the secret without identifying the person."""
    quasi = quasi or ["zip", "birth_year", "sex"]
    leaks = []
    for group_key, group in df.groupby(quasi):
        conditions = group[SENSITIVE].unique()
        if len(group) >= 2 and len(conditions) == 1 and conditions[0] != "none":
            leaks.append({"group": group_key, "size": len(group),
                          "leaked_condition": conditions[0]})
    return pd.DataFrame(leaks)
