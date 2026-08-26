"""The linkage attack: re-attach identities to an 'anonymised' release.

Given an anonymised dataset (quasi-identifiers + sensitive value, no names) and a
public dataset the attacker holds (names + the same quasi-identifiers), join them
on the quasi-identifiers. Any anonymised record whose quasi-identifiers are unique
gets exactly one name back — and with it, the sensitive value.

This is Sweeney's attack, and the measured re-identification rate is the finding:
how many people in a 'de-identified' medical release can be named by anyone with a
voter roll.
"""

from __future__ import annotations

import pandas as pd


def linkage_attack(anonymised: pd.DataFrame, public: pd.DataFrame,
                   join_keys: list[str], sensitive: str) -> pd.DataFrame:
    """Return the records the attacker can uniquely re-identify.

    A record is re-identified when the join on quasi-identifiers yields exactly
    one candidate name — the quasi-identifiers were a fingerprint. Ambiguous
    joins (many candidates) are the protection k-anonymity is supposed to
    provide."""
    merged = anonymised.merge(public, on=join_keys, how="inner", suffixes=("", "_public"))

    # Count candidate identities per anonymised record.
    candidates = merged.groupby(join_keys)["name"].nunique()
    unique_keys = candidates[candidates == 1].index

    reidentified = merged.set_index(join_keys).loc[
        merged.set_index(join_keys).index.isin(unique_keys)
    ].reset_index()

    return reidentified[join_keys + ["name", sensitive]].drop_duplicates()


def reidentification_rate(anonymised: pd.DataFrame, public: pd.DataFrame,
                          join_keys: list[str], sensitive: str) -> dict:
    reidentified = linkage_attack(anonymised, public, join_keys, sensitive)
    return {
        "reidentified": len(reidentified),
        "released": len(anonymised),
        "rate": len(reidentified) / len(anonymised) if len(anonymised) else 0.0,
    }
