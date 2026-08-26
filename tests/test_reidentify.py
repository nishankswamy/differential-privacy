"""The re-identification attack and k-anonymity's failure modes."""

import pytest

from privacy.anonymize import (
    enforce_k_anonymity,
    generalise,
    homogeneity_leak,
    k_anonymity_level,
)
from privacy.generate import generate_population, public_voter_roll
from privacy.reidentify import linkage_attack, reidentification_rate


@pytest.fixture(scope="module")
def pop():
    return generate_population()


def test_quasi_identifiers_are_a_fingerprint(pop):
    """Sweeney's result: the (zip, birthdate, sex) triple uniquely identifies
    nearly everyone."""
    combo = pop.groupby(["zip", "birthdate", "sex"]).size()
    unique_fraction = (combo == 1).sum() / len(pop)
    assert unique_fraction > 0.9


def test_linkage_attack_reidentifies(pop):
    """Dropping the name is not anonymisation: a public voter roll re-attaches
    identities."""
    voters = public_voter_roll(pop)
    naive = pop.drop(columns=["name"])
    result = reidentification_rate(naive, voters, ["zip", "birthdate", "sex"], "condition")
    assert result["rate"] > 0.4     # a large fraction re-identified


def test_k_anonymity_stops_linkage(pop):
    """Properly enforced, k-anonymity does defeat the linkage attack — its real
    strength, before its weakness."""
    kanon = enforce_k_anonymity(pop, k=5)
    assert k_anonymity_level(kanon) >= 5

    voters = public_voter_roll(pop).assign(condition="none")
    voters_gen = generalise(voters, 4, 5)[["name", "zip", "birth_year", "sex"]]
    result = reidentification_rate(
        kanon[["zip", "birth_year", "sex", "condition"]],
        voters_gen, ["zip", "birth_year", "sex"], "condition")
    assert result["reidentified"] == 0


def test_homogeneity_attack_leaks_despite_k_anonymity(pop):
    """The weakness: a k-anonymous group where everyone shares the sensitive
    value leaks it without identifying anyone. k-anonymity protects identity,
    not the secret."""
    kanon = enforce_k_anonymity(pop, k=5)
    leaks = homogeneity_leak(kanon)
    assert len(leaks) >= 1
    assert (leaks["leaked_condition"] == "cancer").any()  # the injected cohort


def test_generalisation_reduces_uniqueness(pop):
    fine = generalise(pop, 5, 1)
    coarse = generalise(pop, 3, 10)
    assert k_anonymity_level(coarse) >= k_anonymity_level(fine)
