"""Day 20: anonymisation fails. Re-identify a synthetic 'de-identified' release.

    python demo_reidentify.py
"""

from __future__ import annotations

from privacy.anonymize import enforce_k_anonymity, homogeneity_leak, k_anonymity_level
from privacy.generate import generate_population, public_voter_roll
from privacy.reidentify import linkage_attack, reidentification_rate


def main() -> None:
    pop = generate_population()
    voters = public_voter_roll(pop)

    print("A 'de-identified' hospital release: names dropped, everything else kept.\n")

    naive = pop.drop(columns=["name"])
    r = reidentification_rate(naive, voters, ["zip", "birthdate", "sex"], "condition")
    print(f"  Attacker joins it to a public voter roll on (zip, birthdate, sex):")
    print(f"  -> {r['reidentified']:,} of {r['released']:,} people re-identified "
          f"({r['rate']:.0%}) — names re-attached to medical conditions.\n")

    examples = linkage_attack(naive, voters, ["zip", "birthdate", "sex"], "condition").head(3)
    for _, row in examples.iterrows():
        print(f"    {row['name']}  (zip {row['zip']}, born {row['birthdate']}, "
              f"{row['sex']})  ->  {row['condition']}")

    print("\n" + "=" * 66)
    print("Fix attempt: k-anonymity (k=5) — generalise until groups have >= 5")
    print("=" * 66 + "\n")

    kanon = enforce_k_anonymity(pop, k=5)
    print(f"  Achieved k={k_anonymity_level(kanon)}, kept {len(kanon):,} of "
          f"{len(pop):,} rows (some suppressed, ZIP and age generalised).")

    voters_gen = voters.assign(condition="none")
    from privacy.anonymize import generalise
    voters_gen = generalise(voters_gen, 4, 5)[["name", "zip", "birth_year", "sex"]]
    r2 = reidentification_rate(
        kanon[["zip", "birth_year", "sex", "condition"]],
        voters_gen, ["zip", "birth_year", "sex"], "condition")
    print(f"  Linkage attack now: {r2['reidentified']} re-identified "
          f"({r2['rate']:.0%}). k-anonymity stopped the linkage.\n")

    leaks = homogeneity_leak(kanon)
    print(f"  BUT — {len(leaks)} k-anonymous group(s) leak the condition anyway:")
    for _, leak in leaks.iterrows():
        print(f"    group {leak['group']}: all {leak['size']} members have "
              f"'{leak['leaked_condition']}'")
    print(f"""
  Anyone who knows a target is in that group learns their diagnosis without
  ever identifying *which* record is theirs. k-anonymity protected identity;
  it did not protect the secret — which is what you actually wanted. This is
  the homogeneity attack, and it's why l-diversity, then differential privacy,
  had to be invented.

  Day 21 builds differential privacy, which makes a guarantee about *every*
  individual regardless of what the attacker already knows.
""")


if __name__ == "__main__":
    main()
