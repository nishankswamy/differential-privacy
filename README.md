# Privacy-Preserving Analytics

Show that anonymisation fails by re-identifying a synthetic "de-identified"
dataset, then implement differential privacy properly — mechanisms, sensitivity,
a privacy budget, local DP — and measure what real privacy costs.

Days 20–22 of a 30-day challenge, all three done. Every attack is on data
generated here; nobody real is in it.

```bash
pip install -r requirements.txt
python demo_reidentify.py   # anonymisation fails
python evaluate_main.py     # the utility/privacy tradeoff, measured
pytest                      # 26 tests
```

## The finding

**Anonymisation as usually practised doesn't work, and the fix isn't a better
anonymiser — it's a different definition of privacy.**

Three measured results tell the story:

1. **Dropping names re-identifies 60% of people.** A synthetic "de-identified"
   hospital release, joined to a public voter roll on `(zip, birthdate, sex)`,
   re-attaches names to medical conditions for 1,195 of 2,000 people. This is
   Latanya Sweeney's attack, reproduced — the quasi-identifier triple is a
   fingerprint (>90% of people uniquely identified by it).

2. **k-anonymity stops the linkage attack but leaks the secret anyway.** Enforced
   properly (k=5), the linkage attack drops to 0 re-identifications. But an
   oncology cohort — 18 people sharing a ZIP and age band, all with cancer —
   forms a k-anonymous group that is *homogeneous* on the diagnosis. Anyone who
   knows a target is in that group learns their cancer status without ever
   identifying which record is theirs. **k-anonymity protects identity, not the
   secret** — which is usually what you actually wanted.

3. **Differential privacy makes a guarantee that survives both.** DP promises the
   output is almost unchanged whether or not any individual is in the data, so no
   attacker — with any side information — learns much about a specific person. The
   utility cost is real and measured below.

## Anonymisation fails (Day 20)

```
A 'de-identified' release: names dropped, everything else kept.
  Attacker joins it to a voter roll on (zip, birthdate, sex):
  -> 1,195 of 2,000 people re-identified (60%)

    person_3  (zip 02109, born 19570401, M)  ->  hypertension
    person_6  (zip 02104, born 19730921, M)  ->  diabetes

Fix attempt: k-anonymity (k=5)
  Linkage attack now: 0 re-identified. k-anonymity stopped it.
  BUT — 1 k-anonymous group leaks the condition anyway:
    group (0219*, 1950, F): all 18 members have 'cancer'
```

The homogeneity attack is why l-diversity, then differential privacy, had to be
invented. Each is a response to a hole in the last.

## Differential privacy, done correctly (Day 21)

The mechanism: add calibrated noise to the true answer. Two numbers set the
noise, and getting either wrong silently voids the guarantee:

**Sensitivity (Δf)** — the most one person can change the answer. A count: 1. A
sum of values in [0, C]: C. **This is the most common way real DP is broken** — a
sum with no clamp has unbounded sensitivity, so one person with a huge value means
no finite noise is enough. `private_sum` clamps first, and `test_private_sum_clamps`
proves one 10⁹ outlier can't dominate the result.

**Epsilon (ε)** — the privacy budget. Smaller ε = more noise = stronger privacy.
It's not a probability; it bounds a *ratio of likelihoods*, which is what makes it
composable.

```
Laplace mechanism:   answer + Laplace(Δf / ε)              →  ε-DP
Gaussian mechanism:  answer + Normal(σ from Δf, ε, δ)      →  (ε,δ)-DP
```

The noise is unbiased, which is the subtle danger: average enough DP answers and
the noise cancels, recovering the truth. `test_laplace_is_unbiased` confirms it
over 20,000 runs — and that's exactly why a **budget** is non-negotiable.

### The budget (why DP without one is theatre)

Privacy loss **composes**: ask an ε-query twice, you've spent 2ε, because the two
noisy answers average to half the noise. The `PrivacyBudget` refuses any query
that would overspend — a rejected query costs nothing, an exhausted budget stops
answering. Two composition rules:

| queries at ε=0.1 | basic (sum) | advanced (√k) |
|---|---|---|
| 10 | 1.0 | 1.8 (basic wins) |
| 100 | 10.0 | 6.3 |
| 1,000 | 100.0 | **27.1** |

Advanced composition grows like √k instead of k, permitting far more queries per
budget — but it has overhead and *loses* to basic at small k, which the code and
`test_advanced_composition_overhead_at_small_k` both state honestly.

## What real privacy costs (Day 22)

Relative error by query type across epsilon:

| ε | count err | mean err | sum err |
|---|---|---|---|
| 0.01 | 63.9% | 6.9% | 7.8% |
| 0.1 | 6.5% | 0.7% | 0.7% |
| 1.0 | 0.7% | 0.07% | 0.07% |
| 5.0 | 0.1% | 0.01% | 0.01% |

Same noise, wildly different relative impact: a count of 159 is small, so Laplace
noise is a big fraction of it; a sum over 2,000 ages is large, so the same noise
is a rounding error. **Sensitivity and the value's scale together decide utility,
not epsilon alone.**

### Local vs central DP

Central DP trusts a curator to hold raw data and noise the answers. **Local DP**
(randomised response) trusts no one — each person perturbs their own answer before
it leaves their device (what Apple/Google run on phones). The measured cost:

| ε | local error | central error | local penalty |
|---|---|---|---|
| 0.5 | 0.011 | 0.001 | ~11x |
| 1.0 | 0.009 | 0.001 | ~15x |
| 2.0 | 0.005 | 0.000 | ~17x |

Local DP is 10–20x noisier at the same ε, because it adds noise to *every* answer
rather than once to the aggregate. You pay that to avoid trusting a curator.

The randomised-response mechanism is Warner's (truth with probability p, else
report the *flipped* bit), calibrated `p = e^ε/(e^ε+1)` so the likelihood ratio is
exactly `e^ε`. A subtle trap the code documents: the "randomise to a uniform coin"
variant with the same p gives a *different* ratio (`2e^ε+1`) and silently weaker
privacy — an early version of this made exactly that error, caught by a test that
measures the ratio directly.

## Depth questions

**What does ε=1 actually promise a person in the dataset? Say it without the word
"privacy".**
That any conclusion an observer draws about you from the released result is at
most `e^1 ≈ 2.7×` more (or less) likely than it would be if your record were
removed entirely. So joining the dataset changes what anyone can infer about you
by at most that bounded factor — whether the query is asked once. It is a
worst-case, side-information-proof bound on how much your participation shifts any
inference.

**You answered 100 queries at ε=0.1 each. What's your real total privacy loss?**
Not 0.1 — that was per query. Under basic composition it's 100 × 0.1 = 10, which
is weak privacy. Under advanced composition it's ≈6.3 at a small added δ. Either
way the point stands: the budget is spent across the *workload*, and a system that
answers unlimited ε=0.1 queries has ε→∞ effective loss. The noise on any single
answer is irrelevant if an attacker can ask again.

**Why is local DP so much noisier than central, and when is the cost worth it?**
Central adds noise once, to the aggregate — error shrinks like 1/√n as the crowd
grows. Local adds noise to each of n answers, so the errors accumulate and only
partially cancel; you need far more people for the same accuracy (measured: 10–20x
worse at equal ε). It's worth it precisely when you cannot trust the collector
with raw data — a phone vendor gathering typing statistics can't promise not to
be breached or subpoenaed, so it never collects the truth in the first place.

**Your k-anonymised dataset was re-identified 0% by linkage but leaked cancer for
a whole cohort. What does that say about published "anonymised" datasets?**
That "we removed identifiers and generalised the quasi-identifiers" is not a
privacy guarantee — it's a hope that no group is homogeneous and no auxiliary
dataset exists. Both hopes fail routinely (the Netflix Prize, the Massachusetts
GIC release, the AOL search logs). A dataset is only safely publishable if the
*mechanism that produced it* has a formal guarantee like DP; otherwise its safety
depends on what the attacker happens to know, which you don't control.

## Layout

```
privacy/
  generate.py    synthetic population + public voter roll + oncology cohort
  anonymize.py   k-anonymity, generalisation, the homogeneity leak
  reidentify.py  the linkage attack
  dp.py          Laplace/Gaussian mechanisms, sensitivity helpers
  budget.py      privacy budget + basic/advanced composition
  local_dp.py    randomised response (Warner), local-vs-central error
demo_reidentify.py   Day 20: anonymisation fails
evaluate_main.py     Day 22: the utility/privacy tradeoff
tests/               26 tests
```

## What I'd do differently

<!-- Fill this in. -->

## Known gaps

- The DP here is per-query with basic/advanced composition. Production systems use
  a privacy accountant (Rényi DP / zCDP) for much tighter composition over large
  workloads — the right next step.
- No DP for histograms or top-k, where per-bin sensitivity and thresholding get
  subtle. Counts, sums, means only.
- k-anonymity enforcement is a naive generalisation ladder, not an optimal
  algorithm (Mondrian, Incognito). It demonstrates the failure modes, not
  state-of-the-art anonymisation.
- Sensitivity is computed analytically per query. Real systems need it for
  arbitrary user-defined queries, which requires either static analysis or a
  restricted query language.
