# Research → implementation map

Source: `md(1).md` — competitive landscape, adoption barriers by domain, and ten
recommendations. This file scores each recommendation against what is now
**measured** in the repository, names the module that owns it, and turns the
three target verticals into concrete tests and benchmarks.

Written 2026-08-19, against branch `feat/validation-harness`.

## Scorecard

| # | Recommendation | Owner | Status |
|---|---|---|---|
| 1 | Close the validation gap programmatically | `cynepic-testkit` | **Partial** — measurement built, enforcement not |
| 2 | Publish Rust-specific benchmarks | `benches/` | **Partial** — harness exists, no comparison baseline |
| 3 | Formalize assumption elicitation | `cynepic-causal::dag` | **Not started** — and C11 blocks it |
| 4 | Harden each crate to standalone quality | all | **In progress** — lint policy landed, C-series open |
| 5 | Explicit latency-vs-rigor tiers | `cynepic-router` | **Not started** |
| 6 | Permissive licensing | workspace | **Done** — Apache-2.0, commit `fe43d8a` |
| 7 | Neutral governance and docs | `docs/` | **Partial** — FINDINGS.md is the first ADR-grade doc |
| 8 | Python and WASM interop | `bindings/` | **Not started** — WASM partial |
| 9 | Target high-confidence verticals | `cynepic-testkit::dgp` | **Not started** — specced below |
| 10 | Composable guardrail templates | `cynepic-guardian` | **Not started** |

Two are done or nearly so. Seven are untouched. That ratio is the honest
summary.

---

## The finding that changes the strategy

The research opens by citing the parent Python project's evidence: 43/43
falsifiable hypotheses at Grade A+, 1,138× more accurate causal effect
estimation than a raw LLM baseline, 100% Guardian policy-violation detection.
Those results belong to `projectcarfcynepic`. They are not transitive to this
workspace, and the coverage harness has now demonstrated why.

`cynepic-causal`'s IPW estimator has **0.0% confidence-interval coverage** on
seven of eight DGP cells, with point-estimate bias from +1.8 to +9.6. It is not
a slower version of a validated method; it is a different, broken one that
happens to share a name. See `docs/FINDINGS.md`.

**Therefore: no benchmark, README, or pitch in this repository may cite the
Python project's accuracy numbers as evidence about these crates.** Doing so
would be the precise failure the research warns about in recommendation 2 —
substantiating Rust claims with Python evidence — except worse, because the
Rust implementation is now measured and does not support them.

This is not a reason to be quiet about the project. It is the reason the
coverage table is worth publishing: it is the only artifact here that could
have come back bad, and did, which is exactly what makes the OLS rows credible.

---

## Where each recommendation lands

### 1. Close the validation gap — the differentiator, half-built

The research names this the single biggest barrier to adoption: causal estimates
have no held-out set, so there is no global validator. `cynepic-testkit` answers
the version of that objection that is actually answerable — an *estimator* can
be validated even when an *estimate* cannot — by simulating from a known truth
and measuring interval coverage.

That half is built and it works: it found C13, which no unit test in the
repository had caught.

The half that is missing is the one the research actually specified: *"enforced
via Rust's type system so a causal estimate literally cannot be exported without
an attached robustness report."* Today `ATEResult` is a bare struct of two
floats. Nothing stops it being serialized, logged, or returned to a caller with
no provenance at all.

**Next**: make `ATEResult` unconstructible outside the crate and carry a
`Robustness` field — refutation outcomes, coverage class of the estimator that
produced it, and the identification assumption it rests on. `cynepic-core`
already has `EpistemicState` and `ConfidenceLevel` for exactly this shape of
provenance; the causal crate does not use them. That is the integration gap
worth closing first, because it is the one no competitor has closed.

### 2. Rust-specific benchmarks — started, and deliberately incomplete

`benches/` now exists with a rule: correctness gates the claim. The IPW group is
named `ipw_BROKEN_C5_C13` so the label survives copy-paste.

The research asks for comparison against DoWhy, EconML and DeepCausality. That
is not yet possible and should not be faked. A cross-language comparison needs
the Python bindings from recommendation 8 to run both sides on identical data,
and a comparison against DeepCausality needs a shared task both libraries
actually implement. Until then the benchmarks measure this crate against itself
over time, which is the honest scope.

The MCMC benchmarks already establish a methodological position worth keeping:
they report **time per effective sample**, not samples per second. `AdaptiveMH`
is 15% slower per run and 1.7× cheaper per independent draw, so throughput
alone reports an improvement as a regression.

### 3. Assumption elicitation — blocked by C11

The research is emphatic that structure elicitation is the hardest step and
wants per-edge confidence and provenance metadata, plus guardrails against
silent execution on an underspecified graph.

`CausalDag::add_edge` currently returns `()` and accepts cycles and self-loops
(C11). There is no point designing provenance metadata onto a type that does not
yet enforce it is a DAG. **Fix C11 and C2 first** — C2 introduces
`VarKind::{Observed, Latent}`, which is the first piece of per-node metadata and
the natural place to hang confidence and provenance next to.

The "guardrail preventing silent execution when the graph is underspecified" is
C2's `Result<AdjustmentSet, NotIdentifiable>` almost exactly. The research and
the findings ledger converged on the same fix from opposite directions, which is
a good sign about both.

### 4. Standalone crate quality — the lint policy is the mechanism

`[workspace.lints]` now enforces the conventions and records the rest as debt
with counts. `cynepic-testkit` is the quality bar made executable for the causal
crate; the other five have no equivalent.

The research's specific advice — interoperate with `bayes-rs`/`bayes_estimate`
rather than reimplementing MCMC — is worth taking seriously in light of the ESS
benchmark. A 0.7% mixing efficiency on a bimodal target is not competitive, and
the fix is HMC/NUTS, which is a large piece of work to own. Adopting a mature
sampler crate and keeping `cynepic-bayes` as the belief-tracking and conjugate-
prior layer is likely the better trade.

### 5. Latency-vs-rigor tiers — the natural home of the Cynefin routing

Fidelity tiers per Cynefin domain is the most architecturally distinctive
suggestion in the research, and `cynepic-router` already has the classifier and
`CostTier` machinery to express it. Clear → cached/deterministic; Complicated →
OLS with refutation; Complex → full counterfactual with sensitivity analysis;
Chaotic → escalate.

Prerequisite: the tiers must be labelled with measured cost and measured
coverage, or they are guesses. Both harnesses now exist. This is the highest-
leverage item that is *not* blocked by a correctness finding.

### 6. Licensing — done

Apache-2.0 as of commit `fe43d8a`, with per-crate `LICENSE`/`NOTICE` and a CI
job asserting every published tarball ships both. The research identified BSL as
"the single largest structural adoption barrier". Removed.

### 7. Governance and docs — one real artifact so far

`docs/FINDINGS.md` is the first document here at the standard the research asks
for: it states what is broken, with measurements, before stating what works.
That ordering is the credibility mechanism.

The research points at DeepCausality's LF AI & Data Foundation status and
runnable example binaries. Foundation affiliation is premature while thirteen
correctness specs are open. Runnable examples are not — `examples/coverage_report`
is the first, and the vertical scenarios below would be the next three.

### 8. Python and WASM interop — unchanged, and now more clearly prerequisite

Still not started. Worth noting it is a dependency of recommendation 2's
cross-library comparison, so it is further up the critical path than its
"Phase 2" label suggests.

### 9. Verticals — specced below

### 10. Guardrail templates — after C2

Pre-built confounder templates per vertical need a graph type that can express
"this variable is assumed unobserved" and refuse to proceed when it is not. That
is C2. The templates themselves are cheap once the type exists.

---

## Verticals as tests and benchmarks

Recommendation 9 names three verticals where the parent project's evidence is
strongest: supply-chain disruption, healthcare treatment effects, and financial
risk. Each becomes a named DGP cell, a test, and a benchmark. The point is not
to reproduce the Python results — see above — but to make each vertical's
*failure mode* something this workspace measures.

Each is specified as: the structure that makes it hard, the DGP knobs that
encode it, and what the test asserts.

### Supply chain — `Dgp::supply_chain()`

**Hard because**: disruptions are rare, effects are lagged, and treatment
(re-routing, buffering) is assigned in response to a forecast — so the
confounder is itself a prediction of the outcome.

| Knob | Value | Encodes |
|---|---|---|
| `confounding` | 2.5 | Response-to-forecast assignment |
| `overlap` | 0.15 | Intervention is rare; few treated units resemble controls |
| `heterogeneity` | 2.0 | Effect depends on route criticality |

**Test**: coverage under weak overlap. This is where IPW's extreme weights bite
hardest and where the honest answer may be that the ATE is not estimable — an
overlap diagnostic that *refuses* is the deliverable, not a number.
**Benchmark**: `backdoor_find` on a 200-node supply graph; this is the
interactive path, so latency is user-visible.

### Healthcare — `Dgp::clinical()`

**Hard because**: the estimand distinction is load-bearing. ATE, ATT and LATE
diverge under effect heterogeneity, and reporting one while labelling it another
is a clinical error, not a rounding error. The research's "98% match vs RCT
ground truth" is an ATE claim; a per-protocol analysis estimates something else.

| Knob | Value | Encodes |
|---|---|---|
| `heterogeneity` | 2.5 | Effect varies by patient severity |
| `instrument_strength` | 0.6 | Randomized encouragement, imperfect compliance |
| `nonlinearity` | 1.5 | Dose-response is not linear |

**Test**: assert 2SLS recovers **LATE** and not ATE, and that the two differ by
more than their standard errors on this DGP — so a mislabelled estimand fails
loudly. `GroundTruth` already carries `ate`, `att`, `atc` and `late` separately
to make this checkable.
**Benchmark**: 2SLS across `n`, since trial-scale data is small and the
interesting cost is per-analysis, not per-row.

### Financial risk — `Dgp::market()`

**Hard because**: heteroskedasticity is the regime, not an edge case. Volatility
clusters, so classical standard errors understate uncertainty exactly when
uncertainty matters most. A Kupiec-style backtest is a coverage test in
different clothing — which makes this vertical the most direct fit for the
harness already built.

| Knob | Value | Encodes |
|---|---|---|
| `heteroskedastic` | true | Volatility clustering |
| `confounding` | 1.5 | Exposure responds to conditions that drive returns |
| `noise` | 2.0 | Low signal-to-noise |

**Test**: interval coverage as the backtest. Nominal 95% intervals must contain
the truth ~95% of the time; under-coverage is a capital-adequacy failure, and
this is the one vertical where the regulator's test and the statistician's test
are literally the same test. This is also the cell where HC1/HC3 robust standard
errors (C5) must prove themselves.
**Benchmark**: `ols_adjusted` at n=50,000, the only vertical whose data volume
justifies a Rust implementation on throughput grounds alone.

### Sequencing

None of the three can produce a trustworthy number before Tier 1. Build the DGP
cells now — they cost little and sharpen the fix targets — and add the coverage
assertions as the findings close. The supply-chain overlap diagnostic and the
healthcare estimand test are both *new capability*, not just validation, and are
the two most defensible things to build next.

---

## What to do next, in order

1. **C11 + C12** — cheapest of the findings, and C11 unblocks recommendation 3.
2. **C13 + C5** — IPW is the only measured-wrong number the crate ships.
3. **C2** — unblocks recommendations 3 and 10, and introduces the first node
   metadata.
4. **Robustness-carrying `ATEResult`** — recommendation 1's missing half, and
   the wiring of `EpistemicState` into the causal crate.
5. **Vertical DGP cells** — cheap, and they aim everything above.
6. **Fidelity tiers in the router** — the distinctive architecture claim, now
   backed by two measurement harnesses.

Python bindings, foundation affiliation, and cross-library benchmarks stay
parked until the crate's own numbers are ones worth comparing.
