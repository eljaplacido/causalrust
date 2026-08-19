# Open correctness findings — `cynepic-causal`

> **Status: `cynepic-causal` produces provisional numbers.** OLS is validated
> and sound. IPW, identification, refutation and the DAG's own invariants are
> not. Do not use this crate to make a decision you cannot afford to get wrong
> until Tier 1 closes.

This is the single source of truth for the C-series findings. Everything else
that mentions them — the `allow` entries in `[workspace.lints]`, the `#[ignore]`
reasons in `crates/cynepic-causal/tests/findings.rs`, the status table in
`CLAUDE.md` — points here and must not restate the detail.

## How a finding moves

```
confirmed  →  failing spec in tests/findings.rs, #[ignore]d  →  fix + delete
              the #[ignore] in the same PR  →  ratchet count drops
```

A fix that arrives without deleting its `#[ignore]` has not been demonstrated.
`scripts/findings-ratchet.sh` enforces the count in CI: it may only ever go
down.

## Ledger

| ID | Component | Defect | Severity | Specs | Tier |
|----|-----------|--------|----------|-------|------|
| [C1](#c1) | `estimate::linear` | Singular design reported with zero uncertainty | **Critical** | 1 | 1 |
| [C2](#c2) | `identify` | Adjustment set may name an unobservable variable; identification cannot fail | **Critical** | 2 | 1 |
| [C5](#c5) | `estimate::propensity` | Variance formula does not describe the point estimate | High | 1 | 1 |
| [C7](#c7) | `refute` | Verdicts are magic constants; placebo ignores the estimator given | High | 2 | 1 |
| [C8](#c8) | `refute` | Hand-rolled LCG as the source of randomness | Medium | — | 1 |
| [C10](#c10) | `estimate` | Panics or invents numbers on degenerate input | High | 3 | 1 |
| [C11](#c11) | `dag` | `CausalDag` does not enforce acyclicity | **Critical** | 2 | 1 |
| [C12](#c12) | `dsep` | Unknown variables reported as d-separated | **Critical** | 1 | 1 |
| [C13](#c13) | `estimate::propensity` | Propensity model never converges | **Critical** | 1 | 1 |

Nine findings, 13 open specs. `estimate::linear`'s OLS path is **not** on this
list: it is measured sound (see [Validated](#validated)).

Severity is about silence, not size. A defect that makes the crate panic is
High. A defect that makes it return a confident, plausible, wrong number is
Critical, because nothing downstream can detect it.

---

## <a id="c13"></a>C13 — the propensity model never converges

**The largest defect in the crate, and it was invisible until coverage was
measured.**

`PropensityScoreEstimator::ipw` fits its logistic regression with a fixed 100
iterations of gradient descent at `lr = 0.1`, starting from zero, with no
convergence check and no way for a caller to find out:

```rust
let learning_rate = 0.1;
let n_iterations = 100;
for _ in 0..n_iterations { /* ... no stopping rule ... */ }
```

On the benign DGP (n=2000, p=3, confounding=1.0, overlap=0.35) it stops at
roughly 55% of the true coefficients. Propensity scores compress toward 0.5,
the weights under-correct, and most of the confounding survives:

| Configuration | Bias |
|---|---|
| 100 iterations (as shipped) | **+1.80** |
| 10,000 iterations | +0.26 |
| 500,000 iterations | +0.26 |
| Oracle propensity, no fitting | +0.13 |
| Naive difference in means (no adjustment at all) | +3.40 |

```
beta after 100 iterations   [0.02, 0.90, 0.78, 0.76]
true logit coefficients     [0.00, 1.43, 1.43, 1.43]
```

10k and 500k agree to four decimals, so +0.26 is the converged answer and
non-convergence accounts for **86% of the bias**. The shipped estimator removes
about 47% of the confounding it exists to remove, and reports a 95% interval
0.4 wide while doing it.

Measured coverage across the standard grid is **0.0%** in seven of eight cells:

```
 LOW  benign                bias=+1.8123  coverage=0.0%   width=0.400
 LOW  strong-confounding    bias=+3.3053  coverage=0.0%   width=0.260
 LOW  weak-overlap          bias=+3.5190  coverage=0.0%   width=0.259
 LOW  high-dim              bias=+9.6242  coverage=0.0%   width=1.241
```

Nominal 95% intervals containing the truth 0% of the time is not a tuning
issue.

**Why the test suite missed it.** `ipw_known_effect` asserts
`|ate - 5.0| < 2.0` — a 40% tolerance — on a single binary covariate with
propensities of 0.25 and 0.75. That problem is easy enough that 100 iterations
does converge, and the tolerance is wide enough to accept a broken estimator
anyway.

**Fix.** Replace gradient descent with IRLS/Newton, which converges in
5–10 iterations for this class of problem. Return `Err(NotConverged { iters,
gradient_norm })` rather than a silent partial fit. Expose the fitted scores so
a caller can check the propensity model separately from the effect estimate —
`Dataset::propensity` carries the truth specifically to make that comparison
possible, and today there is nothing to compare against.

Spec: `c13_ipw_must_remove_most_of_the_confounding`. It asserts IPW removes
>90% of the confounding bias that naive difference-in-means leaves behind. It
currently removes 47%; a converged fit reaches ~92%. The spec avoids depending
on the fitted scores precisely because they are not observable — which is part
of the finding.

---

## <a id="c1"></a>C1 — a singular design matrix is reported as a precise result

`solve_normal_equation` returns a zero vector on a small pivot and
`invert_matrix` returns a zero matrix, both without signalling. With two exactly
collinear covariates the caller receives a **non-zero ATE with a standard error
of exactly 0.0** — infinite confidence, which passes any significance test
downstream.

Zero uncertainty is never a legitimate output of a finite sample.

**Scope, measured.** The damage is confined to the variance. Appending an
exactly collinear column does *not* move the treatment coefficient — the
aliasing collapses the redundant column's own coefficient and leaves the rest
of the solution intact — so the point estimate survives and only the standard
error is destroyed. `c1_collinear_column_must_not_change_the_estimate` was
written expecting the opposite, passed, and now runs as a regression guard
instead. That narrows C1 from "the estimate is arbitrary" to "the estimate is
fine and reported with infinite confidence", which is still Critical: an
`se` of 0.0 makes every downstream significance test succeed.

**Fix.** Rank-revealing QR (or SVD with a condition-number check). Return
`Err(RankDeficient { rank, expected, aliased })` naming the offending columns.

Spec: `c1_standard_error_is_never_exactly_zero`.

---

## <a id="c2"></a>C2 — identification can return an adjustment set you cannot measure

`BackdoorCriterion::find` returns `Option<HashSet<String>>` and, in practice,
always `Some`. Two consequences:

1. There is no concept of a latent variable, so on the crate's own front-door
   test DAG — where `U` is documented as unobserved — it returns `{U}` as the
   adjustment set. Adjusting for `U` is impossible by construction.
2. Identification cannot fail. A criterion that never returns "no" is not a
   criterion; it is a formatting function.

`None` currently means "no adjustment needed", so it cannot also mean "not
identifiable" — the two are opposite conclusions sharing one representation.

**Fix.** Introduce `VarKind::{Observed, Latent}`. Return
`Result<AdjustmentSet, NotIdentifiable>` with the blocking paths named.

Specs: `c2_adjustment_set_must_not_contain_a_latent_variable`,
`c2_identification_can_fail`.

---

## <a id="c5"></a>C5 — IPW's variance does not describe IPW's point estimate

The point estimate is Hájek (normalised by the sum of weights). The variance
sums Horvitz–Thompson contributions (`y_i / e_i`, unnormalised) and centres them
on the Hájek estimate. The two are on different scales, so the subtraction is
not a residual and the result is not that estimator's variance.

There are also no heteroskedasticity-robust (HC1/HC3) standard errors anywhere
in the crate.

Distinct from [C13](#c13) and **smaller**: fixing the variance alone leaves the
point estimate biased by +1.80. Both must land for IPW coverage to be nominal.
This ordering was not obvious before the coverage harness existed, and the
original finding had them the other way round.

**Why the metamorphic relations do not catch it.** Duplicating the dataset
must leave the point estimate unchanged and shrink the standard error by √2.
IPW satisfies both halves, because duplication probes how a variance scales
with `n`, and this variance gets the `n` dependence right while getting the
*scale* wrong. A formula uniformly wrong by a constant factor still shrinks by
√2. The spec file previously claimed duplication was "unusually good at
exposing" C5; it is not, and that claim is now corrected in place.

Coverage catches it, because coverage compares an interval against a known
truth rather than against another interval. This is the clearest argument in
the repository for why the harness had to include ground-truth simulation and
not just metamorphic relations.

**Fix.** Hájek-consistent influence-function variance; HC1/HC3 for OLS.

Spec: `c5_ipw_intervals_must_achieve_nominal_coverage`.

---

## <a id="c7"></a>C7 — refutation verdicts are magic constants

```rust
let passed = relative_change < 0.15; // Less than 15% change
```

The verdict never consults the estimate's own uncertainty. A 14% shift on a
tightly-estimated effect passes; a 16% shift on one with an interval spanning
zero fails. Both verdicts are noise.

`placebo_treatment` additionally hardcodes `difference_in_means` and drops the
covariates, so it refutes an estimator the caller did not run. A placebo test
that passes for a different estimator is not evidence about yours.

Being a *relative* threshold, it is also not scale-invariant in the way the
metamorphic scaling relation requires.

**Fix.** Express verdicts in standard errors of the original estimate. Take the
estimator as a parameter instead of assuming one.

Specs: `c7_refutation_verdict_must_depend_on_uncertainty`,
`c7_placebo_must_distinguish_adjusted_from_confounded`. The latter replaced a
spec that asserted `result.passed` on a single estimate — satisfied by a
refuter that always returns `passed`, and therefore evidence of nothing.

---

## <a id="c8"></a>C8 — hand-rolled LCG

```rust
self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
```

A bare linear congruential generator drives the random-common-cause and
subset refuters. LCG low bits are notoriously non-random and the generator is
unseedable from outside, so a refutation result cannot be reproduced.

**Fix.** `rand_chacha::ChaCha8Rng` with an explicit seed on the public API. The
dependency is already in the workspace for `cynepic-testkit`.

No spec — this one is verified by the C7 rewrite plus the existing
determinism tests.

---

## <a id="c10"></a>C10 — degenerate input panics or invents numbers

Three separate paths:

- **Empty arm.** `difference_in_means` substitutes `0.0` for a missing arm's
  mean. A dataset with no control units returns a confident effect equal to the
  treated mean.
- **Mismatched lengths.** `assert_eq!` on caller-supplied input. A service
  embedding the crate dies on malformed data. CLAUDE.md states library code
  returns `Result`; this is the counterexample.
- **`n = 0`.** No guard.

**Fix.** `Result`-returning API; `EstimationError::{EmptyArm, LengthMismatch,
NoObservations}`. This is also what clears `clippy::indexing_slicing` (104
violations) and `unwrap_used` (7).

Specs: `c10_single_arm_data_must_not_produce_an_effect`,
`c10_mismatched_lengths_must_not_panic`,
`c10_empty_dataset_must_not_produce_an_effect`.

---

## <a id="c11"></a>C11 — `CausalDag` does not enforce that it is a DAG

`add_edge` returns `()` and accepts both cycles and self-loops. A type named
`CausalDag` that holds a cyclic graph invalidates every algorithm built on it:
d-separation, backdoor search and the counterfactual engine all assume
acyclicity and will silently produce nonsense or fail to terminate.

The acyclicity check already exists in `cynepic-graph`. It was never applied
here.

**Fix.** `add_edge` returns `Result<(), DagError::WouldCreateCycle { path }>`.

Specs: `c11_cycles_must_be_rejected_at_construction`,
`c11_self_loops_must_be_rejected`.

---

## <a id="c12"></a>C12 — unknown variables are reported as d-separated

`d_separated` returns `true` for variable names the graph has never heard of.
`true` means "conditionally independent", so a typo in a variable name reads as
a *positive* finding of independence — the answer most likely to be acted on.

This is the cheapest finding to fix and among the most dangerous to leave.

**Fix.** `Result<bool, DsepError::UnknownVariable { name }>`.

Spec: `c12_unknown_variable_must_not_be_reported_as_independent`.

---

## <a id="validated"></a>Validated

Measured, not asserted. From `cargo run -p cynepic-causal --example
coverage_report --release`, 300 replications per cell, nominal 95%:

| Cell | Bias | Coverage |
|---|---|---|
| benign | +0.0060 | 97.3% |
| strong-confounding | +0.0057 | 94.7% |
| weak-overlap | +0.0048 | 95.3% |
| nonlinear | −0.0012 | 97.0% |
| heteroskedastic | +0.0086 | 97.0% |
| heterogeneous-effects | +0.0060 | 96.7% |
| small-n | +0.0167 | 95.3% |
| high-dim | +0.0090 | 95.3% |

`LinearATEEstimator::ols_adjusted` achieves nominal coverage on every cell in
the standard grid, with bias below 0.02 throughout — including under
heteroskedasticity and outcome nonlinearity, where classical standard errors
were expected to degrade. C5's HC1/HC3 work remains worth doing for
adversarial DGPs, but it is not currently a source of wrong answers.

Slight over-coverage (97.3% on benign) means intervals are a little wider than
they need to be. That is the safe direction and is not a defect.

Seven specs run in the default suite as regression guards rather than
accusations:

| Guard | What it locks in |
|---|---|
| `c5_ols_intervals_must_achieve_nominal_coverage` | OLS coverage on a benign DGP |
| `c5_coverage_survives_heteroskedasticity` | 97.0% coverage under heteroskedasticity |
| `metamorphic_duplication_shrinks_se_by_sqrt_two` | diff-in-means variance scales correctly in `n` |
| `c5_ipw_duplication_shrinks_se_by_sqrt_two` | IPW variance scales correctly in `n` |
| `metamorphic_scaling_outcome_scales_effect` | estimates are scale-equivariant |
| `metamorphic_irrelevant_covariate_does_not_move_estimate` | the solver does not fit noise |
| `c1_collinear_column_must_not_change_the_estimate` | rank deficiency does not move the point estimate |

Each of these began as an `#[ignore]`d accusation and was moved here after
measurement showed the behaviour was already correct. The ratchet enforces the
distinction in both directions: an open spec that starts passing fails the
build, so nothing sits behind an `#[ignore]` pretending to be outstanding
work.

**These rows are the reason the rest of this document is credible.** A report
that found nothing wrong would not be worth reading, and a harness that could
only confirm success would not be a harness.

## Reproducing

```bash
cargo test -p cynepic-causal --test findings -- --ignored     # watch them fail
cargo run  -p cynepic-causal --example coverage_report --release
./scripts/findings-ratchet.sh                                  # the CI gate
```

Every DGP is seeded with ChaCha8, so any failure is reproducible from the seed
printed in its message, on any platform.
