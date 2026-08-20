# Correctness findings — cynepic-rs

> **Status: twelve findings closed, two open, across three measured crates.**
>
> - `cynepic-causal` — `ols_adjusted` is nominal on all nine grid cells; IPW
>   under-covers by ~3 points where the weights are heavy ([C14](#c14), open).
>   The gap narrowed 87.3% → 90.3% → 91.8% across two rounds, and the third
>   round found why no interval can close it: at strong confounding **100% of
>   the intervals that miss are the ones reporting a below-median standard
>   error**. Under heavy weights a narrow interval is the case to distrust.
> - `cynepic-bayes` — every conjugate interval is calibrated and both samplers
>   pass simulation-based calibration ([B1](#b1), closed).
> - `cynepic-router` — the keyword classifier scores **macro F1 0.290** with
>   **0.000 recall on Chaotic** ([R1](#r1), open). A new `LexicalClassifier`
>   takes those to **0.608** and **0.500** under cross-validation, which is most
>   of the distance to the bar but not all of it, so the finding stays open.
>
> - `cynepic-guardian` — circuit breaker, rate limiter and loop detector are
>   property-tested over operation sequences ([G1](#g1), closed).
>
> - `cynepic-graph` — **no findings.** Fifteen execution properties hold:
>   determinism across runs and across rebuilds, `max_steps` as a hard bound
>   (exactly `n` node executions at every budget tested), checkpoint resume
>   reproducing an uninterrupted run at every cut point and across a JSON round
>   trip, budget charged against the original rather than reset on resume, node
>   timeouts bounding the wait, and every started node reporting exactly one
>   terminal event.
>
> Five of six publishable crates are now measured. `cynepic-core` is types and
> traits plus `special`, which is checked against closed forms.
>
> **A crate with no findings is a result, not an absence of one** — but only
> because the properties were written to be capable of failing, and four other
> crates written the same way did fail.

Single source of truth for every finding in the workspace. Everything else that
mentions them — the `allow` entries in `[workspace.lints]`, the spec names in
the suites below, the status table in `CLAUDE.md` — points here and must not
restate the detail.

| Series | Crate | Spec suite |
|---|---|---|
| C | `cynepic-causal` | `tests/findings.rs` |
| B | `cynepic-bayes` | `tests/calibration.rs` |
| R | `cynepic-router` | `tests/routing_accuracy.rs` |
| G | `cynepic-guardian` | `tests/guardrails.rs` |
| — | `cynepic-graph` | `tests/execution_properties.rs` (no findings) |

## How a finding moves

```
confirmed  →  failing spec in the crate's suite, #[ignore]d  →  fix + delete
              the #[ignore] in the same PR  →  ratchet count drops
```

`scripts/findings-ratchet.sh` enforces three invariants in CI: the open count
may only go down, the specs still compile, and **every open spec still fails**.
That third check is what caught four mislabelled specs during the re-baseline,
and it is why a fixed finding cannot quietly stay on the books.

## Ledger

| ID | Component | Defect | Severity | Status |
|----|-----------|--------|----------|--------|
| [C1](#c1) | `estimate::linear` | Singular design reported with zero uncertainty | Critical | **Closed** |
| [C2](#c2) | `identify` | Adjustment set could name a latent variable; identification could not fail | Critical | **Closed** |
| [C5](#c5) | `estimate::propensity` | Variance formula did not describe the point estimate | High | **Closed** |
| [C7](#c7) | `refute` | Verdicts were magic constants; placebo ignored the estimator | High | **Closed** |
| [C8](#c8) | `refute` | Hand-rolled LCG as the source of randomness | Medium | **Closed** |
| [C10](#c10) | `estimate` | Panicked or invented numbers on degenerate input | High | **Closed** |
| [C11](#c11) | `dag` | `CausalDag` did not enforce acyclicity | Critical | **Closed** |
| [C12](#c12) | `dsep` | Unknown variables reported as d-separated | Critical | **Closed** |
| [C13](#c13) | `estimate::propensity` | Propensity model never converged | Critical | **Closed** |
| [C14](#c14) | `estimate::propensity` | Under heavy weights a *small* SE marks the intervals that miss | High | **OPEN** (mechanism identified) |
| [B1](#b1) | `bayes::priors` | Beta interval was a normal approximation | High | **Closed** |
| [R1](#r1) | `router::classifier` | Keyword classifier has no reach on natural phrasing | **Critical** | **OPEN** (answered, below bar) |
| [G1](#g1) | `guardian::circuit_breaker` | No half-open state; whole load restored on a timer | **Critical** | **Closed** |

```
open specs:   18   →   20   →   13   →   2   →   5
            initial  measured  re-baselined  Tier 1  + bayes & router measured
```

The count rose at the last step because two more crates were measured for the
first time. A ledger that only shrinks is a ledger that has stopped looking.

Severity is about silence, not size. A defect that makes the crate panic is
High. One that makes it return a confident, plausible, wrong number is Critical,
because nothing downstream can detect it.

---

## <a id="g1"></a>G1 — the circuit breaker had no half-open state · Closed

The breaker held a single `is_open` boolean. Once `reset_timeout` elapsed,
`allow()` returned `true` to **every** caller while `is_open()` still reported
`true`. The code's own comment said "Allow one attempt (half-open)"; nothing
counted the attempts.

Measured: **100 of 100 calls admitted** after the timeout.

The consequence is precise. A dependency goes down, the breaker trips and
shields it, the timeout elapses — and instead of one probe deciding whether the
dependency has recovered, the entire backed-up load arrives at once. That is the
thundering herd the breaker was installed to prevent, now delivered on a timer,
and it repeats every `reset_timeout` for as long as the dependency stays down.

**Fixed** by replacing the boolean with a three-valued state atomic and making
the `Open -> HalfOpen` transition a compare-and-exchange. Exactly one caller
wins the exchange and becomes the probe; every other caller observes `HalfOpen`
and is refused.

A boolean cannot express this. Two concurrent callers both read "timeout
elapsed" and, with nothing to claim, both proceed — which is why the fix is a
state machine rather than a counter. The test that pins it down races 64 tasks
at a just-expired breaker and asserts exactly one is admitted.

Also added: `record_failure` from `HalfOpen` re-opens for a *full* timeout
rather than leaving the original trip time in place, `probes_admitted()` so
"how often did we retry a dead dependency" is answerable at all, and a clamp on
a zero `failure_threshold`, which would otherwise trip before any failure and
pin the breaker open forever.

**One spec was not a witness.** `g1_failed_probe_reopens_the_breaker` passed
before the fix — `record_failure` already restarted the trip clock — so it never
demonstrated the defect. Per the ratchet's third invariant it runs as a guard
rather than an accusation.

Specs: `g1_half_open_admits_exactly_one_probe`,
`g1_failed_probe_reopens_the_breaker`, plus eleven guardrail properties covering
threshold accuracy, concurrent failure counting, burst capacity, per-key
isolation, retry hints, non-consuming peek, refill rate, overvisit detection,
alternation detection and reset.

---

## <a id="r1"></a>R1 — the keyword classifier has no reach on natural phrasing · **OPEN**

Measured against a 96-query corpus in `cynepic-testkit::corpus`, balanced 24 per
domain across four verticals, and written **without reference to the
classifier's keyword lists**. A corpus assembled by reading those lists would
have scored well by construction and measured nothing.

| domain | precision | recall | F1 |
|---|---|---|---|
| Clear | 0.833 | 0.417 | 0.556 |
| Complicated | 1.000 | 0.083 | 0.154 |
| Complex | 1.000 | 0.292 | 0.452 |
| Chaotic | **0.000** | **0.000** | **0.000** |

**accuracy 0.198, macro F1 0.290** — against a four-class random baseline of
about 0.25.

**Chaotic recall is zero.** All 24 live-incident queries are missed, because
none of them happens to contain "emergency", "crisis", "outage", "breach",
"urgent" or "critical failure". Answering "the site is down and we do not know
why" from a cached lookup is the exact failure the Cynefin split exists to
prevent, which makes this the highest-severity finding in the workspace despite
being in the least statistically sophisticated crate.

**The decomposition says what kind of problem it is:**

```
no keyword matched at all (-> Disorder):  75/96  (78%)
matched, but the wrong domain:             2/96  ( 2%)
```

Those two failures call for opposite remedies, and only the second is a tuning
problem. At 78% no-signal the approach has no *reach* on ordinary phrasing.
**Editing the keyword lists against this corpus would overfit to it** and leave
the next phrasing just as unreachable, so it has deliberately not been done —
the number stands as the measurement of what a keyword matcher is worth here.

### Answered, but not to the bar · `LexicalClassifier`

`cynepic-router::lexical` replaces list membership with tf-idf over unigrams and
bigrams, one centroid per domain, cosine similarity. Every term contributes,
weighted by how well it distinguishes the domains, instead of a handful deciding
and the rest being discarded.

Measured by **4-fold cross-validation on the corpus** — train on three quarters,
score the quarter never seen, rotate:

| | keyword | lexical | R1's bar |
|---|---|---|---|
| macro F1 | 0.290 | **0.608** | 0.70 |
| Chaotic recall | **0.000** | **0.500** | 0.80 |
| queries with no signal | 78% | **11%** | — |

**The specs still fail**, by 0.09 on F1 and 0.30 on recall, so R1 stays open and
the ratchet is unchanged. But 0.290 → 0.608 against a 0.25 random baseline, and
0.000 → 0.500 on the class that matters most, is most of the distance.

Cross-validation is the headline rather than a single train-and-score run
because the shipped exemplars were written by someone who had read this corpus.
Nothing was copied, but it cannot be un-read and "I was careful" is not a
measurement. No CV fold can be contaminated by its own training data, and it is
also the harder test — 72 training examples against a purpose-built 48.

**Why not an embedding model.** Still wanted, still on the roadmap. This
establishes what the cheap approach is worth first, so a later claim about
embeddings has a real number to beat rather than 0.290. It also costs nothing to
carry: no model file, no inference runtime, no new dependency, `wasm32-wasip1`
still builds, results are bit-identical across platforms, and
`LexicalClassifier::explain` names the terms that drove a verdict — which an
embedding model gives up.

### Two things the measurement caught

**Entropy ran backwards.** Mean entropy came out at 0.734 on ambiguous input
against 0.819 on answerable input — the wrong way round, which would have made
an escalation policy fire on exactly the queries it should have let through. A
query matching one stray term puts all of its tiny mass on a single domain and
so looks maximally *decisive*; a rich, specific query matches terms across
several domains and looks uncertain. Fixed by shrinking the score distribution
toward uniform in proportion to the evidence — the standard treatment of a weak
likelihood — rather than by rescaling until the test passed.

**Cosine cannot tell ambiguity from ignorance.** It is length-normalised, which
is what makes it robust to query length and also what makes it blind to how much
evidence there is: normalising divides the magnitude away. Measured
(p10/p50/p90):

| | cosine | evidence mass |
|---|---|---|
| ambiguous | 0.00 / 0.09 / 0.19 | 0.0 / 1.0 / 3.3 |
| answerable | 0.16 / 0.24 / 0.35 | 1.8 / 4.6 / 7.4 |

Cosine overlaps; mass barely does. Abstention is gated on mass.

### The guard that was deliberately changed

The keyword-era guard required **all ten** contentless queries to return
`Disorder`. The keyword classifier met it trivially, because it had no reach and
abstained on 78% of *answerable* queries too — abstention was a byproduct of
blindness, not a designed safety property, and a guard a blind classifier passes
for free measures nothing.

The lexical classifier answers five of the ten. The frontier is measured and
there is no free point on it:

| evidence threshold | ambiguous answered | macro F1 | Chaotic recall | corpus silent |
|---|---|---|---|---|
| 1.5 (default) | 5/10 | 0.608 | 0.500 | 11% |
| 2.5 | 3/10 | 0.541 | 0.458 | 18% |
| 4.0 | 1/10 | 0.494 | 0.375 | 36% |

Buying back the old guard costs a fifth of the macro F1 and a quarter of the
Chaotic recall — it buys silence on ten contentless queries by missing more live
incidents. So the property is restated as what matters operationally: most
contentless input still abstains, **nothing contentless is ever answered
confidently** (no ambiguous query may take more than half the posterior), and
entropy still separates the two populations. A deployment needing the stricter
behaviour has `with_min_evidence` and the table above telling it the price.

**Fix.** The embedding classifier already on the roadmap closes the remaining
0.09 of F1 and 0.30 of recall. This finding is what turns that from a
nice-to-have into a quantified requirement, and now gives it two before numbers
rather than one.

**What makes it survivable in the meantime.** The classifier abstains rather
than guessing: no signal returns `Disorder` at zero confidence, and only 2% of
queries are confidently misrouted. Mean entropy is 1.000 on ambiguous input
against 0.781 on answerable input, so an escalation policy has a usable signal
to trigger on. Those properties are asserted as regression guards and must not
be traded away for a higher accuracy score — a confident wrong route is worse
than an admitted unknown at every ratio.

Specs: `r1_chaotic_queries_are_routed_to_chaotic`, `r1_macro_f1_is_usable`,
`r1_most_queries_produce_signal`. Guards:
`abstains_on_ambiguous_input_rather_than_guessing`,
`confident_misrouting_stays_rare`, `precision_is_high_where_the_classifier_fires`,
`entropy_separates_ambiguous_from_answerable`.

### Fixed alongside it

Scoring was `matches / keywords.len()`, so a single match against the
four-keyword `Clear` list outscored a single match against the seven-keyword
`Complicated` list, 0.25 to 0.14. List length is an authoring artifact and
carries no evidence about the query. Now scored by matched query coverage, which
also makes a specific phrase outrank a generic one.

Worth recording: **fixing it did not move any headline number**, because the
binding constraint is the 78% that match nothing at all, not the tie-breaking
among the rest. A real bug that is not the cause of the symptom is still worth
fixing, and worth reporting as not having helped.

---

## <a id="b1"></a>B1 — the Beta credible interval was a normal approximation · Closed

`BetaBinomial::credible_interval_95` built `mean ± 1.96·sd` and clamped to
`[0, 1]`. A Beta density is symmetric only when `alpha == beta`, so a symmetric
interval is in the wrong place whenever the counts are unbalanced.

**The hypothesis was wrong as first stated.** Measured *marginally* — averaging
over draws from the prior — the interval is fine: 95.0% to 95.9% coverage across
`n = 2` to `n = 200` and at both boundaries. Errors at low `p` cancel errors at
high `p`.

It is wrong *conditionally*, at fixed `p`, which is the question a reliability
monitor actually asks: a tool has one real reliability, not a draw from a prior.
But departures from 95% at fixed `p` are also just binomial discreteness, which
no interval method escapes at small `n` — so the finding required a comparison
against an exact interval on identical data, not against nominal:

| p | shipped (before) | exact reference |
|---|---|---|
| 0.02 | 99.9% | 81.4% |
| 0.05 | 98.9% | 91.2% |
| 0.50 | **89.4%** | **97.8%** |
| 0.95 | 98.8% | 91.3% |
| 0.98 | 99.9% | 81.3% |

Under-covering by 8.4 points at moderate `p` — the ordinary operating range —
and over-covering at the boundaries where the symmetric interval overran
`[0, 1]` and was clamped rather than corrected. Wrong in both directions, and
the direction that mattered was the dangerous one.

**Fixed** with `cynepic-bayes::special`: ln-gamma (Lanczos), regularised
incomplete beta and gamma (Lentz continued fraction and series), quantiles by
bisection. Bisection rather than Newton because a tool with 200 successes and no
failures gives `Beta(201, 1)`, where the density occupies the last thousandth of
the range and a Newton step leaves `[0, 1]`.

`GammaPoisson` gained an interval it never had. A Gamma posterior is
right-skewed at low counts, so adding a symmetric one would have repeated the
mistake that had just been caught.

Post-fix the shipped interval matches the reference to nine decimal places at
every `p`, and the residual departures from 95% are identical in both columns —
confirming they are discreteness, not implementation.

The guard that catches a regression is not the coverage test, which must stay
loose enough to tolerate discreteness. It is exact agreement with an
independently written reference across a 6×6 grid of counts.

Specs: `b1_shipped_interval_matches_the_exact_reference`,
`b1_interval_is_in_range_without_clamping`,
`b1_interval_is_asymmetric_when_the_posterior_is`,
`b1_beta_binomial_intervals_are_calibrated_at_small_n`.

---

## <a id="c14"></a>C14 — a small standard error is a danger signal, not a precise one · **OPEN, mechanism identified**

Coverage under heavy weights: **87.3% → 90.3% → 91.8%** across two rounds of
partial fixes. Still below nominal, and the third round established *why no
interval can close it*, which is a different claim from the one this finding
started with.

### The result that reframes it

Over 800 replications, of the replications whose interval **failed to cover**:

| cell | corr(\|error\|, se) | share of misses from a **below-median** SE |
|---|---|---|
| benign | 0.067 | 40.0% |
| moderate-overlap | 0.076 | **98.5%** |
| strong-confounding | −0.036 | **100.0%** |
| high-dim | 0.509 | 82.0% |

Under independence that last column is 50%. At strong confounding it is
**100%**: every single failure to cover happened in a replication that reported
*below-median* uncertainty.

The interval is narrowest exactly when it most needs to be wide. A t-interval
assumes the error and the standard error are independent, and here they are
anti-informative. **No degrees-of-freedom rule can repair that**, because a dof
rule scales every interval by the same factor — reaching the bad replications
means grossly over-covering all the others. That is visible directly: a fixed
`dof = 4` brings the heavy cells to 96.0% and takes `benign` to 99.8%.

The mechanism is that the heavy-weight units carry the correction that removes
confounding bias. A sample that happens not to contain them produces both a
small variance estimate *and* an estimate that is systematically off. It is
confidently wrong, and it is confident *because* it is wrong.

**The operational consequence, which is the useful part:** under heavy weights,
a small reported standard error is not evidence of precision. Read
`Diagnostics::effective_n` and `Diagnostics::variance_dof` first. A dof in the
single digits on `n = 2000` means the interval rests on a handful of
observations, and a *narrow* interval in that regime is the case to distrust
most.

### Two fixes that did land, and what they were worth

Measured across the DGP grid, 500 replications, coverage at 95% nominal:

| cell | before | after |
|---|---|---|
| benign | 95.6% | 95.6% |
| moderate-overlap | 90.4% | **91.2%** |
| strong-confounding | 91.2% | **91.8%** |
| nonlinear | 96.4% | 96.4% |
| heteroskedastic | 96.2% | 96.6% |
| heterogeneous-effects | 95.6% | 95.6% |
| small-n | 95.4% | 96.2% |
| **high-dim** | 90.4% | **91.8%** |

**1. The projection was not paying for its coefficients.** `residual_psi` is the
residual from a `k`-coefficient least-squares fit, so `sum(residual^2)` is a
*residual* sum of squares and is biased low by `(n - k)/n` — the same reason an
OLS variance divides by `n - k`. The variance divided by `n^2` and made no such
correction. At `n = 2000` with a few covariates that is a quarter of a percent
and invisible, which is why it survived; at **high-dim** (`p = 25`, so `k = 26`,
on `n = 400`) it is **6.5% of the variance**. That cell covers at 89.2% with a
bias of **0.02 sd** — nothing was wrong with the point estimate, the interval
was simply too narrow. `high-dim` was not previously recorded as affected.

**2. Satterthwaite carried a Gaussian assumption into a correction that exists
because Gaussianity failed.** `nu = 2 (sum psi^2)^2 / (sum psi^4 - ...)`. The
factor of two is `Var(chi^2_nu) = 2 nu`, which holds for squares of Gaussians.
Under heavy tails `psi^2` has a coefficient of variation above the Gaussian
value, so the true dof is *below* `2 x Kish`. Dropping the factor gives Kish's
effective sample size — the same quantity already reported as `effective_n`,
applied to the influence contributions instead of the weights.

The new rule's dof now matches the dof the observed noise implies, which is the
check that it is right rather than merely helpful:

| cell | dof implied by observed cv(se) | Satterthwaite | Kish |
|---|---|---|---|
| moderate-overlap | 6.8 | 16.5 | **8.2** |
| strong-confounding | 7.8 | 14.2 | **7.1** |

And it is invisible where it should be: at `benign` it takes the dof from 803 to
334, and `t(0.975, 334)` is 1.967 against 1.963.

### Three hypotheses refuted by measurement

**"The propensity model is estimated from the same data, and that is the second
noise source."** This was the finding's own stated explanation for the residual
gap. It is wrong. Substituting the DGP's *true* propensity isolates the sources,
and the noise does not move:

| cell / propensity | cv(se) |
|---|---|
| moderate-overlap / **true** | 0.276 |
| moderate-overlap / fitted | 0.272 |
| strong-confounding / **true** | 0.255 |
| strong-confounding / fitted | 0.253 |

Propensity estimation contributes essentially nothing to the noise in the
standard error. It is heavy tails, entirely. (It contributes a great deal to the
*efficiency*: the true-propensity estimator has sd 0.237 against 0.148 for the
fitted one, which is the textbook result and a good check that the substitution
worked.)

**"The sample fourth moment is biased down, so compute it from the model."**
`S4`'s median is 1.39e-1 against a mean of 1.66e-1 at strong confounding, so the
downward bias is real and it does inflate `nu`. Replacing it with an expectation
over `T_i ~ Bernoulli(e_i)` — using every unit in both arms, so the `1/e` tail
is fully represented — made it **worse**: `nu` went from 14.5 to 30.6 and
coverage fell. The reason is an assumption hidden in the substitution: it treats
the outcome residual's moments as independent of the propensity, and under
confounding they are not independent — *that is what confounding means*. Units
with extreme `e` also have extreme outcomes, so the true fourth moment far
exceeds the product of the marginals. The code is kept, unused and documented,
so the next person does not have the idea twice.

**"It is the overlap clamp."** Clamping to `[0.02, 0.98]` binds on **0.69%** of
units at moderate-overlap and **1.00%** at strong confounding, and **0.00%** at
high-dim — which covers at 91.8% regardless. Not the mechanism.

### The ceiling, and why the bar stays where it is

Residual bias caps achievable coverage independently of the interval:

| cell | bias / sd | coverage ceiling |
|---|---|---|
| moderate-overlap | 0.19 | ~94.6% |
| strong-confounding | 0.27 | ~94.1% |
| high-dim | 0.02 | ~95.0% |

`P(|Z + 0.27| < 1.96)` is 94.1%, not 95%. So two of the three cells cannot reach
nominal by any interval construction, and `high-dim` — which can — is the cell
where the remaining work is well defined.

The specs stay at a **3-point** bar. Monte Carlo standard error at 300
replications is about 1.2 points, so 3 points is a real requirement. Coverage is
now 91.2–91.8%, so they still fail, by 1.2 to 1.8 points.

### What is left

- **`high-dim` is the tractable one.** Zero bias, 91.8% coverage, `k/n = 6.5%`.
  A cross-fitted (sample-split) projection would remove the remaining in-sample
  dependence that `(n - k)` only approximates.
- **The heavy cells need a different estimator, not a different interval.**
  Augmented IPW / doubly-robust estimation attacks the bias, which is the
  binding constraint there. That is new capability, not a fix to this one.
- **Surface the anti-informative property.** `variance_dof` in the single digits
  is the signal, and it is currently a field a caller has to know to read.

**Until it closes**, prefer `ols_adjusted` when the outcome model is plausibly
linear — it is nominal on every cell — and treat a narrow IPW interval under
heavy weights as the case to distrust, not the case to trust.

Specs: `c14_ipw_coverage_is_nominal_under_strong_confounding`,
`c14_att_coverage_is_nominal_under_strong_confounding`. Guards:
`few_variance_degrees_of_freedom_widen_the_interval`,
`ipw_bootstrap_agrees_on_the_point_estimate_and_labels_itself`,
`projection_correction_scales_with_the_coefficients_spent`,
`effective_dof_is_the_kish_effective_count`,
`the_rule_never_claims_more_than_satterthwaite_did`. Diagnostics:
`cargo test -p cynepic-causal --lib c14 -- --ignored --nocapture --test-threads=1`.

---

## <a id="c1"></a>C1 — singular design reported with zero uncertainty · Closed

`solve_normal_equation` returned a zero vector on a small pivot and
`invert_matrix` returned a zero matrix, neither signalling. With exactly
collinear covariates the caller received a plausible ATE and a standard error of
exactly 0.0 — infinite confidence, which passes every downstream significance
test.

**Fixed** by replacing Gaussian elimination with **Householder QR with column
pivoting**. Pivoting moves dependent columns to the end, so rank deficiency is
not merely tolerated but *detected*, and the offending columns can be named:
`EstimationError::RankDeficient { rank, expected, aliased }`.

QR also never forms `X'X`, which squares the condition number — the usual reason
a normal-equations solver loses precision that QR keeps.

Specs: `c1_rank_deficient_design_is_an_error_naming_the_aliased_columns`,
`c1_standard_error_is_never_exactly_zero`.

---

## <a id="c2"></a>C2 — identification could not fail · Closed

`BackdoorCriterion::find` returned `Option<HashSet<String>>` and in practice
always `Some`: the parents of the treatment, with no check that they were
measurable or that they blocked anything. On the crate's own front-door fixture,
where `U` is documented as unobserved, it returned `{U}`.

`None` already meant "no adjustment needed" — a *successful* identification — so
it could not also mean "not identifiable". Opposite conclusions sharing one
representation.

**Fixed** with `VarKind::{Observed, Latent}` on the DAG and
`Result<AdjustmentSet, IdentificationError>` from the criterion. Failure
distinguishes `RequiresLatent` (measure the variable) from `NotIdentifiable`
(change strategy), because the remedies differ.

Returned sets are now **verified**, not constructed: validity is checked by
d-separation on the graph with the treatment's outgoing edges deleted, rather
than assumed from a parent-set heuristic.

Specs: `c2_adjustment_set_never_contains_a_latent_variable`,
`c2_identification_can_fail`,
`c2_no_adjustment_needed_is_distinct_from_not_identifiable`.

---

## <a id="c5"></a>C5 — IPW's variance did not describe IPW's point estimate · Closed

The point estimate was Hájek (normalised by the sum of weights); the variance
summed Horvitz–Thompson contributions (`y_i / e_i`, unnormalised) and centred
them on the Hájek estimate. Different scales, so the subtraction was not a
residual and the result was not that estimator's variance.

**Fixed** with an influence-function variance derived from the same estimating
equation as the point estimate — each arm's residual taken against that arm's
own Hájek mean, scaled by that arm's own mean weight — plus the
estimated-propensity projection described in [C14](#c14). HC1 and HC3 robust
standard errors were added for OLS at the same time.

**Why the metamorphic relations never caught it.** Duplicating a dataset must
leave the estimate unchanged and shrink the standard error by √2. IPW satisfied
both halves throughout, because duplication probes how a variance scales with
`n`, and this variance got the `n` dependence right while getting the *scale*
wrong. A formula uniformly wrong by a constant factor still shrinks by √2.

Coverage caught it, because coverage compares an interval against a known truth
rather than against another interval. This is the clearest argument in the
repository for why the harness needed ground-truth simulation and not just
metamorphic relations.

Specs: `c5_ipw_intervals_achieve_nominal_coverage`,
`c5_ols_intervals_achieve_nominal_coverage`,
`c5_coverage_survives_heteroskedasticity`,
`c5_ipw_reports_an_influence_function_variance`.

---

## <a id="c7"></a>C7 — refutation verdicts were magic constants · Closed

```rust
let passed = relative_change < 0.15; // Less than 15% change
```

Never consulted the estimate's uncertainty. A 14% shift on a tightly-estimated
effect passed; a 16% shift on one spanning zero failed. `placebo_treatment`
additionally took only `(outcome, ate, tolerance)` — never the treatment, the
covariates, or the estimator — and always re-estimated with
`difference_in_means`, so it produced a verdict about an analysis the caller had
never run.

**Fixed** with a `Study` type carrying data, adjustment set and estimator, and
verdicts expressed in **standard errors**. Doubling the sample now makes the
tests harder to pass, which is what a robustness test should do.

One subtlety found while fixing it: the placebo verdict must be scored against
the **placebo run's own** standard error, not the original's. Randomising
treatment removes the treatment term from the fit, so the placebo's residual
variance legitimately includes everything the real effect used to explain.
Scoring it against the original's much smaller SE failed sound analyses about a
third of the time.

Specs: `c7_refutation_verdict_depends_on_uncertainty`,
`c7_placebo_uses_the_studys_own_estimator`.

---

## <a id="c8"></a>C8 — hand-rolled LCG · Closed

```rust
self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
```

A bare linear congruential generator, unseedable from outside, so a refutation
result could not be reproduced or varied.

**Fixed** with `rand_chacha::ChaCha8Rng` and the seed on the public API.

Spec: `c8_refutation_is_seeded_and_reproducible`.

---

## <a id="c10"></a>C10 — degenerate input panicked or invented numbers · Closed

Three paths: `difference_in_means` substituted `0.0` for a missing arm's mean
(so a single-arm dataset returned a confident effect equal to the treated mean);
`assert_eq!` on caller-supplied lengths killed any embedding service; and `n = 0`
was unguarded.

**Fixed** with a `Result`-returning API throughout and
`EstimationError::{EmptyArm, LengthMismatch, NoObservations, InsufficientData,
ConstantTreatment}`.

The regression test is deliberately broader than the original finding: rather
than enumerating known-bad cases, it throws five degenerate shapes at every
public estimator and requires each to return rather than unwind.

Specs: `c10_single_arm_data_is_an_error`, `c10_mismatched_lengths_do_not_panic`,
`c10_empty_dataset_is_an_error`, `c10_no_estimator_panics_on_degenerate_input`.

---

## <a id="c11"></a>C11 — `CausalDag` did not enforce that it is a DAG · Closed

`add_edge` returned `()` and accepted cycles and self-loops. A type named
`CausalDag` holding a cyclic graph invalidates every algorithm built on it —
d-separation, backdoor search and the counterfactual engine all assume
acyclicity and will silently produce nonsense or fail to terminate.

**Fixed** — `add_edge` returns `Result<(), DagError>` and rejects any edge whose
target already reaches its source, naming the cycle. The check runs *before*
insertion, so the graph is valid at every point rather than inserted and rolled
back. Deserialisation goes through the same path, so a serialised cyclic graph
is rejected on load.

Specs: `c11_cycles_are_rejected_at_construction`, `c11_self_loops_are_rejected`,
`c11_acyclicity_holds_under_arbitrary_insertion_orders`.

---

## <a id="c12"></a>C12 — unknown variables reported as d-separated · Closed

`d_separated` returned `true` for names the graph had never heard of. `true`
means "conditionally independent", so a typo read as a *positive* finding of
independence — the answer most likely to be acted on.

**Fixed** — returns `Result<bool, DsepError>`, and checks the conditioning set
too. That last part matters most in practice: adjustment sets are usually
assembled programmatically, and one mismatched name silently validated the whole
set.

Specs: `c12_unknown_variable_is_an_error`,
`c12_unknown_conditioning_variable_is_an_error`.

---

## <a id="c13"></a>C13 — the propensity model never converged · Closed

The largest defect in the crate, invisible until coverage was measured. The
logistic regression ran a **fixed 100 iterations of gradient descent at lr=0.1**
with no convergence check and no way for a caller to find out.

| configuration | bias |
|---|---|
| 100 iterations (as shipped) | **+1.80** |
| 10,000 iterations | +0.26 |
| 500,000 iterations | +0.26 |
| oracle propensity, no fitting | +0.13 |
| naive difference in means (no adjustment at all) | +3.40 |

```
beta after 100 iterations   [0.02, 0.90, 0.78, 0.76]
true logit coefficients     [0.00, 1.43, 1.43, 1.43]
```

Non-convergence accounted for **86%** of the bias. The estimator removed about
47% of the confounding it exists to remove, while reporting a 95% interval 0.4
wide. Measured coverage was 0.0% on seven of eight cells.

**Why the test suite missed it.** `ipw_known_effect` asserted
`|ate - 5.0| < 2.0` — a 40% tolerance — on one binary covariate with propensities
of 0.25 and 0.75, a problem easy enough that 100 iterations does converge. The
replacement test asks for 2%.

**Fixed** with IRLS / Fisher scoring, converging in under ten iterations, and
`EstimationError::NotConverged` rather than a silent partial fit. The fitted
model is now returned by `fit_propensity` so a caller can check whether the
propensity model recovered the assignment mechanism separately from whether the
effect estimate is right.

Two things were found while fixing it:

- **Separation needs its own detection.** Under perfect separation the
  coefficients diverge but the *score vanishes* as fitted probabilities
  saturate, so IRLS reports convergence at an arbitrary finite point. The
  gradient test cannot see it. `EstimationError::Separation` checks for fitted
  probabilities pinned at 0 or 1.
- **Overlap is checked, not clipped.** Silently clamping propensities to
  `[0.01, 0.99]` converts a violated assumption into a plausible number.
  `EstimationError::InsufficientOverlap` now refuses when more than 10% of units
  fall outside `[0.02, 0.98]` — which is why the `weak-overlap` cell shows
  `refused=300` rather than a confident wrong answer.

Specs: `c13_ipw_removes_most_of_the_confounding`,
`c13_propensity_fit_converges_and_reports_it`,
`c13_weak_overlap_is_refused_or_diagnosed`.

---

## <a id="measured-coverage"></a>Measured coverage

`cargo run -p cynepic-causal --example coverage_report --release`, 300
replications per cell, nominal 95%.

### `LinearATEEstimator::ols_adjusted` — nominal on every cell

| cell | bias | coverage | width |
|---|---|---|---|
| benign | −0.0010 | 95.0% | 0.181 |
| strong-confounding | +0.0019 | 96.7% | 0.207 |
| moderate-overlap | +0.0023 | 96.3% | 0.205 |
| weak-overlap | +0.0064 | 94.3% | 0.281 |
| nonlinear | −0.0158 | 93.0% | 0.523 |
| heteroskedastic | −0.0032 | 94.7% | 0.339 |
| heterogeneous-effects | −0.0001 | 97.7% | 0.226 |
| small-n | −0.0044 | 94.3% | 0.745 |
| high-dim | +0.0043 | 97.0% | 0.418 |

### `PropensityScoreEstimator::ipw` — nominal on 7 of 8 estimable cells

| cell | bias | coverage | width |
|---|---|---|---|
| benign | −0.0006 | 94.7% | 0.195 |
| strong-confounding | +0.0331 | **87.3%** | 0.575 |
| moderate-overlap | +0.0170 | 92.3% | 0.552 |
| weak-overlap | — | *refused, 300/300* | — |
| nonlinear | −0.0173 | 93.7% | 0.568 |
| heteroskedastic | −0.0033 | 94.7% | 0.355 |
| heterogeneous-effects | +0.0003 | 98.0% | 0.242 |
| small-n | +0.0007 | 95.0% | 0.825 |
| high-dim | +0.0024 | 93.0% | 0.831 |

`refused` is a result, not a gap. Weighting cannot manufacture a comparison the
data does not contain; on the weak-overlap cell (71% of units outside the
overlap bounds, effective sample size ~3 of 2000) refusing is correct, and the
previous implementation's confident numbers there were the failure.

`att` shares the weighting machinery and shows the same pattern: nominal on six
of eight, under-covering on `strong-confounding` (86.7%) and `moderate-overlap`
(89.3%). Both are C14.

### For comparison — before Tier 1

```
 LOW  benign                bias=+1.8123  coverage=0.0%   width=0.400
 LOW  strong-confounding    bias=+3.3053  coverage=0.0%   width=0.260
 LOW  weak-overlap          bias=+3.5190  coverage=0.0%   width=0.259
 LOW  high-dim              bias=+9.6242  coverage=0.0%   width=1.241
```

## A note on the DGP itself

Fixing the estimators exposed a defect in the *test harness*: `overlap` was
applied to an unstandardised assignment index, so its meaning drifted with `p`
and `confounding`. Raising `p` from 3 to 25 silently turned a well-overlapped
world into a positivity violation, which meant the `high-dim` cell was measuring
overlap while claiming to measure dimensionality. The index is now standardised
by `sqrt(p)` and `overlap = 1.0` really is benign.

Related, and documented in `cynepic_testkit::dgp`: **effective sample size is
non-monotonic in overlap** and cannot diagnose a positivity violation alone. It
collapses to ~1 at `overlap = 0.03` and then *recovers to 613* at
`overlap = 0.01`, where 95% of units are extreme — because under
near-deterministic assignment almost every unit lands in the arm it was nearly
certain to get, so its weight is ≈1. A monitor thresholding on ESS alone passes
the worst case and fails the middling one. The extreme-propensity fraction is
monotone and is the diagnostic to threshold on.

## Reproducing

```bash
cargo test -p cynepic-causal --test findings                  # closed findings
cargo test -p cynepic-causal --test findings -- --ignored     # C14, still failing
cargo run  -p cynepic-causal --example coverage_report --release
./scripts/findings-ratchet.sh                                 # the CI gate
```

Every DGP is seeded with ChaCha8, so any failure is reproducible from the seed
printed in its message, on any platform.
