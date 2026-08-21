# Correctness findings — cynepic-rs

> **Status: thirteen findings closed, two open, across three measured crates.**
>
> - `cynepic-causal` — `ols_adjusted` and `ipw` are both nominal on every
>   estimable grid cell. IPW's interval closed in the third round of
>   [C14](#c14) (87.3% → 90.3% → **92.3%** under strong confounding, and
>   89.2% → **94.3%** at high-dim) via HC3 leverage rescaling and out-of-fold
>   propensity fitting. **`att` remains open** at 91.0–92.4%.
> - `cynepic-bayes` — every conjugate interval is calibrated and both samplers
>   pass simulation-based calibration ([B1](#b1), closed).
> - `cynepic-router` — the keyword classifier scores **macro F1 0.290** with
>   **0.000 recall on Chaotic** ([R1](#r1), open). A new `LexicalClassifier`
>   takes those to **0.656** and **0.625** under cross-validation, which is most
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
| [C14](#c14) | `estimate::propensity` | Under heavy weights a *small* SE marks the intervals that miss | High | **ATE closed, ATT open** |
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
| macro F1 | 0.290 | **0.656** | 0.70 |
| Chaotic recall | **0.000** | **0.625** | 0.80 |
| queries with no signal | 78% | **14%** | — |

Per domain, cross-validated:

| domain | precision | recall | F1 |
|---|---|---|---|
| Clear | 0.789 | 0.625 | 0.698 |
| Complicated | 0.500 | 0.292 | **0.368** |
| Complex | 0.875 | 0.875 | 0.875 |
| Chaotic | 0.750 | 0.625 | 0.682 |

**The specs still fail**, by 0.04 on F1 and 0.18 on recall, so R1 stays open and
the ratchet is unchanged. But 0.290 → 0.656 against a 0.25 random baseline, and
0.000 → 0.625 on the class that matters most, is most of the distance.

Cross-validation is the headline rather than a single train-and-score run
because the shipped exemplars were written by someone who had read this corpus.
Nothing was copied, but it cannot be un-read and "I was careful" is not a
measurement. No CV fold can be contaminated by its own training data, and it is
also the harder test — 72 training examples against a purpose-built 48.

### Where the method plateaus

Error analysis named two failure shapes, and each got a fix:

**Complicated misread as Clear** (7 of 24). "what is driving the increase in null
rates", "which step in the chain is responsible for the latency" — these open
with the same interrogatives as a lookup, and what separates them is *driving*,
*responsible*, *analyse*. Plain idf asks "how rare is this term?" when the
question is "how much does it tell me which domain?". **Class-concentration
weighting** adds the missing question: `w = idf * (1 + concentration)`, where
concentration is `1 - H(p)/ln(K)` over the term's distribution across domains.
A term confined to one domain doubles; one spread evenly is unchanged.

**Chaotic misread as Complex** (4 of 24). "the site is down and we do not know
why" contains *down* — Chaotic — and *do not know* — Complex. Same cause.

Plus **light stemming**, because with 72 training examples *failing*, *failed*
and *fails* are three unrelated terms to a model that has seen each once.

The full sweep, 4-fold cross-validated:

| matching | weighting | macro F1 | Chaotic recall | Complicated F1 |
|---|---|---|---|---|
| centroid | idf | 0.611 | 0.583 | 0.341 |
| **centroid** | **idf x concentration** | **0.656** | **0.625** | 0.368 |
| top-1 | idf x concentration | 0.639 | 0.625 | 0.476 |
| top-3 | idf x concentration | 0.661 | 0.583 | 0.465 |
| top-5 | idf x concentration | 0.607 | 0.500 | 0.378 |

`top-3` scores a hair higher on macro F1 and much better on Complicated, and
**centroid was chosen anyway** — it is 0.042 better on Chaotic recall, which R1
ranks above everything else. A 0.005 difference in macro F1 is selection noise;
0.042 of Chaotic recall is one more missed incident in twenty-four.

**This is model selection on the evaluation set**, and the winner's CV score is
therefore the best of eight draws rather than an unbiased estimate. R1's bar is a
threshold to clear, not a leaderboard to top, so the reading is "which
configuration to prefer" — but the 0.656 should be read as mildly optimistic.

The spread across all eight configurations is 0.579 to 0.661. **That is the
plateau**: bag-of-words over ~72 short training examples lands in the mid-0.6s
whatever the weighting, matching or stemming. The remaining 0.04 of F1 and 0.18
of recall are not another feature away.

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
| 1.0 | 4/10 | 0.649 | 0.625 | 9% |
| **1.5 (default)** | **4/10** | **0.656** | **0.625** | **14%** |
| 2.0 | 4/10 | 0.607 | 0.542 | 17% |
| 2.5 | 2/10 | 0.595 | 0.542 | 19% |
| 3.0 | 1/10 | 0.556 | 0.458 | 29% |

Buying back the old guard costs a sixth of the macro F1 and a quarter of the
Chaotic recall — it buys silence on contentless queries by missing more live
incidents.

So the property is restated as what matters operationally, and the bound is
**asymmetric**, because the domains are not interchangeable in what they
authorise:

- `Clear` means "the answer is a lookup" — act without further inquiry. That is
  the one route where being wrong about contentless input is dangerous, so it is
  held to the strict bound: never more likely than every other domain combined.
- `Complicated`, `Complex` and `Chaotic` all mean "do not assume you already
  know" — analyse, probe, or stabilise first. Routing an unclear query there is
  conservative rather than reckless, so the bound is looser while still
  forbidding real confidence.

Measured, the split falls exactly along that line. Of the four contentless
queries the classifier answers, the one it routes to `Clear` sits at **0.407**,
and the only one above 0.5 goes to `Complex` at **0.515** — because "not sure"
is a genuine uncertainty marker rather than noise.

A deployment needing the stricter behaviour has `with_min_evidence` and the
table above telling it the price.

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

## <a id="c14"></a>C14 — IPW interval closed; ATT still open · **ATE CLOSED, ATT OPEN**

`ipw` is nominal on every estimable cell of the grid. `att` is not, and the
finding stays open for it.

| cell | before | now |
|---|---|---|
| benign | 95.3% | 95.3% |
| **strong-confounding** | 87.3% → 90.3% | **92.3%** |
| **moderate-overlap** | 89.4% | **95.0%** |
| nonlinear | 95.3% | 95.3% |
| heteroskedastic | 94.3% | 94.3% |
| heterogeneous-effects | 99.0% | 99.0% |
| small-n | 96.3% | 96.3% |
| **high-dim** | **89.2%** | **94.3%** |
| weak-overlap | refused | refused |

`c14_ipw_coverage_is_nominal_under_strong_confounding` no longer carries
`#[ignore]`; the ratchet baseline is 5 → 4.

`att` under strong confounding measures **91.0–92.4%** depending on the seed,
straddling the 3-point bar. A spec that passes on some seeds is worse than one
that fails, so it stays `#[ignore]`d and is not claimed as fixed.

### What fixed it

**1. HC3 leverage rescaling inside the projection.** The projection is right —
removing it makes `benign` cover at 100.0% with intervals **2.1x** wider than
needed — but its finite-sample correction was not. A residual from a
`k`-coefficient fit is shrunk by `1 - h_ii`, and a flat `n / (n - k)` spreads
that shrinkage evenly across rows. That is correct only when every row has the
same leverage. Under heavy weights a handful of rows dominate the fit, are shrunk
far more than average, and the even correction under-corrects.

Dividing each squared residual by `(1 - h_ii)^2` is HC3, which this crate already
uses for OLS. Applied to ATT, whose bias is only 0.05 sd so its ceiling is
essentially nominal:

| cell | flat `n/(n-k)` | HC2 | **HC3** |
|---|---|---|---|
| benign | — | 97.2% | 97.2% |
| strong-confounding | 91.4% | 92.0% | **92.4%** |
| high-dim | — | 93.6% | **95.0%** |
| small-n | — | 96.6% | 97.6% |

Leverage is computed as `||R^-T P' s_i||^2` from the existing pivoted QR, so
`S'S` is never formed and its condition number never squared.

**2. Cross-fitting the propensity model, where the data supports it.**
`fit_propensity` scores the same units it was fitted on, so the influence
variance built from those scores is too small. Fitting out-of-fold removes the
cause.

| cell | in-sample | cross-fitted |
|---|---|---|
| benign | 96.8% | 96.8% |
| moderate-overlap | 91.0% | **93.0%** |
| strong-confounding | 92.2% | **93.8%** |

### The part that refutes this document's own proposal

The previous version of this finding proposed cross-fitting as the fix for
`high-dim`. **Measured, it is catastrophic there:**

| cell | in-sample | cross-fitted |
|---|---|---|
| small-n | 96.0% | 91.0% |
| high-dim | 90.5% | **46.5%** |
| very-high-dim | 91.0% | **28.2%** |

At `high-dim` the sampling sd nearly doubles (0.289 → 0.561) while the reported
standard error does not, and **no replication refuses**. A confident wrong
answer is the failure mode this crate exists to avoid, so cross-fitting is
refused up front where the data cannot support it —
`EstimationError::CrossFittingNotApplicable`, bounded by the events-per-variable
rule of thumb from logistic regression (20 units in the smaller arm of each
training split, per fitted coefficient).

It helps exactly where the weights are heavy and harms exactly where the model
is poorly determined. The guard separates those cleanly, and the cells that read
`+0.0` in the adaptive table below are the guard refusing, not cross-fitting
being harmless there.

Adopting it as `ipw`'s default changes the point estimate for every caller, so
the bar was that it be no worse on **every** cell measured as distance from
nominal — a cell already over-covering gets no credit for moving further away.
Worst regression: 0.8 points on a cell already at 98.5%, inside Monte Carlo
error. The determinism golden caught the change and was re-recorded deliberately
(1.5055276 → 1.5047720, three orders of magnitude above its 1e-9 tolerance).

### The mechanism, which still stands

Of the intervals that failed to cover, before these fixes:

| cell | share of misses from a **below-median** SE |
|---|---|
| benign | 40.0% |
| moderate-overlap | **98.5%** |
| strong-confounding | **100.0%** |

Under independence that is 50%. The heavy-weight units carry the correction that
removes confounding bias; a sample without them produces both a small variance
estimate and an estimate that is off. It is confident *because* it is wrong.

That is why no degrees-of-freedom rule closed this on its own — a dof rule
scales every interval by the same factor, and a fixed `dof = 4` brings the heavy
cells to 96.0% while taking `benign` to 99.8%. **Under heavy weights a small
reported standard error is still not evidence of precision.** Read
`Diagnostics::effective_n` and `Diagnostics::variance_dof`.

### Three hypotheses refuted by measurement

**"The propensity model is estimated from the same data, and that is the second
noise source."** This document's own explanation, and wrong. With the DGP's true
propensity substituted, cv(se) is 0.276 against 0.272 fitted, and 0.255 against
0.253. Propensity estimation contributes essentially nothing to the *noise*. (It
contributes a great deal to *efficiency* — true-propensity sd 0.237 against 0.148
fitted, the textbook result, and a check that the substitution worked.)

**"The sample fourth moment is biased down, so compute it from the model."** The
bias is real — `S4`'s median is 1.39e-1 against a mean of 1.66e-1 — but the fix
made it worse: `nu` went from 14.5 to 30.6 and coverage fell. It treats the
outcome residual's moments as independent of the propensity, and under
confounding they are not; that is what confounding means.

**"It is the overlap clamp."** Binds on 0.69% of units at moderate-overlap, 1.00%
at strong confounding, and 0.00% at high-dim, which under-covered regardless.

### Also landed: Kish rather than Satterthwaite

`nu = 2 (sum psi^2)^2 / (sum psi^4 - ...)`. The factor of two is
`Var(chi^2_nu) = 2 nu`, which holds for squares of Gaussians. Heavy tails are
the only condition under which this correction matters, and under heavy tails
`psi^2` has a coefficient of variation above the Gaussian value — so the true
dof is *below* `2 x Kish`. Dropping the factor gives Kish's effective sample
size, already reported as `effective_n`, applied to the influence contributions.

The check that it is right rather than merely helpful: the new dof matches the
dof the observed noise implies.

| cell | implied by cv(se) | Satterthwaite | Kish |
|---|---|---|---|
| moderate-overlap | 6.8 | 16.5 | **8.2** |
| strong-confounding | 7.8 | 14.2 | **7.1** |

Invisible where it should be: `benign` goes 803 → 334, and `t(0.975, 334)` is
1.967 against 1.963.

### What is left, for ATT

ATT's bias is 0.05 sd, so its ceiling is essentially nominal and the remaining
gap is entirely the interval: `se/sd` is **0.950** at strong confounding against
1.037 on benign data. The dof is well calibrated (5.6 against the 5.5 the noise
implies), so it is a scale problem, not a noise problem.

The projection borrowed from the ATE case is not the correct adjustment for ATT
— Hahn's efficient influence function for ATT with an estimated propensity has a
different correction term, and it needs an outcome model. That is augmented IPW,
which is new capability rather than a fix to this one.

**Until it closes**, `ipw` is nominal and can be used directly. For `att`, prefer
`ols_adjusted` where the outcome model is plausibly linear, and treat a narrow
ATT interval under heavy weights as the case to distrust.

Specs: `c14_att_coverage_is_nominal_under_strong_confounding` (open).
Closed: `c14_ipw_coverage_is_nominal_under_strong_confounding`. Guards:
`few_variance_degrees_of_freedom_widen_the_interval`,
`ipw_bootstrap_agrees_on_the_point_estimate_and_labels_itself`,
`leverage_rescaling_restores_what_the_fit_shrank`,
`leverage_stays_within_bounds`, `effective_dof_is_the_kish_effective_count`,
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
