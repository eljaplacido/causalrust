# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
with the pre-1.0 caveat that minor versions may break the API.

## [Unreleased]

### Added

- **Causal estimation from Python.** `estimate_ate`, `estimate_ate_weighted`
  and `estimate_att`, returning a `CausalEffect` that carries its estimand,
  population, standard-error kind, interval and weighting diagnostics — the
  flagship capability was previously Rust-only.
- **`statsmodels` comparison** (`scripts/compare_statsmodels.py`) — the speed
  question a data scientist actually asks. Faster on seven of nine tasks;
  the two losses are in the table.
- **Router latency benchmark** (`examples/router_latency`) — the routing path
  had never been timed.
- **`audit_trail` and `monitor_drift` implemented** in the MCP server. Both were
  advertised in the manifest; `audit_trail` returned "Unknown tool".

### Fixed

- **`cynepic-router::drift` was never declared as a module.** The file existed,
  the feature was advertised in the crate docs and the MCP manifest, and none of
  it compiled — including four tests that had never run.
- **The MCP server classified with the weak keyword classifier.** Now uses
  `LexicalClassifier`, and returns the terms behind each verdict.
- **`docs/integration.md` documented a Python API that does not exist.**
  `from cynepic import BackdoorCriterion, PolicyChain` was an ImportError.
  Rewritten around the four real surfaces, with every example executed.


## [0.3.0] — 2026-08-20

The theme of this release is that the claims became **measurable**, and then
most of them turned out to be wrong. Every number below is reproducible from a
seeded command; see [docs/FINDINGS.md](causalrust/docs/FINDINGS.md).

Thirteen correctness findings closed, two open and quantified. The findings
ratchet went 18 → 4.

### Added

- **`cynepic-testkit`** (internal, `publish = false`) — the validation harness.
  Seeded ground-truth data-generating processes over the seven knobs that break
  estimators; bias, RMSE and **confidence-interval coverage** over replications;
  metamorphic relations; Bayesian credible-interval coverage and
  **simulation-based calibration**; a 96-query labelled routing corpus. Each
  harness is self-validated — fed a deliberately wrong input and asserted to
  complain.
- **`docs/FINDINGS.md`** — single source of truth for every correctness finding,
  with measurements.
- **`scripts/findings-ratchet.sh`** — CI gate over five test suites. The open
  count may only go down, the specs must still compile, and every open spec must
  still fail.
- **`ATEResult` provenance.** Private fields, no public constructor. Every
  estimate carries its `Estimand` (ATE / ATT / ATC / LATE), its `StdErrorKind`,
  and `Diagnostics` from the fit — rank, convergence, propensity range, Kish
  effective sample size, first-stage F, and the effective degrees of freedom of
  the variance estimate.
- **Student-t confidence intervals** when the estimator reports few effective
  degrees of freedom for its variance.
- **`PropensityScoreEstimator::att`** and **`ipw_bootstrap`**.
- **`BetaBinomial::credible_interval`** and **`GammaPoisson::credible_interval`**
  — exact quantiles, via a new `cynepic-core::special` (log-gamma, incomplete
  beta and gamma, Student-t).
- **Seeded MCMC.** `with_seed()` on `MetropolisHastings`, `MultiDimMH` and
  `AdaptiveMH`.
- **`CausalDag::mark_latent`** and `VarKind`, so identification can decline.
- **Criterion benchmarks** for the estimators and samplers. MCMC is measured in
  *time per effective sample*, never samples per second.
- **`[workspace.lints]`** — the project's conventions as build failures, with
  remaining debt recorded as measured counts.
- **Apache-2.0** per-crate `LICENSE` and `NOTICE`, CI-asserted in every tarball.
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue and PR
  templates, Dependabot.

### Added — measurement infrastructure

- **Cross-implementation parity** (`crates/cynepic-causal/tests/parity.rs`) —
  258 cases against numpy, scipy and networkx, fixtures committed so the suite
  needs no Python. Worst quantile disagreement 5.5e-12 across 116 cases; 134
  d-separation verdicts match `networkx.is_d_separator` exactly.
- **Refusal measurement** (`tests/refusal.rs`, `examples/refusal_report`) — the
  false-answer rate, target zero, measured at zero. On an exactly collinear
  design `numpy.linalg.lstsq` returns **−0.000562** with no error or warning;
  this crate returns `RankDeficient { aliased: ["covariate[2]"] }`.
- **Reproducibility suites** — replay must be bit-identical; portability is
  bit-identical only for the operations IEEE-754 requires to be correctly
  rounded. Committed bit patterns for the arithmetic-only estimators are
  checked on ubuntu, macOS and windows every push.
- **Comparison harnesses** — `scripts/compare_networkx.py`,
  `compare_langgraph.py`, `compare_opa.py`, plus `causal_latency`,
  `bayes_latency`, `graph_latency` and `guardian_latency` examples.
- **41 tests for the three surfaces** — `cynepic-server`, `cynepic-mcp` and
  `cynepic-pyo3` previously had none between them.
- **`LexicalClassifier`** (`cynepic-router`) — tf-idf over unigrams and bigrams
  with class-concentration weighting, nearest centroid, and an inspectable
  `explain`. No model file, no inference runtime, no new dependency.
- **`PropensityScoreEstimator::ipw_cross_fitted`** and
  `cross_fitted_propensity`, with an events-per-variable guard that refuses
  where splitting the sample would make the estimate worse.
- **`ToolBeliefSet::len`/`is_empty`** (`cynepic-bayes`).

### Fixed

Eleven correctness findings closed. Measured interval coverage, 300
replications, nominal 95%:

| estimator | before | after |
|---|---|---|
| `ols_adjusted` | 94.7–97.3% | **93.0–97.7%, all 9 cells** |
| `ipw` | **0.0% on 7 of 8 cells** | **nominal on 8 of 8** |
| `ipw` bias | +1.8 to +9.6 | < 0.04 |

- **C1** — a rank-deficient design returned a plausible ATE with a standard error
  of exactly `0.0`. Householder QR with column pivoting now detects it and names
  the aliased columns.
- **C2** — identification could not fail, and returned adjustment sets containing
  unobservable variables. Now `Result<AdjustmentSet, IdentificationError>`, with
  sets verified by d-separation rather than assumed from a heuristic.
- **C5** — IPW paired a Hájek point estimate with a Horvitz–Thompson variance.
  Replaced with an influence-function variance derived from the same estimating
  equation, plus HC1/HC3 for OLS.
- **C7 / C8** — refutation verdicts were a hardcoded `relative_change < 0.15`
  that never consulted the estimate's uncertainty, over an unseedable LCG. Now
  expressed in standard errors, over `ChaCha8Rng` with the seed on the API, and
  taking the estimator it is asked to refute.
- **C10** — degenerate input panicked or invented numbers. `Result` throughout.
- **C11** — `CausalDag` accepted cycles and self-loops. Rejected at insertion,
  including on deserialisation.
- **C12** — `d_separated` reported unknown variables as *independent*, turning a
  typo into a positive finding. Now `Result`, and the conditioning set is checked
  too.
- **C13** — the propensity model ran 100 fixed gradient-descent steps with no
  convergence check, stopping at ~55% of the true coefficients and leaving 53%
  of the confounding in place. IRLS converges in under ten. Separation and
  insufficient overlap are now errors rather than silently clipped.
- **B1** — `BetaBinomial`'s credible interval was a normal approximation to a
  Beta: 89.4% conditional coverage at p=0.5 where the exact interval gives 97.8%.
- **B2** — MCMC samplers drew from the thread-local entropy generator, so results
  were not reproducible.
- **G1** — the circuit breaker had no half-open state. After the reset timeout
  **100 of 100** callers were admitted, delivering the thundering herd it exists
  to prevent, on a timer.
- **R1 (partial)** — classifier scoring depended on how many keywords an author
  happened to write. Now scored by matched query coverage. *This fixed a real
  bug and moved no headline number*; the binding constraint is elsewhere.
- Entropy scoring, silently dropped from the router in an earlier merge,
  restored.
- **CI had never run.** Every workflow since March failed in under 10 seconds on
  an action that does not exist (`dtolnay/rust-action`; it is
  `dtolnay/rust-toolchain`). `main` did not compile: 32 errors across the PyO3,
  MCP and HTTP components, all fixed.
- `pyo3` 0.24 → 0.29 (RUSTSEC-2026-0176, RUSTSEC-2026-0177); `rand` ≥ 0.9.3
  (RUSTSEC-2026-0097).

### Fixed — this round

- **C14, the IPW interval.** Nominal on every estimable grid cell. Strong
  confounding 87.3% → 90.3% → **92.3%**, moderate overlap 89.4% → **95.0%**,
  high-dim **89.2% → 94.3%**. Two causes: the projection's residual sum of
  squares was corrected by a flat `n/(n-k)` where leverage is uneven (now HC3),
  and the propensity model scored the units it was fitted on (now out-of-fold
  where the data supports it). `att` remains open at 91.0–92.4%.
- **A d-separation defect found by the parity work.** `d_separated(X, Y | {X})`
  returned `Ok(true)` — a positive finding of independence for a query that has
  no answer. `networkx.is_d_separator` raises. Now an error.
- **A Python circuit breaker that could not trip.** `PyCircuitBreaker::
  record_failure` and `record_success` had empty bodies, so `is_open` was
  always `False`.
- **`ToolBeliefSet.__repr__`** reported `tools=0` regardless of contents, and
  `cynepic.__version__` was a hardcoded literal. Both now derive from reality.
- **A 200-iteration bisection loop** in `cynepic-core::special` where ~60
  exhausts `f64`. Credible intervals were **1.2x slower than scipy**; they are
  now 2.9x faster, with bit-identical output.
- **Example binaries collided** on one `target/debug/examples/` path, which
  fails intermittently on Windows. Renamed per crate.

### Changed — this round

- **Version 0.2.0 → 0.3.0.** Pre-1.0, so a minor bump carries the breaking
  changes listed below.
- **`ipw` and `att` now cross-fit the propensity model** where the data
  supports it, which changes the point estimate. Adopted only after measuring
  every grid cell as distance from nominal; worst regression 0.8 points on a
  cell already over-covering at 98.5%.
- **Effective degrees of freedom use Kish's effective count**, not
  Satterthwaite's — the factor of two assumes squares of Gaussians, and heavy
  tails are the only condition under which the correction matters at all.
- **`docs.rs` metadata** on all six published crates, so optional features are
  documented rather than silently absent.
- **README performance table** carries measured figures where they exist, and
  marks two rows "not a fair comparison" rather than deleting them.

### Changed — breaking

- Every estimator returns `Result`. `ATEResult` fields are private; use the
  accessors.
- `CausalDag::add_edge` and `d_separated` return `Result`.
- `BackdoorCriterion::find` and `FrontDoorCriterion::find` return
  `Result<AdjustmentSet, IdentificationError>` rather than `Option<HashSet>`.
- `refute`'s free functions are replaced by `Refuter` and `Study`.
- `BetaBinomial::new`, `NormalNormal::new` and `GammaPoisson::new` return
  `Result`.
- `pyo3`'s `extension-module` is now an opt-in feature. Build wheels with
  `maturin build --features extension-module`.
- Licence changed from BSL 1.1 to **Apache-2.0**. The BSL already named
  Apache-2.0 as its Change License; this brings the date forward, so no
  downstream user loses a right.

### Known issues

- **C14** (open, ATE half closed) — `ipw` is nominal on every estimable cell;
  **`att` under-covers at 91.0–92.4%** against a 3-point bar. ATT's bias is
  0.05 sd, so its ceiling is nominal and the gap is entirely the interval: the
  projection borrowed from the ATE case is not the correct adjustment for ATT.
  Closing it needs an outcome model — augmented IPW — which is new capability
  rather than a fix. Prefer `ols_adjusted` for ATT where the outcome model is
  plausibly linear.
- **R1** (open) — `LexicalClassifier` scores macro F1 **0.656** and Chaotic
  recall **0.625** under 4-fold cross-validation, against a bar of 0.70 and
  0.80 and a random baseline of 0.25. The keyword classifier it replaces
  scored 0.290 and **0.000**. Across eight configurations the spread is
  0.579–0.661, so bag-of-words over ~72 short training examples has plateaued;
  the rest is what an embedding classifier is for.
- **Under heavy propensity weights, a small standard error is not evidence of
  precision.** At strong confounding, 100% of the intervals that failed to
  cover were the ones reporting a *below-median* standard error. Read
  `Diagnostics::effective_n` and `variance_dof`; a narrow interval in that
  regime is the case to distrust.
- **Per-call allocation counts are unmeasured** — they need a counting
  `GlobalAlloc`, which `forbid(unsafe_code)` refuses. Peak RSS is measured.
- **The built Python wheel has no Python-level test.** The Rust side of the
  binding is covered; `maturin build` output has never been imported in CI.
- Not published to crates.io, so `cargo-semver-checks` has no baseline.
- `wasm32-unknown-unknown` (browser) builds for nothing yet;
  `wasm32-wasip1` is gated in CI for core, causal and bayes.

## [0.2.0] — 2026-03-13

Initial workspace: six crates covering Cynefin routing, causal inference,
Bayesian priors and sampling, policy guardrails, and typed workflow graphs.

[Unreleased]: https://github.com/eljaplacido/causalrust/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/eljaplacido/causalrust/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/eljaplacido/causalrust/releases/tag/v0.2.0
