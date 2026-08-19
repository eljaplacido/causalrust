# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
with the pre-1.0 caveat that minor versions may break the API.

## [Unreleased]

The theme of this release is that the statistical claims became **measurable**,
and then most of them turned out to be wrong. Every number below is reproducible
from a seeded command; see [docs/FINDINGS.md](causalrust/docs/FINDINGS.md).

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

- **C14** (open) — IPW/ATT intervals under-cover under strong confounding
  (ATT 89.3% against nominal 95%). The standard error is *noisy*, not biased: a
  coefficient of variation of 0.26 implies ~7.5 effective degrees of freedom
  despite n=2000. A Satterthwaite t-interval recovers most of it; the residual
  needs a cross-fitted projection.
- **R1** (open) — the keyword classifier scores macro F1 0.290 against a 0.25
  random baseline, with **0.000 recall on Chaotic**: 78% of natural phrasing
  matches no keyword. This is a reach problem requiring the embedding
  classifier, not a tuning problem. It abstains rather than guessing, which is
  what makes it survivable behind an escalation policy.
- **Performance figures in `causalrust/README.md` and `docs/PITCH.md` are
  unverified design targets**, marked as such in place. They cite a
  `benchmarks/` directory that does not exist.
- Not published to crates.io, so `cargo-semver-checks` has no baseline.
- `wasm32-unknown-unknown` (browser) builds for nothing yet;
  `wasm32-wasip1` is gated in CI for core, causal and bayes.

## [0.2.0] — 2026-03-13

Initial workspace: six crates covering Cynefin routing, causal inference,
Bayesian priors and sampling, policy guardrails, and typed workflow graphs.

[Unreleased]: https://github.com/eljaplacido/causalrust/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/eljaplacido/causalrust/releases/tag/v0.2.0
