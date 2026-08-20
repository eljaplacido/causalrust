# Completion Roadmap — cynepic-rs

## Current State (v0.2.0-dev)

**317 tests** across 10 crates. Five of six publishable crates have a validation
harness capable of failing; four of them did. PyO3, MCP and HTTP surfaces exist
and compile.

| Crate | Status | Evidence |
|---|---|---|
| `cynepic-causal` | Measured | OLS nominal on 9/9 coverage cells; IPW nominal on 8/8 estimable |
| `cynepic-bayes` | Calibrated | Exact conjugate intervals; both samplers pass simulation-based calibration |
| `cynepic-guardian` | Property-tested | 15 guardrail properties |
| `cynepic-graph` | Property-tested | 15 execution properties, no findings |
| `cynepic-router` | Measured, weak | macro F1 0.290 vs a 0.25 random baseline |
| `cynepic-core` | Types + `special` | Checked against closed forms |

**The roadmap is subordinate to [FINDINGS.md](FINDINGS.md).** Two findings are
open and both gate a 1.0:

- **C14** — IPW/ATT intervals under-cover under strong confounding (ATT 89.3%).
- **R1** — the keyword classifier has 0.000 recall on Chaotic; 78% of natural
  phrasing matches no keyword. A reach problem, not a tuning one.

Closing those matters more than anything below.

---

## Phase 1: Hardening & CI (v0.2.0) — COMPLETE

### 1.1 Build & CI
- [x] CI that actually runs. Every workflow from March to August failed in under
      10 seconds on an action reference that does not resolve
      (`dtolnay/rust-action`; it is `dtolnay/rust-toolchain`), so nothing had
      ever been built or tested. 11 jobs now, gated behind one `ci-ok` check
- [x] 3-OS x 2-toolchain test matrix (ubuntu, macOS, windows; stable and 1.85)
- [x] Lint: fmt, clippy on both feature configurations, rustdoc link checking
- [x] Feature powerset via `cargo hack`
- [x] MSRV verified, and the manifest checked against the workflow's pin
- [x] Supply chain: `cargo deny` over advisories, licences, bans, sources
- [x] Findings ratchet across five suites
- [x] `wasm32-wasip1` gated for core, causal, bayes
- [x] Coverage, miri, and package verification (LICENSE + NOTICE in every tarball)
- [x] `msvc_spectre_libs` no longer blocks Windows — the whole matrix is green
- [ ] `rust-toolchain.toml` pinning MSRV

### 1.2 Completed Implementations (was "stubs")
- [x] `cynepic-causal`: Full OLS with covariate adjustment
- [x] `cynepic-causal`: d-separation (Bayes-Ball)
- [x] `cynepic-causal`: Front-door criterion
- [x] `cynepic-causal`: Propensity score IPW estimation
- [x] `cynepic-causal`: Instrumental variable 2SLS estimation
- [x] `cynepic-causal`: Multiple refutation tests (placebo, random cause, subset, bootstrap)
- [x] `cynepic-graph`: Per-node timeout via `tokio::time::timeout`
- [x] `cynepic-graph`: Cycle detection before execution (DFS)
- [x] `cynepic-graph`: Checkpointing with serialize/deserialize
- [x] `cynepic-graph`: Event hooks (GraphHook trait + EventCollector)
- [x] `cynepic-bayes`: Multi-dimensional MH sampler
- [x] `cynepic-bayes`: Adaptive MH (Robbins-Monro acceptance targeting)
- [x] `cynepic-bayes`: Dirichlet-Multinomial conjugate prior
- [x] `cynepic-bayes`: BeliefTracker streaming updates
- [x] `cynepic-bayes`: ToolBelief / ToolBeliefSet reliability tracking
- [x] `cynepic-guardian`: LoopDetector (overvisit + alternation detection)
- [x] `cynepic-guardian`: RiskAwareEvaluator (Bayesian risk scoring)
- [x] `cynepic-guardian`: RateLimiter (token-bucket)
- [x] `cynepic-guardian`: EscalationManager (HITL lifecycle)
- [x] `cynepic-router`: BudgetTracker (cost-aware routing)
- [x] `cynepic-router`: ClassifierMetrics (confusion matrix, precision/recall/F1)
- [x] Test coverage: 99 tests across workspace

### 1.3 API Surface Polish
- [x] `Result`-returning API across `cynepic-causal`; no panics on caller input
- [x] `[workspace.lints]` enforcing the stated conventions, with remaining debt
      recorded as measured counts rather than omitted
- [x] Provenance on every estimate: `Estimand`, `StdErrorKind`, `Diagnostics`,
      with `ATEResult` unconstructible without them
- [ ] `missing_docs = "deny"` — 86 violations, tracked in `[workspace.lints]`
- [ ] Clear the panic-freedom debt set (`indexing_slicing` 104, `unwrap_used` 7,
      `expect_used` 4)
- [ ] `#[must_use]` on all `Result`-returning functions

---

## Phase 2: Integration Interfaces (v0.3.0) — IN PROGRESS

**Goal:** Each crate is usable from Python, HTTP, and MCP.

All three surfaces exist and compile. They were merged in a non-compiling state
— 32 errors, written against an API that did not exist — and were fixed once CI
could see them.

### 2.1 PyO3 Bindings (`bindings/pyo3/`)
- [x] `cynepic` module wrapping the core types, DAG, priors, circuit breaker
- [x] `extension-module` as an opt-in feature, so `cargo build --workspace`
      links on macOS; maturin turns it on
- [ ] Full surface: estimators returning `Result` with the estimand attached
- [ ] `maturin` release build, publish to PyPI as `cynepic`
- [ ] NumPy interop (ndarray <-> numpy, ideally zero-copy)

### 2.2 HTTP API (`crates/cynepic-server/`)
- [x] Axum server: classify, estimate, bayes update, policy evaluate
- [x] Estimation errors map to 422 rather than 500 — they describe the caller's
      data, not a service fault
- [x] Responses carry the estimand, standard-error kind, interval and diagnostics
- [ ] OpenAPI spec via `utoipa`
- [ ] Docker image
- [ ] Integration tests against a running server

### 2.3 MCP Tool Server (`bindings/mcp/`)
- [x] JSON-RPC 2.0 over stdio, with the version field actually validated
- [x] Tools: `classify_domain`, `estimate_ate`, `check_policy`, `update_belief`,
      `detect_loop`, `run_counterfactual`
- [ ] `monitor_drift` — currently returns `not_implemented` with what it would
      need. A monitor that always answers "no drift" is worse than an absent one
- [ ] Verified against a real MCP client

### 2.4 WASM Target
- [ ] `cynepic-core`, `cynepic-bayes`, `cynepic-causal` compilable to `wasm32-unknown-unknown`
- [ ] `wasm-pack` build + npm package

---

## Phase 3: Advanced Capabilities (v0.4.0)

### 3.1 cynepic-causal
- [ ] Polars DataFrame integration for data ingestion
- [ ] Sensitivity analysis (Rosenbaum bounds)
- [ ] Propensity score matching (not just IPW)

### 3.2 cynepic-bayes
- [ ] Hamiltonian Monte Carlo (HMC) via `burn` autodiff
- [ ] No U-Turn Sampler (NUTS)
- [ ] Gaussian Process prior

### 3.3 cynepic-router
- [ ] Embedding-based classifier via `candle` (sentence transformers)
- [ ] HNSW nearest-neighbor index
- [ ] Confidence calibration (Platt scaling)
- [ ] A/B routing for model comparison

### 3.4 cynepic-graph
- [ ] Parallel branch execution (fan-out/fan-in)
- [ ] LLM nodes (`async-openai` integration)
- [ ] Memory nodes (Neo4j via `neo4rs`)

### 3.5 cynepic-guardian
- [ ] Cedar policy engine support
- [ ] Persistent audit trail (PostgreSQL via `sqlx` or `sled` embedded)
- [ ] Audit trail export as OpenTelemetry spans

---

## Phase 4: Ecosystem & Publishing (v1.0.0)

### 4.1 Crates.io Publication
- [ ] Stabilize public API
- [ ] Publish all 6 crates + `cynepic` umbrella crate
- [ ] Semantic versioning enforcement

### 4.2 Examples
- [x] `examples/coverage_report`, `calibration_report`, `classifier_report` —
      the three measurement artifacts, published by CI on every run
- [ ] End-to-end demos per vertical (supply chain, clinical, financial risk;
      specced in [research.md](../research.md))

---

## <a id="benchmarking-what-still-has-to-be-proven"></a>Benchmarking — what still has to be proven

The README carries a table of **assumed** speedups over NetworkX, PyMC, OPA and
LangGraph. Nothing measures them. This section is what would have to exist
before any of them may be quoted.

### The ordering problem

Speed is the least important thing on this list, and it is the only thing the
table currently claims.

For causal inference the product is **a number someone will act on**. A
treatment effect that arrives in 10µs and is wrong is worth less than one that
takes 10ms and is right — and worth much less than one that takes 10ms and
*declines to answer* because the data cannot support it. The same holds for a
policy verdict and for a routing decision.

So the benchmark suite is ordered by what a wrong answer costs, not by what is
easy to measure:

| # | Claim | Instrument | Status |
|---|---|---|---|
| 1 | **Agreement** — same answer as a reference implementation on the same data | Golden-file parity against numpy/scipy/networkx | **Built**, 258 cases |
| 2 | **Coverage** — intervals deliver the confidence they claim | `cynepic-testkit` harness | **Built**, published per run |
| 3 | **Refusal** — declines when the data cannot support an answer | Adversarial input suite, refusal rate | **Partly built** |
| 4 | **Reproducibility** — same seed, same answer, every platform | Cross-platform determinism tests | **Partly built** |
| 5 | **Tail latency** — p99, not mean | criterion distributions | **Not built** |
| 6 | **Footprint** — allocations and peak memory | counting allocator | **Not built** |
| 7 | **Throughput** — the table's claim | criterion vs a Python harness | **Not built** |

### 1. Agreement (the claim that matters most)

A cross-language speedup claim is only interesting if both sides compute the
same thing. Before any "1000x faster than NetworkX" is meaningful, we need
"identical to NetworkX", and today nothing checks that.

- [x] Golden-file fixtures generated by `scripts/generate_parity_fixtures.py`
      and committed, with the library versions that produced them recorded
- [x] `crates/cynepic-causal/tests/parity.rs` — 258 cases, all passing

| Quantity | Here | Reference | Cases | Worst disagreement |
|---|---|---|---|---|
| Least squares ATE | pivoted Householder QR | numpy SVD (`lstsq`) | 5 | < 1e-9 |
| Classical SE | closed form | numpy `sigma^2 (X'X)^-1` | 4 | < 1e-8 |
| Welch SE | closed form | recovered from `scipy.stats.ttest_ind` | 3 | < 1e-9 |
| Beta / Gamma / t quantiles | Lentz CF + bisection | scipy (Boost) | 116 | **5.5e-12** |
| D-separation | Bayes-Ball | `networkx.is_d_separator` | 134 | none |

The pairs use genuinely different algorithms, which is what makes agreement
evidence rather than tautology. The 134 d-separation verdicts include the
collider cases — conditioning on a collider *opens* a path, the direction an
implementation is most likely to get backwards and the one a unit test written
by the same author is least likely to catch.

No Python at test time: references are computed once and committed, so CI stays
dependency-free.

- [ ] Extend to IPW and 2SLS once a reference implementation is pinned
- [ ] Disagreement is a **finding**, not a benchmark result — it goes in
      FINDINGS.md with the measurement

### 3. Refusal

The property no Python equivalent has, and the one most worth advertising.

- [ ] A corpus of inputs that *should* be refused: rank-deficient designs,
      empty arms, weak instruments, no overlap, separation, unknown variables
- [ ] Report the refusal rate and, crucially, the **false-answer rate** — inputs
      that should have been refused and were answered instead
- [ ] Compare against DoWhy/statsmodels on the same inputs. Where they return a
      number and we return an error, that is the product

### 5. Tail latency, not throughput

For an embedded decision layer the p99 is what a caller feels; the mean is what
a marketing table quotes.

- [ ] Report p50/p95/p99 from criterion's distributions rather than the mean
- [ ] Latency under adversarial input, not just the happy path
- [ ] Record the CPU. A number without the machine is not reproducible

### 6. Footprint

The claim that "no GC pauses, embeddable in any service" rests on, and is
currently unmeasured.

- [ ] Allocation counts per estimate via a counting allocator
- [ ] Peak RSS on the largest grid cell
- [ ] Confirm no allocation in the circuit-breaker and rate-limiter hot paths

### 7. Throughput (the README table)

Last, deliberately.

- [ ] A committed Python harness pinning the versions compared against
- [ ] Same data, same task, both sides — trivially unfair comparisons are the
      norm in this genre and are worth nothing
- [ ] Run on a quiet machine with the CPU recorded, never on a shared CI runner
- [ ] Publish the cells where we are *slower*. A table with no losses is
      advertising

Only when 1, 3 and 5 exist does 7 become a claim rather than a hope. Until
then the README says "assumed", which is accurate.

### 4.3 Documentation Site
- [ ] mdBook or similar for user-facing docs
- [ ] Tutorials per crate

---

## Priority Matrix

| Item | Impact | Effort | Priority |
|------|--------|--------|----------|
| CI/CD pipeline | High | Low | **Done** |
| OLS covariate adjustment | High | Medium | **Done** |
| d-separation, front-door | High | Medium | **Done** |
| IPW + IV estimation | High | Medium | **Done** |
| MCMC variants (adaptive, multi-dim) | High | Medium | **Done** |
| Guardian safety (loops, HITL, rate limit) | High | Medium | **Done** |
| Router metrics + budget | Medium | Low | **Done** |
| Graph (timeout, cycles, hooks, checkpoint) | High | Medium | **Done** |
| PyO3 bindings (causal + bayes) | Very High | Medium | P1 |
| HTTP API server | High | Medium | P1 |
| MCP tool server | High | Medium | P1 |
| Embedding classifier | High | High | P2 |
| HMC/NUTS sampler | High | High | P2 |
| Cedar policies | Medium | Low | P2 |
| WASM target | Medium | Medium | P3 |
| Crates.io publish | High | Low | P3 (after API stable) |
