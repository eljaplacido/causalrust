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

### 4.2 Examples & Benchmarks
- [ ] `examples/` directory with end-to-end demos
- [ ] `cargo bench` via `criterion`
- [ ] Compare vs DoWhy, PyMC, OPA

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
