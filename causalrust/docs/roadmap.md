# Completion Roadmap — cynepic-rs

## Current State (v0.2.0-dev)

~6,800 LOC across 6 crates, 99 tests, all layers implemented with solid coverage. CI configured, no bindings, no persistence, no HTTP API.

---

## Phase 1: Hardening & CI (v0.2.0) — MOSTLY COMPLETE

### 1.1 Build & CI
- [x] GitHub Actions CI: build + test on ubuntu-latest
- [x] Lint: clippy + rustfmt checks
- [x] Feature matrix: test guardian with and without `rego`
- [x] MSRV check on 1.85
- [ ] `rust-toolchain.toml` pinning MSRV
- [ ] Resolve `msvc_spectre_libs` issue on Windows (nalgebra transitive dep)

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
- [ ] `#[must_use]` on all Result-returning functions
- [ ] `deny(missing_docs)` lint for all crates
- [ ] Per-crate CHANGELOG.md

---

## Phase 2: Integration Interfaces (v0.3.0)

**Goal:** Each crate is usable from Python, HTTP, and MCP.

### 2.1 PyO3 Bindings (`bindings/python/`)
- [ ] `cynepic-py` package wrapping all 6 crates
- [ ] Priority: `cynepic-causal` (accelerate DoWhy bottlenecks)
- [ ] Priority: `cynepic-bayes` (fast conjugate updates for Python ML pipelines)
- [ ] `maturin` build system, publish to PyPI as `cynepic`
- [ ] Numpy interop via `numpy` PyO3 crate (ndarray <-> numpy zero-copy)

### 2.2 HTTP API (`crates/cynepic-server/`)
- [ ] Axum-based HTTP server exposing all crates as REST endpoints
- [ ] OpenAPI spec generation via `utoipa`
- [ ] Docker image

### 2.3 MCP Tool Server (`crates/cynepic-mcp/`)
- [ ] MCP protocol implementation (JSON-RPC over stdio)
- [ ] Tools: `classify_query`, `estimate_ate`, `update_belief`, `evaluate_policy`, `run_workflow`
- [ ] Compatible with Claude Desktop, Cursor, VS Code MCP clients

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
