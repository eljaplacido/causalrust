# RALP Loop State — cynepic-rs

## Current Status: ITERATION 2 COMPLETE (PyO3 + MCP + Server)

## Iteration 2 — 2026-05-24

### Completed Fixes

#### Security Hardening
- [x] Converted `assert!` in `BetaBinomial::new`, `NormalNormal::new`, `DirichletMultinomial::new`, `GammaPoisson::new` to return `PriorResult<T>` with `PriorError`
- [x] `DirichletMultinomial::uniform` returns `PriorResult<Self>` with validation
- [x] `DirichletMultinomial::marginal_mean` / `marginal_variance` return `PriorResult<f64>` instead of panicking
- [x] `DirichletMultinomial::update` returns `PriorResult<()>` with length validation
- [x] `ToolBelief::with_prior` returns `PriorResult<Self>`
- [x] `LoopDetector` history upgraded from `Vec<String>` to `VecDeque<String>` (O(1) pop_front)
- [x] `AuditTrail` now has `recent_entries(limit)` and `with_entries(f)` to avoid full clone

#### New Crates — Full Vertical Stack
- [x] **`cynepic-pyo3`** (`bindings/pyo3/`) — PyO3 bindings: `CynefinDomain`, `CausalDag`, `BetaBinomial`, `CircuitBreaker`, `ToolBeliefSet`
- [x] **`cynepic-mcp`** (`bindings/mcp/`) — MCP JSON-RPC stdio server: 8 cognitive tools (`classify_domain`, `estimate_ate`, `check_policy`, `update_belief`, `detect_loop`, `audit_trail`, `run_counterfactual`, `monitor_drift`)
- [x] **`cynepic-server`** (`crates/cynepic-server/`) — Axum HTTP API: `GET /health`, `POST /router/classify`, `POST /causal/estimate`, `POST /bayesian/update`, `POST /guardian/evaluate`, `POST /graph/execute`

## Iteration 1 — 2026-03-15

### Completed Fixes

#### Code Examples (both READMEs)
- [x] Removed `.unwrap()` from `dag.add_edge()` calls (returns `()`)
- [x] Fixed `LoopViolation::NodeOvervisit` -> `NodeOvervisited`
- [x] Fixed `CircuitBreaker::record_failure()` — added `.await`, removed `mut`
- [x] Fixed `d_separated` — uses `HashSet<String>` not slice
- [x] Fixed `is_closed()` -> `is_open()` (method doesn't exist)
- [x] Re-exported `FnNode` from `cynepic_graph`

#### Test Counts (all docs)
- [x] Updated guardian: 19 -> 22 (3 rego-gated tests were undercounted)
- [x] Updated total: 96 -> 99 everywhere (README, CLAUDE.md, roadmap.md, CRATE_GUIDE.md)

#### Phantom Dependencies
- [x] Removed `axum` from cynepic-router/Cargo.toml (never imported)
- [x] Commented out `polars`, `async-openai`, `axum` from workspace Cargo.toml (planned, not used)

#### Doc Consistency
- [x] Fixed PITCH.md: updated from stale v0.1 (2200 LOC, 33 tests) to v0.2 (6800 LOC, 99 tests)
- [x] Fixed integration.md: "cynepic-causal uses Polars internally" -> clarified as ndarray + planned Polars
- [x] Fixed architecture.md: added 9 missing source files to file trees (all 6 crate trees updated)
- [x] Fixed architecture.md: dependency table now marks axum/async-openai/polars as "(planned, v0.3/v0.4)"
- [x] Fixed CLAUDE.md: added 3 missing doc files to repository layout + EXPERIMENTS.md
- [x] Updated CLAUDE.md test count from 98 -> 99

#### Re-exports
- [x] Added `FnNode` to cynepic-graph re-exports
- [x] Added `KeywordClassifier`, `ClassifierError`, `CostTier`, `RouteTarget`, `RouterConfig` to cynepic-router re-exports

#### Security Fixes
- [x] CausalDag: custom `Deserialize` impl rebuilds graph from variables+edges (was silently broken)
- [x] CircuitBreaker: upgraded all atomics from `Ordering::Relaxed` to `Ordering::SeqCst`
- [x] NormalNormal::update: early return on empty observations (prevents NaN corruption)
- [x] MetropolisHastings::new: validates proposal_std > 0, finite; n_samples > 0
- [x] MultiDimMH::new: validates all proposal_stds > 0, finite; n_samples > 0
- [x] DirichletMultinomial: added bounds checks to marginal_mean/marginal_variance
- [x] EventCollector: replaced `.lock().unwrap()` with `.unwrap_or_else(|e| e.into_inner())`

#### .gitignore
- [x] Added .env, *.pem, *.key, credentials.*, config.local.*, .claude/

#### EXPERIMENTS.md
- [x] Complete rewrite — fixed all ~18 API errors across 10 experiments

### Verified
- [x] 50 tests pass (core:8, guardian:19 no-rego, router:13, graph:10)
- [x] causal + bayes can't build on this Windows machine (msvc_spectre_libs) — documented in CLAUDE.md

### Remaining (lower priority, not blocking launch)
- [ ] ~15 assert! in lib code (priors, samplers, DirichletMultinomial) — document as known, convert to Result in v0.3
- [ ] LoopDetector Vec::remove(0) is O(n) — could use VecDeque
- [ ] AuditTrail.entries() clones entire Vec under lock
- [ ] WORKFLOWS.md may have stale API references (not audited in detail)
- [ ] inner README.md loop detection example not verified (may need NodeOvervisited fix)
