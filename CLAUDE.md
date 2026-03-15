# cynepic-rs — CLAUDE.md

> Agentic navigation guide for the cynepic-rs monorepo.
> This file is optimized for LLM context — keep it under 300 lines.

## Identity

**cynepic-rs** is a Cargo workspace of 6 independently publishable Rust crates implementing the [CARF/CYNEPIC](https://github.com/eljaplacido/projectcarfcynepic/) decision intelligence architecture. The goal is modular, library-first components usable as standalone Rust crates, PyO3 Python extensions, MCP tools, or HTTP API endpoints for agentic AI, MLOps, and data engineering workflows.

## Repository Layout

```
causalrust/                    # Git root
├── CLAUDE.md                  # This file (agentic nav)
├── causalrust/                # Cargo workspace root
│   ├── Cargo.toml             # Workspace manifest (resolver 2)
│   ├── Cargo.lock             # Pinned dependencies
│   ├── README.md              # Quick start
│   ├── crates/
│   │   ├── cynepic-core/      # Shared types, traits, errors
│   │   ├── cynepic-guardian/   # Policy guardrails, circuit breaker, audit
│   │   ├── cynepic-causal/    # Causal inference (DAG, ATE, refutation)
│   │   ├── cynepic-router/    # Cynefin semantic classifier + routing
│   │   ├── cynepic-bayes/     # Bayesian priors, MH sampler, belief state
│   │   └── cynepic-graph/     # StateGraph<S> workflow orchestration
│   ├── EXPERIMENTS.md         # Hands-on copy-paste experiments
│   └── docs/
│       ├── architecture.md    # Full architecture reference
│       ├── integration.md     # Interop guide (Python, TS, Java, MCP)
│       ├── roadmap.md         # Completion roadmap with phases
│       ├── CRATE_GUIDE.md     # Per-crate developer guide
│       ├── WORKFLOWS.md       # Real-world workflow patterns
│       └── PITCH.md           # Positioning and value proposition
```

## Build & Test

```bash
cd causalrust/causalrust
cargo build --workspace            # Build all crates
cargo test --workspace             # Run all 99 tests
cargo test -p cynepic-core         # Test single crate
cargo test -p cynepic-guardian --no-default-features  # Guardian without rego
cargo test -p cynepic-guardian --features rego         # Guardian with rego
```

- **Rust edition**: 2024, MSRV 1.85
- **Windows note**: `nalgebra` (transitive via `statrs`) may fail with `msvc_spectre_libs` — use `rustup default stable-x86_64-pc-windows-msvc` and install Spectre-mitigated libs via VS Installer, or build on Linux/WSL.

## Crate Dependency Graph

```
cynepic-core (no internal deps)
  ├── cynepic-guardian  (core)
  ├── cynepic-causal    (core)
  ├── cynepic-router    (core)
  ├── cynepic-bayes     (core)
  └── cynepic-graph     (core)
```

No circular dependencies. Each crate re-exports `cynepic-core`.

## Key Types & Traits

| Type | Crate | Purpose |
|------|-------|---------|
| `CynefinDomain` | core | Enum: Clear, Complicated, Complex, Chaotic, Disorder |
| `AnalyticalEngine` | core | Async trait for domain-specific analysis |
| `PolicyDecision` | core | Approve / Reject { reason } / Escalate { target } |
| `AuditEntry` | core | Append-only audit record (UUID, timestamp, decision) |
| `CircuitBreaker` | guardian | State machine: Closed → Open → HalfOpen |
| `PolicyChain` | guardian | Sequential evaluator chain, short-circuits on reject |
| `LoopDetector` | guardian | Detects node overvisits and alternation thrashing |
| `RiskAwareEvaluator` | guardian | Bayesian risk score → approve/escalate/reject |
| `RateLimiter` | guardian | Token-bucket rate limiting per action/actor |
| `EscalationManager` | guardian | HITL escalation lifecycle (pending/approved/rejected/timed-out) |
| `CausalDag` | causal | petgraph-backed DAG with parent/child queries |
| `d_separated` | causal | Bayes-Ball d-separation test on DAG |
| `BackdoorCriterion` | causal | Finds valid adjustment sets for causal identification |
| `FrontDoorCriterion` | causal | Finds front-door adjustment sets via mediators |
| `LinearATEEstimator` | causal | Difference-in-means + full OLS with covariate adjustment |
| `PropensityScoreEstimator` | causal | IPW estimation via logistic regression |
| `IVEstimator` | causal | Two-stage least squares (2SLS) |
| `ATEResult` | causal | Average Treatment Effect + standard error |
| `BetaBinomial` | bayes | Conjugate prior for binary outcomes |
| `DirichletMultinomial` | bayes | Conjugate prior for categorical data |
| `MetropolisHastings` | bayes | 1D MCMC sampler for arbitrary log-densities |
| `MultiDimMH` | bayes | Multi-dimensional MH with diagonal Gaussian proposal |
| `AdaptiveMH` | bayes | Self-tuning MH (Robbins-Monro acceptance targeting) |
| `BeliefTracker` | bayes | Streaming real-time belief updates with typed observations |
| `ToolBelief` | bayes | Beta-prior reliability tracking for tools/services |
| `ToolBeliefSet` | bayes | Multi-tool reliability monitoring |
| `CynefinRouter` | router | Classifier → domain → route target (budget-aware) |
| `BudgetTracker` | router | Cost tracking with tier-based budget enforcement |
| `ClassifierMetrics` | router | Confusion matrix, precision/recall/F1, misrouting cost |
| `StateGraph<S>` | graph | Typed async workflow graph with conditional edges |
| `Checkpoint<S>` | graph | Serializable execution snapshot for pause/resume |
| `GraphHook` | graph | Event hook trait for observability (NodeStarted/Completed/Failed) |
| `EventCollector` | graph | Collects graph events for testing/analysis |

## Conventions

- **Append-only audit**: `AuditTrail` entries are never mutated or deleted
- **Serde everywhere**: All public types derive `Serialize`/`Deserialize`
- **Async-first**: All engine/policy/node traits are async (tokio)
- **Feature-gated optionals**: `rego` feature on guardian, future `pyo3` features per crate
- **Error types**: Each crate has its own error enum via `thiserror`
- **No unwrap in lib code**: Tests may use `unwrap()`, library code returns `Result`

## Current Status (v0.2.0-dev, release candidate)

| Crate | Status | Tests | Key Capabilities |
|-------|--------|-------|-----------------|
| core | Complete | 8 | Domain enum, engine trait, policy types, audit types |
| guardian | Solid | 22 | Policy chains, Rego, circuit breaker, loop detection, rate limiting, HITL escalation, audit trail |
| causal | Solid | 26 | DAG, d-separation, backdoor/front-door, OLS, IPW, IV/2SLS, 4 refutation tests |
| router | Solid | 13 | Keyword classifier, cost-aware routing, budget tracking, classifier metrics (F1) |
| bayes | Solid | 20 | 4 conjugate priors, 3 MCMC samplers, belief tracker, tool reliability |
| graph | Solid | 10 | StateGraph, conditional edges, cycle detection, timeout, checkpoint/resume, event hooks |

**Total: 99 tests, ~6,800 LOC, 0 warnings.**

### Future Work (not blocking release)
- **Phase 2**: PyO3 bindings, HTTP API (Axum), MCP tool server, WASM target
- **Phase 3**: Polars backend, HMC/NUTS, embedding classifier, Cedar policies, parallel graph branches
- See [docs/roadmap.md](causalrust/docs/roadmap.md) for full details

## What NOT to Do

- Do not add `unsafe` blocks without explicit approval
- Do not introduce circular crate dependencies
- Do not add runtime panics in library code (`unwrap`, `expect` → return `Result`)
- Do not commit Cargo.lock changes without running `cargo test --workspace`
- Do not import Python/JS/TS code into the Rust crates — bindings go in a separate `bindings/` directory
