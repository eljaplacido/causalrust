# Architecture Reference — cynepic-rs

## Overview

cynepic-rs implements the CARF/CYNEPIC complexity-adaptive decision intelligence architecture as a Rust workspace of 6 crates. Each crate is independently publishable to crates.io and usable as a standalone library.

## System Flow

```
                    ┌─────────────────┐
   User Query ───►  │ cynepic-router  │  Classify complexity domain
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  causal   │  │  bayes   │  │ guardian  │  Domain-specific engines
        │ (Complic.)│  │ (Complex)│  │ (Chaotic) │
        └─────┬────┘  └─────┬────┘  └─────┬────┘
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │cynepic-guardian  │  Policy check (approve/reject/escalate)
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  cynepic-graph   │  Orchestrate multi-step workflows
                    └─────────────────┘
```

## Crate Architecture

### cynepic-core

Foundation crate. Zero external runtime dependencies beyond serde/chrono/uuid.

```
core/
├── domain.rs    → CynefinDomain enum (5 variants)
├── engine.rs    → AnalyticalEngine trait (async, generic I/O/Error)
├── error.rs     → CynepicError (policy rejection, low confidence, etc.)
└── policy.rs    → PolicyDecision, EscalationTarget, AuditEntry
```

**Design decisions:**
- `CynefinDomain` methods: `requires_human()` → Disorder, `is_emergency()` → Chaotic
- `AnalyticalEngine` uses associated types (not generics) for zero-cost dispatch
- `AuditEntry` is append-only by design (no `&mut` methods)

### cynepic-guardian

Policy enforcement and safety layer. Wraps `regorus` (Rego evaluator) behind a feature gate.

```
guardian/
├── policy.rs         → PolicyEvaluator trait, PolicyChain, RegoPolicyEvaluator
├── circuit_breaker.rs → CircuitBreaker (Closed/Open/HalfOpen state machine)
└── audit.rs          → AuditTrail (Arc<Mutex<Vec<AuditEntry>>>)
```

**Key patterns:**
- `PolicyChain` evaluates sequentially, short-circuits on first non-Approve
- `CircuitBreaker` is time-based: trips after N failures, resets after timeout
- Rego evaluator is `#[cfg(feature = "rego")]` — not compiled by default

### cynepic-causal

Causal inference engine built on `petgraph` for DAG operations and `ndarray` for numerics.

```
causal/
├── dag.rs          → CausalDag (petgraph wrapper with causal semantics)
├── identify.rs     → BackdoorCriterion (finds adjustment sets)
├── estimate/
│   ├── mod.rs      → Estimator trait
│   └── linear.rs   → LinearATEEstimator (diff-in-means, OLS stub)
└── refute.rs       → RefutationResult, placebo_treatment()
```

**Data flow:** DAG → identify confounders → estimate ATE → refute

### cynepic-router

Cynefin-based semantic classification and cost-aware routing.

```
router/
├── classifier.rs   → QueryClassifier trait, KeywordClassifier
├── router.rs       → CynefinRouter, RoutingDecision
└── config.rs       → RouterConfig, RouteTarget, CostTier
```

**Routing logic:** classify(query) → domain + confidence → if confident: route to target, else: fallback domain

### cynepic-bayes

Bayesian inference with conjugate priors and MCMC sampling.

```
bayes/
├── priors.rs   → BetaBinomial, NormalNormal, GammaPoisson
├── belief.rs   → BeliefState (unified enum over priors)
└── sampler.rs  → MetropolisHastings (Gaussian proposal, warmup + sampling)
```

**Conjugate updates are O(1)**. MH sampler is general-purpose for non-conjugate posteriors.

### cynepic-graph

Typed workflow orchestration inspired by LangGraph.

```
graph/
├── graph.rs   → StateGraph<S>, GraphError
└── node.rs    → Node trait, FnNode, NodeId
```

**Type safety:** `StateGraph<S>` is parameterized by state type — compiler enforces that all nodes accept and return `S`. Conditional edges route based on state inspection.

## Dependency Map (External)

| Dependency | Version | Used By | Purpose |
|-----------|---------|---------|---------|
| serde + serde_json | 1.x | all | Serialization (JSON-first) |
| tokio | 1.x | all | Async runtime |
| tracing | 0.1 | all | Structured logging |
| thiserror | 2.x | all | Ergonomic error types |
| chrono | 0.4 | core | Timestamps in audit entries |
| uuid | 1.x | core | Unique IDs for audit entries |
| regorus | 0.5 | guardian | Rego policy evaluation |
| petgraph | 0.7 | causal | Graph algorithms for DAGs |
| ndarray | 0.16 | causal | Array operations for estimators |
| statrs | 0.18 | bayes | Statistical distributions |
| rand + rand_distr | 0.9 | bayes | RNG for MCMC |
| axum | 0.8 | router | HTTP framework (planned) |
| async-openai | 0.27 | graph | LLM client (planned) |
| polars | 0.46 | causal | DataFrame ops (planned) |

## Thread Safety

All public types are `Send + Sync`. Shared state uses `Arc<Mutex<>>` (audit trail) or is immutable. `StateGraph<S>` requires `S: Send + Sync + 'static`.

## Error Strategy

Each crate defines its own error enum. Errors are non-panicking — all fallible operations return `Result<T, E>`. Cross-crate errors map via `CynepicError` in core.
