# Architecture Reference — cynepic-rs

## Overview

cynepic-rs implements the CARF/CYNEPIC complexity-adaptive decision intelligence architecture as a Rust workspace of 6 crates. Each crate is independently publishable and usable as a standalone library.

See [NOTICE](../../NOTICE) for IP classification and trademark attribution.

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

## Crate Overview

### cynepic-core

Foundation crate providing shared types and traits: `CynefinDomain` (5-variant complexity enum), `AnalyticalEngine` (async analysis trait), `PolicyDecision` (approve/reject/escalate), `AuditEntry` (append-only audit record), and `EpistemicState` (unified session provenance).

### cynepic-guardian

Policy enforcement and safety layer: policy evaluation chains, circuit breaker pattern, loop detection, rate limiting, risk-aware evaluation, bias auditing, HITL escalation lifecycle, and append-only audit trail. Rego evaluation available behind the `rego` feature gate.

### cynepic-causal

Causal inference engine providing the full Pearl-style pipeline: DAG representation, d-separation testing, backdoor/front-door identification, ATE estimation (OLS, IPW, IV/2SLS), refutation tests (placebo, random cause, subset, bootstrap), and counterfactual reasoning.

### cynepic-router

Cynefin-based semantic classification and cost-aware routing with entropy-based confidence scoring, budget tracking, distribution drift detection (KL-divergence), and classifier evaluation metrics.

### cynepic-bayes

Bayesian inference with four conjugate prior families, three MCMC samplers (standard, adaptive, multi-dimensional), streaming belief tracking, and tool/service reliability monitoring.

### cynepic-graph

Typed workflow orchestration: `StateGraph<S>` with compile-time type safety, conditional edges, cycle detection, per-node timeout, serializable checkpointing for pause/resume, and event hooks for observability.

## Design Principles

- **Append-only audit**: Audit trail entries are never mutated or deleted
- **Serde everywhere**: All public types derive `Serialize`/`Deserialize`
- **Async-first**: All engine/policy/node traits are async (tokio)
- **Feature-gated optionals**: `rego` feature on guardian, future `pyo3` features per crate
- **Error types**: Each crate has its own error enum via `thiserror`; no panics in library code
- **Thread safety**: All public types are `Send + Sync`

## Dependency Graph

```
cynepic-core (no internal deps)
  ├── cynepic-guardian  (core)
  ├── cynepic-causal    (core)
  ├── cynepic-router    (core)
  ├── cynepic-bayes     (core)
  └── cynepic-graph     (core)
```

No circular dependencies. All external dependencies use permissive licenses (MIT, Apache 2.0, BSD-3-Clause).
