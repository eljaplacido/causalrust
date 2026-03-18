# cynepic-rs

> Modular Rust libraries for complexity-adaptive decision intelligence.

A Cargo workspace of 6 independently publishable crates implementing the [CARF/CYNEPIC](https://github.com/eljaplacido/projectcarfcynepic/) architecture — causal inference, Bayesian reasoning, policy guardrails, semantic routing, and workflow orchestration for agentic AI, MLOps, and data engineering.

## Why cynepic-rs?

Most agent frameworks focus on LLM orchestration. cynepic-rs provides the **missing infrastructure layer** underneath:

- **Causal correctness** — don't just correlate, identify *why* things happen
- **Calibrated uncertainty** — Bayesian beliefs instead of ad-hoc confidence scores
- **Formal governance** — Rego/Cedar policies with append-only audit trails
- **Type-safe orchestration** — `StateGraph<S>` with compile-time guarantees
- **Microsecond latency** — Rust-native, embeddable in any service

## Crates

| Crate | Description | Tests |
|-------|-------------|-------|
| **[cynepic-core](crates/cynepic-core)** | `CynefinDomain`, `AnalyticalEngine` trait, `PolicyDecision`, `AuditEntry`, `EpistemicState` | 13 |
| **[cynepic-guardian](crates/cynepic-guardian)** | Policy chains, circuit breaker, loop detection, rate limiting, HITL escalation, bias auditing, audit trail | 25 |
| **[cynepic-causal](crates/cynepic-causal)** | Causal DAG, d-separation, backdoor/front-door criteria, OLS/IPW/IV estimation, refutation, counterfactual reasoning | 30 |
| **[cynepic-router](crates/cynepic-router)** | Cynefin classifier, entropy scoring, cost-aware routing, budget tracking, drift detection, classifier metrics | 17 |
| **[cynepic-bayes](crates/cynepic-bayes)** | Beta/Normal/Gamma/Dirichlet priors, MH/Adaptive/Multi-dim MCMC, belief tracking, tool reliability | 20 |
| **[cynepic-graph](crates/cynepic-graph)** | Typed `StateGraph<S>`, conditional edges, cycle detection, per-node timeout, checkpointing, event hooks | 10 |

**Total: 115 tests, ~7,800 LOC across 6 crates.**

## Quick Start

```bash
cd causalrust
cargo build --workspace
cargo test --workspace             # Run all 115 tests
cargo test -p cynepic-causal       # Single crate
cargo doc --workspace --no-deps    # Generate API docs
```

**Requirements:** Rust 1.85+ (edition 2024)

## Architecture

```
Query → cynepic-router (classify Cynefin domain)
         ├── Clear       → Deterministic lookup
         ├── Complicated → cynepic-causal (DAG → identify → estimate → refute)
         ├── Complex     → cynepic-bayes (prior → update → posterior)
         ├── Chaotic     → cynepic-guardian (circuit breaker → emergency response)
         └── Disorder    → Human escalation
All paths → cynepic-guardian (policy: approve / reject / escalate)
Orchestrated by → cynepic-graph (StateGraph<S> with conditional routing)
```

### Dependency Graph

```
cynepic-core (no internal deps)
  ├── cynepic-guardian  (core)
  ├── cynepic-causal    (core)
  ├── cynepic-router    (core)
  ├── cynepic-bayes     (core)
  └── cynepic-graph     (core)
```

No circular dependencies. Each crate re-exports `cynepic-core`.

## Examples

### Causal Inference — Identify and Estimate Treatment Effects

```rust
use cynepic_causal::{CausalDag, BackdoorCriterion, LinearATEEstimator, d_separated};
use ndarray::array;

// Build a causal DAG: confounder → treatment, confounder → outcome, treatment → outcome
let mut dag = CausalDag::new();
dag.add_variable("treatment");
dag.add_variable("outcome");
dag.add_variable("confounder");
dag.add_edge("confounder", "treatment");
dag.add_edge("confounder", "outcome");
dag.add_edge("treatment", "outcome");

// Check d-separation (treatment ⊥ outcome | confounder? No — direct edge)
let conditioning: std::collections::HashSet<String> = ["confounder".into()].into();
assert!(!d_separated(&dag, "treatment", "outcome", &conditioning));

// Identify adjustment set via backdoor criterion
let adjustment = BackdoorCriterion::find(&dag, "treatment", "outcome");
assert!(adjustment.contains(&"confounder".to_string()));

// Estimate average treatment effect
let treatment = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
let outcome   = array![10.0, 12.0, 11.0, 5.0, 6.0, 7.0];
let result = LinearATEEstimator::difference_in_means(&treatment, &outcome);
assert!((result.ate - 5.0).abs() < 0.01);
```

### Bayesian Belief Tracking — Tool Reliability

```rust
use cynepic_bayes::tool_belief::{ToolBelief, ToolBeliefSet};

// Track reliability of external tools/APIs
let mut tools = ToolBeliefSet::new();
tools.add_tool("llm_api", None);    // default Beta(1,1) prior
tools.add_tool("search_api", None);

// Record outcomes
tools.record_success("llm_api");
tools.record_success("llm_api");
tools.record_failure("llm_api");    // 2 successes, 1 failure

tools.record_success("search_api");
tools.record_success("search_api");
tools.record_success("search_api"); // 3 successes, 0 failures

// Query reliability beliefs
let llm = tools.get("llm_api").unwrap();
println!("LLM reliability: {:.2}", llm.reliability());     // ~0.67
println!("Should circuit-break? {}", llm.should_circuit_break(0.3)); // false

let search = tools.get("search_api").unwrap();
println!("Search reliability: {:.2}", search.reliability()); // ~0.80
```

### Policy Guardrails — Circuit Breaker + Rate Limiting

```rust
use cynepic_guardian::{CircuitBreaker, RateLimiter, RateLimitDecision};
use std::time::Duration;

// Circuit breaker: trips after 3 failures, resets after 5s
let cb = CircuitBreaker::new(3, Duration::from_secs(5));
cb.record_failure().await;
cb.record_failure().await;
cb.record_failure().await;
assert!(cb.is_open()); // tripped — block calls

// Rate limiter: 5 requests/sec burst, 2 tokens/sec refill
let mut limiter = RateLimiter::new(5, 2.0);
match limiter.check("user_123") {
    RateLimitDecision::Allowed { remaining_tokens } => {
        println!("Allowed, {} tokens left", remaining_tokens);
    }
    RateLimitDecision::Denied { retry_after_ms } => {
        println!("Denied, retry in {}ms", retry_after_ms);
    }
}
```

### Workflow Orchestration — Typed State Graph

```rust
use cynepic_graph::{StateGraph, FnNode, NodeId};
use std::sync::Arc;

// Build a graph: validate → classify → process
let graph = StateGraph::new()
    .add_node(Arc::new(FnNode::new("validate", |x: i32| async move {
        Ok(x.abs())
    })))
    .add_node(Arc::new(FnNode::new("double", |x: i32| async move {
        Ok(x * 2)
    })))
    .add_node(Arc::new(FnNode::new("negate", |x: i32| async move {
        Ok(-x)
    })))
    .set_entry(NodeId::new("validate"))
    .add_conditional_edge(NodeId::new("validate"), |x: &i32| {
        if *x > 10 { NodeId::new("negate") } else { NodeId::new("double") }
    });

let result = graph.execute(5, 10).await.unwrap();
assert_eq!(result, 10); // |5| = 5, 5 <= 10, so double: 10

let result = graph.execute(-20, 10).await.unwrap();
assert_eq!(result, -20); // |-20| = 20, 20 > 10, so negate: -20
```

## Integration Targets

| Interface | Status | Description |
|-----------|--------|-------------|
| Rust library | **Now** | `cargo add cynepic-*` |
| Python (PyO3) | Planned | `pip install cynepic` — accelerate DoWhy, PyMC workflows |
| HTTP API | Planned | Axum server with OpenAPI spec |
| MCP tools | Planned | JSON-RPC stdio for AI agent tooling |
| WASM | Planned | Browser/edge compute via wasm-pack |

## Documentation

| Document | Purpose |
|----------|---------|
| [EXPERIMENTS.md](EXPERIMENTS.md) | Hands-on experiments to try with each crate |
| [docs/architecture.md](docs/architecture.md) | Full architecture reference |
| [docs/WORKFLOWS.md](docs/WORKFLOWS.md) | Real-world workflow patterns |
| [docs/roadmap.md](docs/roadmap.md) | Development roadmap |
| [docs/integration.md](docs/integration.md) | Interop with Python/TS/Java/MCP |
| [docs/CRATE_GUIDE.md](docs/CRATE_GUIDE.md) | Per-crate developer guide |

## License

[Business Source License 1.1](../LICENSE)

- **Free** for personal, academic, research, educational, evaluation, and development use
- **Commercial/production** use requires a paid license — contact eljailari.suhonen@gmail.com
- **Converts to Apache-2.0** on 2030-03-13 (4-year change date)
