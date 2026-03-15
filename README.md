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
| **[cynepic-core](causalrust/crates/cynepic-core)** | `CynefinDomain`, `AnalyticalEngine` trait, `PolicyDecision`, `AuditEntry` | 8 |
| **[cynepic-guardian](causalrust/crates/cynepic-guardian)** | Policy chains, circuit breaker, loop detection, rate limiting, HITL escalation, audit trail | 22 |
| **[cynepic-causal](causalrust/crates/cynepic-causal)** | Causal DAG, d-separation, backdoor/front-door criteria, OLS/IPW/IV estimation, refutation | 26 |
| **[cynepic-router](causalrust/crates/cynepic-router)** | Cynefin classifier, cost-aware routing, budget tracking, classifier metrics (F1/precision/recall) | 13 |
| **[cynepic-bayes](causalrust/crates/cynepic-bayes)** | Beta/Normal/Gamma/Dirichlet priors, MH/Adaptive/Multi-dim MCMC, belief tracking, tool reliability | 20 |
| **[cynepic-graph](causalrust/crates/cynepic-graph)** | Typed `StateGraph<S>`, conditional edges, cycle detection, per-node timeout, checkpointing, event hooks | 10 |

**Total: 99 tests, ~6,800 LOC across 6 crates.**

## Quick Start

```bash
cd causalrust
cargo build --workspace
cargo test --workspace             # Run all 99 tests
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

## Examples

### Causal Inference — Identify and Estimate Treatment Effects

```rust
use cynepic_causal::{CausalDag, BackdoorCriterion, LinearATEEstimator};
use ndarray::array;

// Build a causal DAG
let mut dag = CausalDag::new();
dag.add_variable("treatment");
dag.add_variable("outcome");
dag.add_variable("confounder");
dag.add_edge("confounder", "treatment");
dag.add_edge("confounder", "outcome");
dag.add_edge("treatment", "outcome");

// Identify adjustment set via backdoor criterion
let adjustment = BackdoorCriterion::find(&dag, "treatment", "outcome");
// adjustment = {"confounder"}

// Estimate average treatment effect
let treatment = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
let outcome   = array![10.0, 12.0, 11.0, 5.0, 6.0, 7.0];
let result = LinearATEEstimator::difference_in_means(&treatment, &outcome);
// result.ate ≈ 5.0
```

### Bayesian Belief Tracking — Tool Reliability

```rust
use cynepic_bayes::tool_belief::ToolBeliefSet;

let mut tools = ToolBeliefSet::new();
tools.add_tool("llm_api", None);
tools.add_tool("search_api", None);

tools.record_success("llm_api");
tools.record_success("llm_api");
tools.record_failure("llm_api");

let llm = tools.get("llm_api").unwrap();
println!("LLM reliability: {:.2}", llm.reliability());           // ~0.67
println!("Circuit break? {}", llm.should_circuit_break(0.3));     // false
```

### Policy Guardrails — Circuit Breaker + Loop Detection

```rust
use cynepic_guardian::{CircuitBreaker, LoopDetector, LoopViolation};
use std::time::Duration;

// Circuit breaker: trips after 3 failures
let cb = CircuitBreaker::new(3, Duration::from_secs(5));
cb.record_failure().await;
cb.record_failure().await;
cb.record_failure().await;
assert!(cb.is_open()); // tripped — block calls

// Loop detector: catch runaway agent loops
let mut detector = LoopDetector::new(5, 3);
for _ in 0..5 {
    let _ = detector.record_visit("retry_api");
}
assert!(matches!(
    detector.record_visit("retry_api"),
    Some(LoopViolation::NodeOvervisited { .. })
));
```

### Workflow Orchestration — Typed State Graph

```rust
use cynepic_graph::{StateGraph, FnNode, NodeId};
use std::sync::Arc;

let graph = StateGraph::new()
    .add_node(Arc::new(FnNode::new("validate", |x: i32| async move { Ok(x.abs()) })))
    .add_node(Arc::new(FnNode::new("double", |x: i32| async move { Ok(x * 2) })))
    .set_entry(NodeId::new("validate"))
    .add_edge(NodeId::new("validate"), NodeId::new("double"));

let result = graph.execute(-5, 10).await.unwrap(); // 10
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
| [EXPERIMENTS.md](causalrust/EXPERIMENTS.md) | Hands-on experiments to try with each crate |
| [docs/architecture.md](causalrust/docs/architecture.md) | Full architecture reference |
| [docs/WORKFLOWS.md](causalrust/docs/WORKFLOWS.md) | Real-world workflow patterns |
| [docs/roadmap.md](causalrust/docs/roadmap.md) | Development roadmap |
| [docs/integration.md](causalrust/docs/integration.md) | Interop with Python/TS/Java/MCP |
| [docs/CRATE_GUIDE.md](causalrust/docs/CRATE_GUIDE.md) | Per-crate developer guide |

## License

[Business Source License 1.1](LICENSE)

- **Free** for personal, academic, research, educational, evaluation, and development use
- **Commercial/production** use requires a paid license — contact eljailari.suhonen@gmail.com
- **Converts to Apache-2.0** on 2030-03-13 (4-year change date)
