# cynepic-rs

> Modular Rust libraries for complexity-adaptive decision intelligence.

A Cargo workspace of 6 independently publishable crates implementing the [CARF/CYNEPIC](https://github.com/eljaplacido/projectcarfcynepic/) architecture — causal inference, Bayesian reasoning, policy guardrails, semantic routing, and workflow orchestration for agentic AI, MLOps, and data engineering.

## Why cynepic-rs?

Most agent frameworks focus on LLM orchestration. cynepic-rs provides the **missing infrastructure layer** underneath:

- **Causal correctness** — don't just correlate, identify *why* things happen
- **Calibrated uncertainty** — Bayesian beliefs instead of ad-hoc confidence scores
- **Formal governance** — Rego policies with append-only audit trails
- **Type-safe orchestration** — `StateGraph<S>` with compile-time guarantees
- **Rust-native** — no GC pauses, no Python runtime, embeddable in any service

## Crates

| Crate | Description | Tests |
|-------|-------------|-------|
| **[cynepic-core](causalrust/crates/cynepic-core)** | `CynefinDomain`, `AnalyticalEngine` trait, `PolicyDecision`, `AuditEntry`, `EpistemicState` | 13 |
| **[cynepic-guardian](causalrust/crates/cynepic-guardian)** | Policy chains, circuit breaker, loop detection, rate limiting, HITL escalation, bias auditing, audit trail | 30 |
| **[cynepic-causal](causalrust/crates/cynepic-causal)** | Causal DAG, d-separation, backdoor/front-door criteria, OLS/IPW/IV estimation, refutation, counterfactual reasoning | 30 |
| **[cynepic-router](causalrust/crates/cynepic-router)** | Cynefin classifier, entropy scoring, cost-aware routing, budget tracking, drift detection, classifier metrics | 17 |
| **[cynepic-bayes](causalrust/crates/cynepic-bayes)** | Beta/Normal/Gamma/Dirichlet priors, MH/Adaptive/Multi-dim MCMC, belief tracking, tool reliability | 20 |
| **[cynepic-graph](causalrust/crates/cynepic-graph)** | Typed `StateGraph<S>`, conditional edges, cycle detection, per-node timeout, checkpointing, event hooks | 10 |

**Total: 120 unit tests + 2 doctests, ~8,100 LOC across 6 crates.**

> **Maturity.** These crates are pre-1.0 and not yet published to crates.io. The
> causal estimators in particular are being hardened — see
> [docs/roadmap.md](causalrust/docs/roadmap.md) for the known correctness gaps
> and the tier they are fixed in. No performance claims are made here until the
> benchmark suite lands; there is currently no `benches/` directory.

## Quick Start

```bash
cd causalrust
cargo build --workspace
cargo test --workspace --all-features   # 120 unit tests + 2 doctests
cargo doc --workspace --no-deps         # Generate API docs
```

**Requirements:** Rust 1.85+ (edition 2024)

Before opening a PR, run what CI runs:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo deny check all                    # requires: cargo install cargo-deny
```

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

[Apache License 2.0](LICENSE) — free for any use, including commercial and production, with a patent grant.

Relicensed from BSL 1.1 on 2026-08-17. The BSL already named Apache-2.0 as its Change License for 2030-03-13; that date was brought forward.

See [NOTICE](NOTICE) for trademark attribution — "CARF" and "CYNEPIC" remain trademarks of Cisuregen, and Apache-2.0 §6 grants no trademark rights.

---

## Contributing

- **[CONTRIBUTING.md](CONTRIBUTING.md)** — the workflow, the full local gate,
  and the findings ratchet.
- **[docs/FINDINGS.md](causalrust/docs/FINDINGS.md)** — every known correctness
  defect, with the measurement behind it. Start here.
- **[SECURITY.md](SECURITY.md)** — report vulnerabilities privately.
- **[CHANGELOG.md](CHANGELOG.md)** — what changed and what it measured.
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)**

The one thing worth knowing before you start: **a claim without a measurement is
not a claim.** Statistical changes need a coverage or calibration number, not a
unit test with a generous tolerance. That standard exists because thirty passing
unit tests once coexisted with an estimator at 0.0% interval coverage.

## Licence

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) — Apache-2.0 grants no trademark
rights, and the CARF/CYNEPIC marks are reserved.
