# Experiments & Quick Start

> Quick-start examples for each crate. For full API documentation, run `cargo doc --workspace --no-deps --open`.

## Setup

```bash
cargo new cynepic-lab
cd cynepic-lab
```

Add to `Cargo.toml`:

```toml
[dependencies]
cynepic-core = { path = "../causalrust/crates/cynepic-core" }
cynepic-guardian = { path = "../causalrust/crates/cynepic-guardian" }
cynepic-causal = { path = "../causalrust/crates/cynepic-causal" }
cynepic-router = { path = "../causalrust/crates/cynepic-router" }
cynepic-bayes = { path = "../causalrust/crates/cynepic-bayes" }
cynepic-graph = { path = "../causalrust/crates/cynepic-graph" }
tokio = { version = "1", features = ["full"] }
ndarray = "0.16"
serde_json = "1"
```

---

## Causal Inference — Build a DAG and Estimate Effects

```rust
use cynepic_causal::{CausalDag, BackdoorCriterion, LinearATEEstimator};
use ndarray::array;

let mut dag = CausalDag::new();
dag.add_edge("confounder", "treatment");
dag.add_edge("confounder", "outcome");
dag.add_edge("treatment", "outcome");

let adjustment = BackdoorCriterion::find(&dag, "treatment", "outcome");
// adjustment = {"confounder"}

let treatment = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
let outcome   = array![10.0, 12.0, 11.0, 5.0, 6.0, 7.0];
let result = LinearATEEstimator::difference_in_means(&treatment, &outcome);
// result.ate ≈ 5.0
```

## Bayesian Belief Updates

```rust
use cynepic_bayes::priors::BetaBinomial;

let mut prior = BetaBinomial::new(1.0, 1.0); // uniform
prior.update(45, 455); // 45 successes, 455 failures
println!("Mean: {:.4}, CI: {:?}", prior.mean(), prior.credible_interval_95());
```

## Policy Guardrails

```rust
use cynepic_guardian::{CircuitBreaker, LoopDetector};
use std::time::Duration;

let cb = CircuitBreaker::new(3, Duration::from_secs(5));
// Trips after 3 failures, resets after 5s timeout

let mut detector = LoopDetector::new(5, 3);
// Flags after 5 visits to same node or 3 alternation cycles
```

## Routing and Classification

```rust
use cynepic_router::classifier::KeywordClassifier;
use cynepic_router::QueryClassifier;

let classifier = KeywordClassifier::default_patterns();
let result = classifier.classify("Why did revenue drop?").await.unwrap();
// result.domain = Complicated, result.entropy = low (confident)
```

## Workflow Orchestration

```rust
use cynepic_graph::{StateGraph, FnNode, NodeId};
use std::sync::Arc;

let graph = StateGraph::new()
    .add_node(Arc::new(FnNode::new("step1", |x: i32| async move { Ok(x + 1) })))
    .add_node(Arc::new(FnNode::new("step2", |x: i32| async move { Ok(x * 2) })))
    .set_entry(NodeId::new("step1"))
    .add_edge(NodeId::new("step1"), NodeId::new("step2"));

let result = graph.execute(5, 10).await.unwrap(); // 12
```

---

For comprehensive API documentation: `cargo doc --workspace --no-deps --open`
