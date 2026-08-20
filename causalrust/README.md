# cynepic-rs

> Modular Rust libraries and runtimes for complexity-adaptive decision intelligence.

A Cargo workspace of 9 crates (6 core + PyO3 bridge + MCP server + HTTP API) implementing the [CARF/CYNEPIC](https://github.com/eljaplacido/projectcarfcynepic/) architecture — causal inference, Bayesian reasoning, policy guardrails, semantic routing, and workflow orchestration for agentic AI, MLOps, and data engineering.

---

## Why cynepic-rs?

Most agent frameworks focus on LLM orchestration. cynepic-rs provides the **missing infrastructure layer** underneath:

- **Causal correctness** — don't just correlate, identify *why* things happen
- **Calibrated uncertainty** — Bayesian beliefs instead of ad-hoc confidence scores
- **Formal governance** — Rego-compatible policies with append-only audit trails
- **Type-safe orchestration** — `StateGraph<S>` with compile-time guarantees
- **Rust-native** — no GC pauses, no Python runtime, embeddable in any service

## Crates

| Crate | Description | Tests |
|-------|-------------|-------|
| **[cynepic-core](crates/cynepic-core)** | `CynefinDomain`, `AnalyticalEngine` trait, `PolicyDecision`, `AuditEntry`, `EpistemicState` | 13 |
| **[cynepic-guardian](crates/cynepic-guardian)** | Policy chains, circuit breaker, loop detection, rate limiting, HITL escalation, bias auditing, audit trail | 30 |
| **[cynepic-causal](crates/cynepic-causal)** | Causal DAG, d-separation, backdoor/front-door criteria, OLS/IPW/IV estimation, refutation, counterfactual reasoning | 30 |
| **[cynepic-router](crates/cynepic-router)** | Cynefin classifier, entropy scoring, cost-aware routing, budget tracking, drift detection, classifier metrics | 17 |
| **[cynepic-bayes](crates/cynepic-bayes)** | Beta/Normal/Gamma/Dirichlet priors, MH/Adaptive/Multi-dim MCMC, belief tracking, tool reliability | 20 |
| **[cynepic-graph](crates/cynepic-graph)** | Typed `StateGraph<S>`, conditional edges, cycle detection, per-node timeout, checkpointing, event hooks | 10 |

**Total: 120 unit tests + 2 doctests, ~8,100 LOC across 6 crates.**

> **Maturity.** Pre-1.0, not yet published to crates.io. The causal estimators
> are being hardened — see [docs/roadmap.md](docs/roadmap.md) for the known
> correctness gaps and the tier that closes each. No performance claims are made
> until the benchmark suite lands.

## Quick Start

```bash
cargo build --workspace
cargo test --workspace --all-features   # 120 unit tests + 2 doctests
cargo test -p cynepic-causal            # Single crate
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

No circular dependencies. Each crate re-exports `cynepic-core`.

---

## Use Cases — When, How, and Why to Apply

### Decision Matrix

| You want to... | Use this crate | Method | What you get |
|---------------|---------------|--------|-------------|
| **Build a causal model** of a system (e.g., "what causes latency spikes?") | `cynepic-causal` | DAG → d-separation → backdoor adjustment → ATE estimation | Causal effect with standard error and refutation tests |
| **Track tool/API reliability** and auto-circuit-break when they degrade | `cynepic-bayes` | Beta-Binomial conjugate prior → belief update per call | Reliability score + automatic "should I stop calling this?" decision |
| **Enforce deployment policies** (e.g., "no deploys to prod on Friday") | `cynepic-guardian` | PolicyChain → evaluate → Approve/Reject/Escalate | Deterministic policy enforcement with audit trail |
| **Classify problem complexity** before choosing an approach | `cynepic-router` | Keyword classifier + entropy scoring → Cynefin domain | Domain + confidence + "which engine should handle this?" |
| **Orchestrate multi-step AI workflows** with safety guarantees | `cynepic-graph` | StateGraph<S> → nodes → conditional edges → execute | Compile-time type safety, per-node timeout, checkpoint/resume |
| **Monitor for distribution drift** in your AI system over time | `cynepic-router` | DriftDetector → KL-divergence → baseline comparison | Alert when routing patterns shift significantly |
| **Call from Python** to accelerate DoWhy/PyMC/numpy workflows | `cynepic-pyo3` | `pip install cynepic` → `import cynepic` | Exact Beta/Gamma quantiles, verified estimators; speedup unmeasured |
| **Give AI agents decision intelligence** via MCP | `cynepic-mcp` | JSON-RPC stdio → 8 cognitive tools | Agents can classify, estimate ATE, check policy, update beliefs |
| **Run a lightweight decision API** without Python dependencies | `cynepic-server` | Axum binary → 6 REST endpoints | Single ~8MB binary, no venv, no pip |
| **Audit AI decisions** for compliance (EU AI Act, SOC2) | `cynepic-guardian` | AuditTrail → `recent_entries()` or `with_entries()` | Immutable append-only log, no-clone access for large trails |

### When NOT to use

- **Pure LLM orchestration** without analytical needs — use LangChain/LangGraph directly
- **Real-time <1µs latency** with zero allocations — cynepic-rs is fast but not hard-real-time
- **No Rust toolchain** and no need for Python/MCP — the standalone crates require `cargo`

---

## Performance — one row measured, the rest assumed

| Operation | Python equivalent | cynepic-rs | Speedup | Status |
|-----------|------------------|------------|---------|--------|
| DAG d-separation | NetworkX | petgraph | **9–19x** | **measured** |
| Beta conjugate prior update | PyMC | direct | ~1,000x | assumed |
| Policy evaluation | OPA sidecar | in-process regorus | ~100x | assumed |
| Circuit breaker check | Python | atomics | ~1,000x | assumed |
| StateGraph step | LangGraph | typed dispatch | ~10x | assumed |

The rows marked **assumed** are what we expect from moving this work out of
Python. No benchmark in this repository produces them, and they should not be
quoted.

The one row that *was* measured came back **50x below its assumption** — 9–19x,
not ~1,000x — and the ratio *falls* as the graph grows, so most of the win is
Python call overhead rather than the algorithm. The assumption was not
dishonest; it was plausible and had never met an instrument. Reproduce it with
`scripts/compare_networkx.py` and `cargo run -p cynepic-causal --example
latency_report --release` on one machine, and see
[docs/roadmap.md](docs/roadmap.md#benchmarking-what-still-has-to-be-proven) for
what each remaining row would take to prove.

**Speed is also not the main reason to use this.** For causal inference the
value is a number you can act on — an interval that covers, an estimator that
declines when the data cannot support an answer, a result that reproduces from
a seed. A fast wrong effect estimate is worth less than a slow right one.

What this repository *can* stand behind today, every figure reproducible from a
seeded command:

- `ols_adjusted` achieves nominal interval coverage on all nine DGP cells
- `ipw` is nominal on all eight estimable cells, and **refuses** the ninth
  rather than answering
- Every conjugate prior's credible interval is calibrated; both MCMC samplers
  pass simulation-based calibration
- Identical seeds give identical chains and identical refutation verdicts

See **[docs/FINDINGS.md](docs/FINDINGS.md)** for those, and
**[docs/roadmap.md#benchmarking](docs/roadmap.md#benchmarking-what-still-has-to-be-proven)**
for what has to be built before the table above means anything.

---

## Embedding — How to Integrate cynepic-rs

### 1. Rust Library (Direct Crate Dependency)

Add to your `Cargo.toml`:
```toml
[dependencies]
cynepic-core = "0.2"
cynepic-causal = "0.2"
cynepic-bayes = "0.2"
cynepic-guardian = { version = "0.2", default-features = false }
cynepic-router = "0.2"
cynepic-graph = "0.2"
```

Use in your code:
```rust
use cynepic_causal::{CausalDag, BackdoorCriterion, LinearATEEstimator};
use cynepic_bayes::BetaBinomial;

// Build a causal model
let mut dag = CausalDag::new();
dag.add_variable("deploy_size");
dag.add_variable("incident_rate");
dag.add_variable("team_experience");
dag.add_edge("team_experience", "deploy_size");
dag.add_edge("team_experience", "incident_rate");
dag.add_edge("deploy_size", "incident_rate");

// Find what to control for
let adjustment = BackdoorCriterion::find(&dag, "deploy_size", "incident_rate");

// Track tool reliability
let mut belief = BetaBinomial::uniform();
belief.update(95, 5); // 95 successes, 5 failures
println!("Tool reliability: {:.2}", belief.mean()); // 0.95
```

### 2. Python (PyO3 Bridge)

```bash
# Install from source (pip package coming soon via maturin)
cd causalrust/bindings/pyo3
maturin develop --release
```

```python
import cynepic

# Cynefin domain classification
domain = cynepic.CynefinDomain.COMPLICATED
print(domain)  # complicated

# Causal DAG analysis
dag = cynepic.CausalDag()
dag.add_variable("deploy_size")
dag.add_variable("incident_rate")
dag.add_variable("team_experience")
dag.add_edge("team_experience", "deploy_size")
dag.add_edge("deploy_size", "incident_rate")
adjustment = dag.find_backdoor_adjustment("deploy_size", "incident_rate")
print(adjustment)  # ['team_experience']

# Bayesian belief tracking
belief = cynepic.BetaBinomial()
belief.update(95, 5)
print(f"Reliability: {belief.mean:.2f}")  # 0.95

# Circuit breaker
breaker = cynepic.CircuitBreaker(failure_threshold=3, recovery_timeout_secs=10)
# breaker.record_failure()  # Call 3x to trip

# Tool reliability set
tools = cynepic.ToolBeliefSet()
tools.add_tool("llm_api")
tools.record_success("llm_api")
tools.record_failure("llm_api")
print(f"LLM reliability: {tools.reliability('llm_api'):.2f}")
```

### 3. MCP Server (AI Agent Integration)

```bash
# Run the MCP server (any MCP-compatible agent can connect)
cd causalrust/bindings/mcp
cargo run
```

Your AI agent config (e.g., Claude Desktop):
```json
{
  "mcpServers": {
    "cynepic": {
      "command": "cargo",
      "args": ["run", "--manifest-path", "causalrust/bindings/mcp/Cargo.toml"]
    }
  }
}
```

Agent can now call tools like:
- `classify_domain` — "Is this DevOps question clear, complicated, or complex?"
- `estimate_ate` — "What's the causal effect of this deploy on error rate?"
- `check_policy` — "Does rollback policy allow this action?"
- `update_belief` — "Update my model with these 50 new data points"
- `detect_loop` — "Am I retrying the same thing too many times?"
- `audit_trail` — "Show me the last 20 decisions made"
- `run_counterfactual` — "What if we'd deployed canary first instead?"
- `monitor_drift` — "Has my routing distribution changed since last week?"

### 4. HTTP API Server (Any Language)

```bash
# Start the server
cd causalrust/crates/cynepic-server
cargo run
# Listening on http://localhost:4310
```

```bash
# Health check
curl http://localhost:4310/health

# Classify a query
curl -X POST http://localhost:4310/router/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "Why did the database connection pool exhaust?"}'
# → {"domain": "Complicated", "confidence": 0.92, "entropy": 0.34}

# Estimate causal effect
curl -X POST http://localhost:4310/causal/estimate \
  -H "Content-Type: application/json" \
  -d '{"treatment": [1,1,1,0,0,0], "outcome": [10,12,11,5,6,7]}'
# → {"ate": 5.0, "std_error": 0.5, "method": "difference_in_means"}

# Update Bayesian belief
curl -X POST http://localhost:4310/bayesian/update \
  -H "Content-Type: application/json" \
  -d '{"successes": 95, "trials": 100}'
# → {"posterior_mean": 0.95, "credible_interval": [0.91, 0.99]}

# Check policy
curl -X POST http://localhost:4310/guardian/evaluate \
  -H "Content-Type: application/json" \
  -d '{"domain": "complicated", "amount": 5000}'
# → {"verdict": "Approved", "reason": null}
```

---

## Metrics & KPIs — What to Track

### Decision Quality

| Metric | Crate | What It Measures | Target |
|--------|-------|-----------------|--------|
| **Router F1** | `cynepic-router` | Classification accuracy across 5 Cynefin domains | ≥ 0.90 |
| **Causal ATE accuracy** | `cynepic-causal` | How close estimated effects are to ground truth | MSE ratio < 0.001 |
| **Bayesian calibration** | `cynepic-bayes` | Posterior coverage of true parameters | ≥ 90% well-calibrated |
| **Policy determinism** | `cynepic-guardian` | Same input → same verdict every time | 100% identical across runs |
| **Loop detection precision** | `cynepic-guardian` | False positives in loop detection | 0% false alarms |

### Performance

| Metric | Crate | What It Measures | Benchmark |
|--------|-------|-----------------|-----------|
| **DAG d-separation latency** | `cynepic-causal` | Time to check d-separation on 100-node DAG | < 100µs |
| **Conjugate prior update** | `cynepic-bayes` | Time to update Beta-Binomial with new evidence | < 1µs |
| **Policy chain evaluation** | `cynepic-guardian` | Time to evaluate a 10-rule policy chain | < 50µs |
| **StateGraph step** | `cynepic-graph` | Per-node execution overhead | < 100µs |
| **Circuit breaker** | `cynepic-guardian` | State transition latency | < 100ns |

### Reliability

| Metric | Crate | What It Measures | Target |
|--------|-------|-----------------|--------|
| **Test coverage** | All | Percentage of lines covered by tests | ≥ 80% |
| **No-panic guarantee** | All | Library constructors returning `Result` instead of panicking | 100% of pub fn |
| **Audit trail integrity** | `cynepic-guardian` | Entries never mutated after writing | Immutable by design |
| **Concurrency safety** | All | Thread-safety under concurrent access | No data races (SeqCst atomics) |

---

## Rust Library Examples

### Causal Inference — Identify and Estimate Treatment Effects

```rust
use cynepic_causal::{CausalDag, BackdoorCriterion, LinearATEEstimator, d_separated};
use ndarray::array;
use std::collections::HashSet;

// Build a causal DAG: confounder → treatment, confounder → outcome, treatment → outcome
let mut dag = CausalDag::new();
dag.add_variable("treatment");
dag.add_variable("outcome");
dag.add_variable("confounder");
dag.add_edge("confounder", "treatment");
dag.add_edge("confounder", "outcome");
dag.add_edge("treatment", "outcome");

// Check d-separation
let conditioning: HashSet<String> = ["confounder".into()].into();
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
tools.register("llm_api");
tools.register("search_api");

// Record outcomes
tools.record("llm_api", true);   // success
tools.record("llm_api", true);
tools.record("llm_api", false);  // failure (2s, 1f)

// Query reliability
let llm = tools.get("llm_api").unwrap();
println!("LLM reliability: {:.2}", llm.reliability());          // ~0.67
println!("Should circuit-break? {}", llm.should_circuit_break(0.3)); // false
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

let graph = StateGraph::new()
    .add_node(Arc::new(FnNode::new("validate", |x: i32| async move { Ok(x.abs()) })))
    .add_node(Arc::new(FnNode::new("double", |x: i32| async move { Ok(x * 2) })))
    .add_node(Arc::new(FnNode::new("negate", |x: i32| async move { Ok(-x) })))
    .set_entry(NodeId::new("validate"))
    .add_conditional_edge(NodeId::new("validate"), |x: &i32| {
        if *x > 10 { NodeId::new("negate") } else { NodeId::new("double") }
    });

let result = graph.execute(5, 10).await.unwrap();
assert_eq!(result, 10); // |5|=5 ≤10 → double: 10
```

---

## Integration Targets

| Interface | Status | Description |
|-----------|--------|-------------|
| Rust library | **Now** | `cargo add cynepic-core cynepic-causal cynepic-bayes cynepic-guardian cynepic-router cynepic-graph` |
| Python (PyO3) | **Now** | `maturin develop` → `import cynepic` — accelerate DoWhy, PyMC, NetworkX workflows |
| HTTP API | **Now** | `cargo run -p cynepic-server` → `localhost:4310` — 6 REST endpoints, single binary |
| MCP tools | **Now** | `cargo run -p cynepic-mcp` → JSON-RPC stdio — 8 cognitive tools for AI agents |
| WASM | Planned | Browser/edge compute via wasm-pack |

---

## Documentation

| Document | Purpose |
|----------|---------|
| [EXPERIMENTS.md](EXPERIMENTS.md) | Hands-on experiments to try with each crate |
| [docs/architecture.md](docs/architecture.md) | Full architecture reference |
| [docs/WORKFLOWS.md](docs/WORKFLOWS.md) | Real-world workflow patterns |
| [docs/roadmap.md](docs/roadmap.md) | Development roadmap |
| [docs/integration.md](docs/integration.md) | Interop with Python/TS/Java/MCP |
| [docs/CRATE_GUIDE.md](docs/CRATE_GUIDE.md) | Per-crate developer guide |

---

## License

[Apache License 2.0](../LICENSE) — free for any use, including commercial and production, with a patent grant.

Relicensed from BSL 1.1 on 2026-08-17. See [NOTICE](../NOTICE) for trademark attribution.
