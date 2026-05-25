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
- **Microsecond latency** — Rust-native, embeddable in any service
- **No panics in library code** — all constructors return `Result`, no crashes from bad input

---

## Crates

| Crate | Lines | Tests | What It Does |
|-------|-------|-------|-------------|
| **[cynepic-core](crates/cynepic-core)** | ~200 | 8 | `CynefinDomain`, `AnalyticalEngine` trait, `PolicyDecision`, `AuditEntry`, `EpistemicState` |
| **[cynepic-guardian](crates/cynepic-guardian)** | ~700 | 22 | Policy chains, Rego engine, circuit breaker, loop detection (O(1) VecDeque), rate limiting, HITL escalation, bias auditing, audit trail |
| **[cynepic-causal](crates/cynepic-causal)** | ~1,400 | 26 | Causal DAG, d-separation, backdoor/front-door criteria, OLS/IPW/IV/2SLS estimation, 4 refutation tests, counterfactual reasoning |
| **[cynepic-router](crates/cynepic-router)** | ~500 | 13 | Cynefin keyword classifier, entropy scoring, cost-aware routing, budget tracking, drift detection, classifier metrics |
| **[cynepic-bayes](crates/cynepic-bayes)** | ~900 | 20 | Beta/Normal/Gamma/Dirichlet priors, MH/Adaptive/Multi-dim MCMC, belief tracker, tool reliability tracking |
| **[cynepic-graph](crates/cynepic-graph)** | ~700 | 10 | Typed `StateGraph<S>`, conditional edges, cycle detection, per-node timeout, checkpoint/resume, event hooks |
| **[cynepic-pyo3](bindings/pyo3)** | ~300 | — | Python bindings — `pip install` to accelerate DoWhy/PyMC with Rust |
| **[cynepic-mcp](bindings/mcp)** | ~280 | — | MCP JSON-RPC stdio server — 8 cognitive tools for AI agents |
| **[cynepic-server](crates/cynepic-server)** | ~200 | — | Axum HTTP API — 6 REST endpoints, single ~8MB binary |

**Total: 99 tests, ~7,800 LOC across 9 crates.**

---

## Quick Start

```bash
cd causalrust
cargo build --workspace              # Build all crates
cargo test --workspace               # Run all 99 tests
cargo test -p cynepic-guardian --no-default-features  # Without Rego
cargo doc --workspace --no-deps      # Generate API docs
```

**Requirements:** Rust 1.85+ (edition 2024)

---

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
| **Call from Python** to accelerate DoWhy/PyMC/numpy workflows | `cynepic-pyo3` | `pip install cynepic` → `import cynepic` | 100-1000x speedup on DAG ops, conjugate priors, policy eval |
| **Give AI agents decision intelligence** via MCP | `cynepic-mcp` | JSON-RPC stdio → 8 cognitive tools | Agents can classify, estimate ATE, check policy, update beliefs |
| **Run a lightweight decision API** without Python dependencies | `cynepic-server` | Axum binary → 6 REST endpoints | Single ~8MB binary, no venv, no pip |
| **Audit AI decisions** for compliance (EU AI Act, SOC2) | `cynepic-guardian` | AuditTrail → `recent_entries()` or `with_entries()` | Immutable append-only log, no-clone access for large trails |

### When NOT to use

- **Pure LLM orchestration** without analytical needs — use LangChain/LangGraph directly
- **Real-time <1µs latency** with zero allocations — cynepic-rs is fast but not hard-real-time
- **No Rust toolchain** and no need for Python/MCP — the standalone crates require `cargo`

---

## Performance Benchmarks

| Operation | Python Equivalent | cynepic-rs | Speedup |
|-----------|------------------|------------|---------|
| DAG d-separation | ~10ms (NetworkX) | ~10µs (petgraph) | **~1,000x** |
| Beta conjugate prior update | ~1ms (PyMC) | <1µs (direct) | **~1,000x** |
| Policy evaluation | ~1ms (OPA sidecar) | ~10µs (in-process regorus) | **~100x** |
| Circuit breaker check | ~100µs (Python) | <100ns (atomics) | **~1,000x** |
| StateGraph step | ~1ms (LangGraph) | <100µs (typed dispatch) | **~10x** |

*Measurements from empirical comparison against standard Python equivalents. See `benchmarks/` for methodology.*

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

[Business Source License 1.1](../LICENSE)

- **Free** for personal, academic, research, educational, evaluation, and development use
- **Commercial/production** use requires a paid license — contact eljailari.suhonen@gmail.com
- **Converts to Apache-2.0** on 2030-03-13 (4-year change date)
