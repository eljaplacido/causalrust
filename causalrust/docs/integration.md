# Integration & Interoperability Guide — cynepic-rs

## Design Principle

Every cynepic crate is **library-first**: no side effects on import, no global state, all types are `Serialize + Deserialize`. This makes each crate consumable as:

1. **Rust library** — `cargo add cynepic-causal`
2. **Python extension** — via PyO3/maturin
3. **HTTP API** — via cynepic-server (Axum)
4. **MCP tool** — via cynepic-mcp (JSON-RPC stdio)
5. **WASM module** — for browser/edge compute
6. **CLI tool** — thin binary wrapping library calls

---

## Python Ecosystem Integration

### PyO3 Binding Strategy

Each crate gets a `#[pymodule]` when the `pyo3` feature is enabled. The `cynepic-py` package unifies them under a single `pip install cynepic` namespace.

```python
# Target API
from cynepic import CausalDag, BackdoorCriterion, BetaBinomial, PolicyChain

dag = CausalDag()
dag.add_variable("X")
dag.add_variable("Y")
dag.add_variable("Z")
dag.add_edge("X", "Y")
dag.add_edge("Z", "X")
dag.add_edge("Z", "Y")

adjustment = BackdoorCriterion.find(dag, treatment="X", outcome="Y")
# Returns: {"Z"}
```

### Interop with Python ML/Data Libraries

| Python Library | cynepic Crate | Integration Path |
|---------------|---------------|-----------------|
| **DoWhy** | cynepic-causal | PyO3 bindings accelerate DAG operations and identification. Use as a fast backend for DoWhy's `CausalModel.identify_effect()`. |
| **EconML** | cynepic-causal | Share treatment effect estimates via Arrow/IPC. cynepic produces `ATEResult`, EconML consumes for heterogeneous effects. |
| **PyMC / ArviZ** | cynepic-bayes | Export MCMC samples as numpy arrays (zero-copy via PyO3 numpy). ArviZ `InferenceData` from cynepic samples. |
| **NumPyro / JAX** | cynepic-bayes | Conjugate prior results as JAX arrays for downstream variational inference. |
| **Polars (Python)** | cynepic-causal | Zero-copy DataFrame sharing via Arrow IPC between Polars-py and cynepic-causal's Polars backend. |
| **Pandas** | cynepic-causal | Convert via Arrow: `pandas.DataFrame → pyarrow.Table → cynepic`. Avoids copy overhead. |
| **LangChain / LangGraph** | cynepic-graph | cynepic-graph as a fast execution backend. Python LangGraph defines the graph; Rust executes it. |
| **CrewAI / AutoGen** | cynepic-router | CynefinRouter as a tool callable by CrewAI agents for query classification. |
| **MLflow / W&B** | cynepic-bayes | Log posterior summaries, treatment effects, and audit trails as MLflow metrics/artifacts. |
| **Prefect / Airflow** | cynepic-graph | cynepic-graph workflows as Prefect tasks. Each node becomes a task in the DAG. |
| **OPA (Python client)** | cynepic-guardian | Drop-in replacement: `RegoPolicyEvaluator` evaluates the same Rego policies, 8x faster than OPA sidecar. |
| **HumanLayer** | cynepic-guardian | Future HITL integration: guardian escalation triggers HumanLayer approval, webhook resumes workflow. |
| **dbt** | cynepic-causal | Causal DAGs validated against dbt model lineage. Ensure data pipeline matches causal assumptions. |

### numpy Zero-Copy Pattern

```rust
// In cynepic-py bindings
#[pyfunction]
fn difference_in_means<'py>(
    py: Python<'py>,
    treatment: PyReadonlyArray1<f64>,
    outcome: PyReadonlyArray1<f64>,
) -> PyResult<(f64, f64)> {
    let t = treatment.as_array();
    let o = outcome.as_array();
    let result = LinearATEEstimator::difference_in_means(
        &t.to_owned(), &o.to_owned()
    );
    Ok((result.ate, result.std_error))
}
```

---

## TypeScript / JavaScript Ecosystem

### WASM Compilation

Core crates (core, causal, bayes) can compile to `wasm32-unknown-unknown` since they have no system dependencies.

```bash
cd crates/cynepic-bayes
wasm-pack build --target web
```

```typescript
// Browser usage
import init, { BetaBinomial } from 'cynepic-bayes';
await init();

const prior = BetaBinomial.new(1.0, 1.0);
prior.update(10, 3); // 10 successes, 3 failures
console.log(prior.mean()); // ~0.786
```

### Interop with TS/JS Libraries

| JS/TS Library | cynepic Crate | Integration Path |
|--------------|---------------|-----------------|
| **LangChain.js** | cynepic-router | WASM classifier as a custom LangChain tool. Or HTTP API via cynepic-server. |
| **Vercel AI SDK** | cynepic-graph | cynepic-graph as a streaming backend (Axum SSE → Vercel AI SDK `useChat`). |
| **ModelFusion** | cynepic-router | CynefinRouter as a model selection strategy. |
| **Rete.js** | cynepic-graph | Visual graph editor (Rete.js) → export JSON → cynepic-graph executes. |
| **Observable / D3** | cynepic-causal | Export DAGs as JSON adjacency lists for D3 force-directed visualization. |
| **Apache Arrow JS** | cynepic-causal | Arrow IPC for zero-copy data exchange between JS DataFrames and Rust. |
| **TensorFlow.js** | cynepic-bayes | WASM belief updates + TFJS model inference in the same browser context. |

---

## Java / JVM Ecosystem

### JNI via jni-rs

For Java/Kotlin/Scala integration, cynepic crates expose a C ABI via `#[no_mangle] extern "C"` functions, consumed through JNI.

| JVM Library | cynepic Crate | Integration Path |
|------------|---------------|-----------------|
| **Apache Spark** | cynepic-causal | UDF wrapping cynepic ATE estimator. Process partitions in Rust for 10-50x speedup. |
| **Apache Flink** | cynepic-graph | cynepic-graph as a Flink ProcessFunction for stateful event processing. |
| **Kafka Streams** | cynepic-router | Classification as a Kafka Streams transformer. |
| **Spring Boot** | cynepic-server | HTTP API consumed as a Spring WebClient service. |
| **OPA Java SDK** | cynepic-guardian | Same Rego policies, Rust-native evaluation via JNI. |

---

## MCP (Model Context Protocol) Integration

### Tool Definitions

cynepic-mcp exposes each crate as an MCP tool:

```json
{
  "tools": [
    {
      "name": "classify_query",
      "description": "Classify a query into a Cynefin complexity domain",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": { "type": "string" }
        },
        "required": ["query"]
      }
    },
    {
      "name": "estimate_treatment_effect",
      "description": "Estimate average treatment effect from observational data",
      "inputSchema": {
        "type": "object",
        "properties": {
          "treatment": { "type": "array", "items": { "type": "number" } },
          "outcome": { "type": "array", "items": { "type": "number" } }
        },
        "required": ["treatment", "outcome"]
      }
    },
    {
      "name": "bayesian_update",
      "description": "Update a Bayesian belief state with new evidence",
      "inputSchema": {
        "type": "object",
        "properties": {
          "prior_type": { "enum": ["beta_binomial", "normal_normal", "gamma_poisson"] },
          "prior_params": { "type": "object" },
          "observations": { "type": "object" }
        },
        "required": ["prior_type", "prior_params", "observations"]
      }
    },
    {
      "name": "evaluate_policy",
      "description": "Evaluate an action against a Rego policy",
      "inputSchema": {
        "type": "object",
        "properties": {
          "action": { "type": "string" },
          "context": { "type": "object" },
          "policy": { "type": "string" }
        },
        "required": ["action", "context", "policy"]
      }
    },
    {
      "name": "build_causal_dag",
      "description": "Build a causal DAG and find adjustment sets",
      "inputSchema": {
        "type": "object",
        "properties": {
          "variables": { "type": "array", "items": { "type": "string" } },
          "edges": { "type": "array", "items": { "type": "array", "items": { "type": "string" } } },
          "treatment": { "type": "string" },
          "outcome": { "type": "string" }
        },
        "required": ["variables", "edges", "treatment", "outcome"]
      }
    }
  ]
}
```

### Usage with AI Agents

```
Human: What's the causal effect of increasing ad spend on revenue,
       controlling for seasonality?

Agent → MCP → classify_query("causal effect of ad spend on revenue")
      → domain: Complicated, confidence: 0.87

Agent → MCP → build_causal_dag(
        variables: ["ad_spend", "revenue", "seasonality"],
        edges: [["ad_spend", "revenue"], ["seasonality", "ad_spend"],
                ["seasonality", "revenue"]],
        treatment: "ad_spend", outcome: "revenue")
      → adjustment_set: {"seasonality"}

Agent → MCP → estimate_treatment_effect(treatment: [...], outcome: [...])
      → ate: 2.34, std_error: 0.41
```

---

## Data Engineering Integration

### Apache Arrow / Parquet

cynepic-causal currently uses ndarray for numeric operations. A future Polars backend is planned (v0.4+) to enable Arrow-native data flows:

```
Parquet file → Polars LazyFrame → cynepic-causal (Rust)   [planned]
             → ATE estimation → JSON result
             → or Arrow IPC → Python/Spark consumer
```

### Integration Points

| Tool | Integration |
|------|------------|
| **dbt** | Validate causal DAG assumptions against dbt model lineage |
| **Great Expectations** | Causal assertions as data quality checks |
| **Dagster / Prefect** | cynepic-graph nodes as orchestrator tasks |
| **Delta Lake / Iceberg** | Read treatment/outcome data from lakehouse tables via Polars |
| **Kafka / Redpanda** | Stream observations → cynepic-bayes for real-time belief updates |
| **OpenTelemetry** | Export audit trails as OTel spans via `tracing-opentelemetry` |

---

## MLOps Integration

### Experiment Tracking

```python
# MLflow integration pattern
import mlflow
from cynepic import estimate_ate, bayesian_update

with mlflow.start_run():
    result = estimate_ate(treatment, outcome)
    mlflow.log_metric("ate", result.ate)
    mlflow.log_metric("ate_std_error", result.std_error)

    belief = bayesian_update("beta_binomial", alpha=1, beta=1, successes=50, failures=10)
    mlflow.log_metric("posterior_mean", belief.mean)
    mlflow.log_metric("posterior_variance", belief.variance)
```

### Model Registry

cynepic models (DAGs, belief states, policies) are serializable JSON — store in any model registry:

```python
# Register a causal DAG as an MLflow model artifact
dag_json = dag.to_json()
mlflow.log_dict(dag_json, "causal_dag.json")
```

---

## Deployment Patterns

| Pattern | Stack | Use Case |
|---------|-------|----------|
| **Sidecar** | cynepic-server Docker container | Kubernetes pod alongside Python ML service |
| **Embedded library** | `cargo add cynepic-*` | Rust microservice with native causal/Bayesian capabilities |
| **Python extension** | `pip install cynepic` | Drop-in acceleration for DoWhy/PyMC workflows |
| **MCP tool** | cynepic-mcp binary | AI agent tooling (Claude, GPT, local models) |
| **Serverless** | WASM on Cloudflare Workers | Edge causal inference / Bayesian updates |
| **Spark UDF** | JNI + cynepic-causal | Distributed treatment effect estimation |
