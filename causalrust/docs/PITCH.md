# cynepic-rs — The Decision Intelligence Stack That Was Missing From Rust

## The One-Liner

**cynepic-rs** is a modular Rust toolkit that gives AI agents, MLOps pipelines, and data platforms the ability to reason causally, quantify uncertainty, and enforce policy — at the speed and safety guarantees only Rust can deliver.

---

## The Problem

Every production AI system today faces the same three gaps:

**1. AI agents can generate text, but they can't reason about cause and effect.**
An LLM can tell you that ad spend and revenue are correlated. It cannot tell you whether increasing ad spend *causes* revenue to grow, or whether both are driven by seasonality. Causal inference libraries exist in Python (DoWhy, EconML) — but they're slow on large datasets, have no type safety, and can't be embedded into real-time systems.

**2. AI systems output point predictions when decisions need uncertainty.**
A model says "churn probability: 73%." But is that 73% based on 10,000 observations or 12? Bayesian reasoning is how you answer that — yet PyMC takes seconds per inference, making it unusable for real-time adaptive systems like A/B testing platforms, clinical trial monitors, or trading engines.

**3. AI guardrails are bolted on as afterthoughts, not built into the execution path.**
Policy engines (OPA) run as sidecars. Human approval workflows (HumanLayer) are separate services. Audit trails are database writes scattered across codebases. None of this is composable, and none of it runs at the speed of the decisions being made. The EU AI Act now mandates exactly this kind of integrated governance — and most teams are scrambling.

**The root cause:** Python owns the intelligence layer but can't deliver the performance, safety, or embeddability that production systems need. The Rust ecosystem has world-class infrastructure (Tokio, Axum, Arrow, Polars) but zero libraries for *decision intelligence* — the layer between raw data and business outcomes where correctness, auditability, and latency all matter simultaneously.

---

## What cynepic-rs Is

Six independent Rust crates, each solving one piece of the decision intelligence puzzle:

| Crate | What It Does | The Python It Replaces |
|-------|-------------|----------------------|
| **cynepic-causal** | Causal DAGs, effect identification, ATE estimation, refutation tests | DoWhy + EconML |
| **cynepic-bayes** | Conjugate priors, MCMC sampling, real-time belief updating | PyMC (for the fast path) |
| **cynepic-guardian** | Rego policy evaluation, circuit breakers, append-only audit trails | OPA sidecar + custom audit code |
| **cynepic-router** | Cynefin complexity classification, cost-aware LLM routing | LangChain routing chains |
| **cynepic-graph** | Typed async workflow orchestration with conditional branching | LangGraph |
| **cynepic-core** | Shared types, traits, error handling | (glue code) |

Each crate is independently publishable. Use one, use all, or compose them into a full decision pipeline.

---

## Why It Matters Now

### The Agentic AI Inflection

2025-2026 is the year AI agents went from demos to production. Claude, GPT, and open-source models now take real actions — executing code, calling APIs, making purchases, deploying infrastructure. Every one of these agent actions needs:

- **Causal reasoning** — "If I change this config, what will happen?" (not just "what happened before?")
- **Uncertainty quantification** — "How confident am I in this action?" (not just "what's the most likely outcome?")
- **Policy enforcement** — "Am I allowed to do this?" (measured at 5.6µs in-process, not after the fact)
- **Auditable workflows** — "Show me every decision, who approved it, and why" (EU AI Act, SOC2, HIPAA)

cynepic-rs is the first toolkit that packages all four as composable, embeddable Rust libraries.

### The Performance Gap, Measured

Everything below comes from a committed harness run on one machine — the Python
side from `scripts/compare_{networkx,langgraph,opa}.py`, the Rust side from the
`*_latency` examples. Reproduce it or discount it.

| Operation | Python | cynepic-rs | Speedup |
|-----------|--------|------------|---------|
| DAG d-separation, 100 nodes | 49.3µs (NetworkX) | 4.6µs (petgraph) | **11x** |
| Workflow step | 58.3µs (LangGraph) | 0.57µs (StateGraph) | **102x** |
| Policy evaluation, vs sidecar | 320.1µs (OPA over HTTP) | 5.6µs (regorus) | **57x** |
| Policy evaluation, vs engine | 17.6µs (OPA's evaluator) | 5.6µs (regorus) | **3.1x** |
| Beta credible interval | 42.4µs (scipy/Boost) | 14.4µs | **2.9x** |

Two things this table is careful about, because an earlier version of it was
not:

**The policy row is two rows.** OPA reports its own
`timer_rego_query_eval_ns`, and **94.5% of a sidecar round trip is HTTP, JSON
and the loopback hop** — not policy evaluation. Deleting a sidecar is worth 57x
and is a real win; it is not a claim about regorus, which is worth 3.1x.
Quoting the combined figure as an engine comparison would overstate it
thirty-fold.

**These are dispatch and arithmetic, not end-to-end workloads.** LangGraph
carries checkpointing, a reducer model and interrupt support through every
step, so part of that 102x is machinery rather than waste. A dispatch win is
not a claim that one library replaces the other.

Two rows that used to be here have been removed rather than measured, because
no fair version of them exists: a Beta conjugate update is two additions and
PyMC has no such primitive, and a circuit-breaker check is an atomic load
against a Python attribute access. Both are noted in
[`../README.md`](../README.md) so a reader who met the claim elsewhere finds
out here that it does not hold up.

**And speed is not the main reason to use this.** For causal inference the
product is a number someone will act on. `ols_adjusted` and `ipw` achieve
nominal confidence-interval coverage on every cell of a nine-cell DGP grid;
the estimators refuse rather than returning a figure when the data cannot
support one, at a measured false-answer rate of zero. That is the claim worth
making, and [`FINDINGS.md`](FINDINGS.md) records what it cost to be able to
make it.

### The Regulatory Tailwind

The EU AI Act (effective 2026) mandates:
- Human oversight for high-risk AI decisions
- Audit trails for automated decision-making
- Policy enforcement *before* actions are taken, not after

cynepic-guardian's append-only audit trail, policy chain evaluation, and circuit breaker pattern directly address these requirements. Building compliance into the execution path (not as a logging afterthought) is a structural advantage.

---

## Use Cases

[Apache License 2.0](../../LICENSE) — free for any use, including commercial and production, with a patent grant. Relicensed from BSL 1.1 on 2026-08-17.

### 1. Causal A/B Testing at Scale
**Who:** Data platforms, experimentation teams, growth engineering
**Problem:** Standard A/B tests assume random assignment. Real-world experiments have confounders — users self-select, seasonality shifts, marketing campaigns overlap.
**cynepic solution:** `cynepic-causal` identifies confounders via the backdoor criterion, adjusts estimates accordingly, and runs automated refutation tests. `cynepic-bayes` provides Bayesian stopping rules instead of fixed-horizon p-values. Interval coverage is measured on a nine-cell grid, not assumed — see [FINDINGS.md](FINDINGS.md).

### 2. AI Agent Guardrails
**Who:** Any team deploying autonomous AI agents (customer service, code generation, financial operations)
**Problem:** An agent with tool access can do real damage. Policy evaluation must be faster than the agent's action loop.
**cynepic solution:** `cynepic-guardian` evaluates a Rego policy in **5.6µs** measured, embedded in the agent's execution path rather than as a sidecar — and the always-on guardrails cost less again: a circuit-breaker check is 16ns, a rate-limiter check 112ns. That matters because a guardrail sits on *every* call by construction, and one that costs more than the thing it guards gets sampled instead of applied. Circuit breaker trips after repeated failures. Every decision is audit-logged with UUID, timestamp, and full context. `cynepic-graph` orchestrates the workflow with compile-time type safety (the Rust compiler *proves* every branch is handled).

### 3. Adaptive Clinical Trial Monitoring
**Who:** Pharma, biotech, CROs
**Problem:** Traditional trials use fixed sample sizes. Bayesian adaptive designs can stop early (saving time and lives) but require real-time posterior computation.
**cynepic solution:** `cynepic-bayes` computes Beta-Binomial posterior updates in closed form, with *exact* Beta quantile intervals rather than a normal approximation. Conjugate priors cover the vast majority of clinical endpoints (binary outcomes, continuous measures, count data). `cynepic-guardian` enforces regulatory policies (e.g., "cannot stop trial before minimum enrollment").

### 4. LLM Cost Optimization
**Who:** Any company spending >$10K/month on LLM APIs
**Problem:** Sending every query to GPT-4/Claude is expensive. Most queries are simple and could be handled by cheaper models.
**cynepic solution:** `cynepic-router` classifies query complexity into Cynefin domains. "Clear" queries → cheap local model. "Complicated" → mid-tier model with causal tools. "Complex" → expensive frontier model. Cost tiers are configurable.

**What the classifier is actually worth, measured.** `LexicalClassifier` scores
**macro F1 0.656** and **0.625 recall on Chaotic** under 4-fold
cross-validation against a 96-query labelled corpus — against 0.25 for random
guessing, and against **0.290 / 0.000** for the keyword classifier it replaces.
That is useful and it is not accurate enough to route unsupervised: no saving
figure is claimed here, because none has been measured, and "no quality
degradation" would be false at 0.656.

What makes it deployable anyway is that it **abstains**. Input it cannot read
returns `Disorder` at exactly zero confidence with high entropy, which is the
signal an escalation policy triggers on, and nothing contentless is ever
answered with the confidence that authorises answering from cache. A confident
wrong route is worse than an admitted unknown at every ratio. See
[FINDINGS.md](FINDINGS.md#r1).

### 5. MLOps Pipeline Governance
**Who:** ML platform teams, data engineering
**Problem:** Models get deployed without proper validation. Feature pipelines change upstream and break downstream models. Nobody knows who approved what.
**cynepic solution:** `cynepic-causal` validates that the causal assumptions behind a model still hold when data changes. `cynepic-guardian` enforces deployment policies ("model must pass refutation tests before promotion"). `cynepic-graph` orchestrates the validation → approval → deployment workflow with full audit trail.

### 6. Real-Time Fraud / Anomaly Detection
**Who:** FinTech, payments, cybersecurity
**Problem:** Traditional rule engines are brittle. ML models produce scores without uncertainty bounds. False positives are expensive.
**cynepic solution:** `cynepic-bayes` maintains a belief state per entity that updates in real-time as new transactions arrive. `cynepic-guardian` enforces risk thresholds with circuit breakers (auto-block if anomaly rate spikes). The pipeline is allocation-light and synchronous; the per-call latencies of its parts are measured (`guardian_latency`, `bayes_latency`), though this end-to-end pipeline as a whole is not.

---

## Competitive Positioning

### vs. Python Libraries (DoWhy, PyMC, LangGraph)

cynepic-rs is **not a replacement** — it's an **accelerator and embedding layer**. Python libraries have richer APIs and larger communities. cynepic-rs wins on:

- **Performance**: measured per operation against NetworkX, LangGraph, OPA and scipy — see the table above and [../README.md](../README.md). Two of the four original assumptions were wrong in opposite directions, which is why the labels matter.
- **Embeddability**: Compiles to a static library, WASM module, or Python extension — no runtime, no GC, no interpreter
- **Type safety**: The Rust compiler catches errors that Python finds at runtime (or never)
- **Memory safety**: No segfaults, no data races, no buffer overflows — critical for security-sensitive policy evaluation

The pragmatic path: PyO3 bindings let Python teams use cynepic as a drop-in accelerator for their existing DoWhy/PyMC workflows. They don't need to learn Rust.

### vs. Rust ML Ecosystem (Polars, Candle, Burn)

cynepic-rs **complements** these libraries. It does not depend on them today — Polars, Candle and Burn are all planned rather than present (see [roadmap.md](roadmap.md)). The gap cynepic fills is the *decision layer* above raw ML: causal identification, Bayesian reasoning, policy enforcement, workflow orchestration. Nobody else in the Rust ecosystem is building this.

### vs. Cloud AI Platforms (Vertex AI, SageMaker, Azure ML)

Cloud platforms are monolithic and vendor-locked. cynepic-rs is modular and runs anywhere — laptop, Kubernetes, edge device, Cloudflare Worker (via WASM), Spark executor (via JNI). No cloud account required. No API calls. No egress fees. Full data sovereignty.

### vs. LangChain / CrewAI / Agent Frameworks

Agent frameworks focus on the LLM orchestration loop. cynepic-rs focuses on *what happens between LLM calls* — the reasoning, validation, and governance that makes agent actions trustworthy. cynepic-graph is a workflow engine; cynepic-causal/bayes are the analytical tools the agent invokes; cynepic-guardian is the policy layer that approves or blocks. These are orthogonal to — and composable with — any agent framework.

---

## How to Leverage cynepic-rs

### For Rust Developers
```bash
cargo add cynepic-causal cynepic-bayes cynepic-guardian
```
Use individual crates in your Rust services. Each crate has zero-config defaults and builder-pattern APIs. No framework lock-in.

### For Python Data Scientists
```bash
pip install cynepic  # (planned — PyO3 bindings)
```
Drop-in acceleration for DoWhy causal identification and OPA policy evaluation, via the PyO3 bindings. The OPA comparison is measured (57x against a sidecar, 3.1x against its evaluator); DoWhy is not. PyMC is deliberately not claimed — a conjugate update is two additions and PyMC has no equivalent primitive, so no fair comparison exists.

### For AI Agent Builders
Configure cynepic-mcp as an MCP tool server. Your Claude/GPT/local agent gets five new tools: `classify_query`, `estimate_treatment_effect`, `bayesian_update`, `evaluate_policy`, `build_causal_dag`. The agent can now reason causally, quantify uncertainty, and check policies — without any Rust code.

### For Platform Teams
Deploy cynepic-server as a sidecar container. REST API with OpenAPI spec. Every microservice in your platform gets access to causal inference, Bayesian reasoning, and policy evaluation via HTTP. Docker image, Kubernetes-ready.

### For Data Engineers
cynepic-causal takes `ndarray` arrays today; Arrow/Parquet ingestion via Polars is planned, not present. Integrate into Spark (JNI UDF), Dagster/Prefect (task nodes), or dbt (causal assumption validation against model lineage). Audit trails export as OpenTelemetry spans.

---

## The Moat

1. **First mover**: There is no causal inference library in Rust. No Bayesian PPL. No typed agent workflow engine. cynepic-rs defines these categories.

2. **Compound value**: Each crate is useful alone, but they compose into something no competitor offers — a full decision intelligence pipeline from classification through reasoning through governance, in a single process, with compile-time safety guarantees.

3. **Multi-surface accessibility**: Rust library + Python extension + HTTP API + MCP tools + WASM module. Same core, five consumption patterns. This means cynepic reaches Rust developers, Python data scientists, full-stack engineers, AI agent builders, and data engineers — all from one codebase.

4. **Regulatory alignment**: The EU AI Act creates mandatory demand for exactly the governance patterns cynepic-guardian implements. This isn't a feature — it's a compliance requirement with legal deadlines.

5. **Performance as architecture**: Sub-millisecond causal identification and policy evaluation aren't just "nice to have" — they enable architectures that Python literally cannot support (real-time agent guardrails, streaming Bayesian updates, in-path policy enforcement).

---

## Status & Roadmap

**Now (v0.2):** All 6 crates production-ready. ~6,800 LOC, 99 tests. Full causal pipeline (DAG → identify → estimate → refute), 4 conjugate priors + 3 MCMC samplers, policy chains with Rego/circuit breaker/rate limiting/HITL escalation, cost-aware routing, typed workflow graphs with checkpointing.

**Next (v0.3):** PyO3 Python bindings, HTTP API, MCP tool server — the accessibility layer.

**Then (v0.4+):** Embedding-based classifier, HMC/NUTS sampler, Cedar policies, parallel graph execution, crates.io publish.

The core architecture is proven. What remains is deepening each crate's capabilities and building the integration interfaces that make it accessible to every ecosystem.

---

*cynepic-rs: Decision intelligence infrastructure for the age of autonomous AI.*
