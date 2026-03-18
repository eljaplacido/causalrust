# Workflow Integration Patterns

> High-level patterns for integrating cynepic-rs into software development, agentic AI, and DevOps/MLOps workflows.

See [NOTICE](../../NOTICE) for IP classification. Detailed implementation patterns are available under commercial license.

---

## Software Development

### Feature Flag Decisions
Use `cynepic-causal` to identify confounders in A/B tests and estimate causal effects rather than relying on correlation. Combine with `cynepic-bayes` for Bayesian stopping rules.

### Code Review Risk Assessment
Use `cynepic-router` to classify change complexity, `cynepic-bayes` to estimate incident probability from historical data, and `cynepic-guardian` to enforce review policies.

### Dependency Update Impact Analysis
Use `cynepic-causal` for dependency impact DAGs and `cynepic-guardian` for policy-gated auto-merge decisions with circuit breaker protection.

---

## Agentic Workflows

### Enhanced Agent Decision Loop
Standard agents: Observe → Think → Act. With cynepic: Observe → Classify → Reason → Validate → Act → Audit. The structured reasoning steps use formal methods (causal inference, Bayesian updating, policy evaluation) rather than ungrounded LLM intuition.

### Governed Tool Use
`cynepic-guardian` evaluates policies in the agent's execution path (not as a sidecar). Every tool invocation is checked against policies and audit-logged. `cynepic-graph` orchestrates multi-step workflows with compile-time type safety.

### Tool Reliability Monitoring
`cynepic-bayes` (ToolBeliefSet) tracks success/failure of external tools using Beta-prior belief updates. Circuit-break unreliable tools automatically.

---

## DevOps & MLOps

### CI/CD Pipeline Governance
Use `cynepic-guardian` policy chains as deployment gates: all policies must approve before promotion. Circuit breaker prevents cascading deployment failures. Full audit trail for every gate decision.

### ML Model Validation
`cynepic-causal` validates causal assumptions behind model features. `cynepic-bayes` provides confidence intervals on evaluation metrics (not just point estimates). `cynepic-guardian` enforces policies on confidence intervals, not point values.

### Data Pipeline Monitoring
`cynepic-bayes` maintains per-table/per-column belief states for streaming anomaly detection. `cynepic-router` classifies anomaly severity. `cynepic-guardian` routes alerts based on severity and risk.

### Drift Detection
`cynepic-router` (DriftDetector) monitors routing distribution changes via KL-divergence to detect shifts in query complexity patterns.

---

## Consumption Methods

| Method | Best For | Latency |
|--------|---------|---------|
| Rust library | Embedded in Rust services | μs |
| Python extension | Data science, ML pipelines | μs (in-process) |
| HTTP API | Polyglot microservices | ~1ms (network) |
| MCP tools | AI agent tooling | ~1ms (stdio) |
| WASM | Browser/edge compute | μs (in-process) |
