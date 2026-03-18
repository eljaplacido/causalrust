# cynepic-rs — Decision Intelligence Infrastructure for Rust

## The One-Liner

**cynepic-rs** is a modular Rust toolkit that gives AI agents, MLOps pipelines, and data platforms the ability to reason causally, quantify uncertainty, and enforce policy — at the speed and safety guarantees only Rust can deliver.

---

## The Problem

Production AI systems face three common gaps:

1. **Causal reasoning** — AI agents can correlate but cannot distinguish causation from correlation. Causal inference libraries exist in Python but lack type safety, embeddability, and real-time performance.

2. **Uncertainty quantification** — Models output point predictions when decisions need calibrated confidence intervals. Bayesian approaches are too slow in Python for real-time use.

3. **Integrated governance** — Policy engines, human approval workflows, and audit trails are bolted on as afterthoughts. The EU AI Act mandates integrated governance for automated decision-making.

**Root cause:** Python owns the intelligence layer but cannot deliver the performance, safety, or embeddability that production systems need. The Rust ecosystem has world-class infrastructure but no libraries for *decision intelligence*.

---

## What cynepic-rs Provides

Six independent Rust crates, each solving one piece of the decision intelligence puzzle:

| Crate | Capability |
|-------|-----------|
| **cynepic-causal** | Causal DAGs, effect estimation, refutation, counterfactual reasoning |
| **cynepic-bayes** | Conjugate priors, MCMC sampling, streaming belief updates |
| **cynepic-guardian** | Policy evaluation, circuit breakers, bias auditing, audit trails |
| **cynepic-router** | Complexity classification, cost-aware routing, drift detection |
| **cynepic-graph** | Typed async workflow orchestration with checkpointing |
| **cynepic-core** | Shared types, traits, epistemic state provenance |

Each crate is independently publishable. Use one, use all, or compose them.

---

## Integration Targets

| Interface | Status |
|-----------|--------|
| Rust library | **Available** |
| Python (PyO3) | Planned |
| HTTP API | Planned |
| MCP tools | Planned |
| WASM | Planned |

---

## License

[Business Source License 1.1](../../LICENSE) — free for non-production use, converts to Apache 2.0 on 2030-03-13.

See [NOTICE](../../NOTICE) for trademark attribution and IP classification.
