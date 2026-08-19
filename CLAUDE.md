# cynepic-rs — CLAUDE.md

> Agentic navigation guide for the cynepic-rs monorepo.
> This file is optimized for LLM context — keep it under 300 lines.

## Identity

**cynepic-rs** is a Cargo workspace of 6 independently publishable Rust crates implementing the [CARF/CYNEPIC](https://github.com/eljaplacido/projectcarfcynepic/) decision intelligence architecture. The goal is modular, library-first components usable as standalone Rust crates, PyO3 Python extensions, MCP tools, or HTTP API endpoints for agentic AI, MLOps, and data engineering workflows.

## Repository Layout

```
causalrust/                    # Git root
├── CLAUDE.md                  # This file (agentic nav)
├── NOTICE                     # IP classification & trademark attribution
├── LICENSE                    # Apache-2.0 (relicensed from BSL 1.1, 2026-08-17)
├── causalrust/                # Cargo workspace root
│   ├── Cargo.toml             # Workspace manifest (resolver 2)
│   ├── Cargo.lock             # Pinned dependencies
│   ├── README.md              # Quick start
│   ├── deny.toml              # cargo-deny policy (licences, advisories, bans)
│   ├── crates/
│   │   ├── cynepic-core/      # Shared types, traits, errors, epistemic state
│   │   ├── cynepic-guardian/   # Policy guardrails, circuit breaker, bias audit
│   │   ├── cynepic-causal/    # Causal inference (DAG, ATE, counterfactual)
│   │   │   ├── tests/findings.rs         # Executable specs for the open findings
│   │   │   ├── benches/estimators.rs     # Criterion; correctness gates the claim
│   │   │   └── examples/coverage_report.rs  # The credibility artifact
│   │   ├── cynepic-router/    # Cynefin classifier + routing + drift detection
│   │   ├── cynepic-bayes/     # Bayesian priors, MH sampler, belief state
│   │   │   └── benches/samplers.rs       # Time per EFFECTIVE sample, not per draw
│   │   ├── cynepic-graph/     # StateGraph<S> workflow orchestration
│   │   └── cynepic-testkit/   # Ground-truth DGPs + coverage harness (publish=false)
│   ├── scripts/
│   │   └── findings-ratchet.sh  # CI gate: open findings count down only
│   ├── EXPERIMENTS.md         # Quick-start examples
│   ├── research.md            # Landscape research -> modules, verticals, sequencing
│   └── docs/
│       ├── FINDINGS.md        # SINGLE SOURCE for the C-series correctness findings
│       ├── architecture.md    # Architecture overview
│       ├── integration.md     # Interop guide (Python, TS, Java, MCP)
│       ├── roadmap.md         # Completion roadmap with phases
│       ├── CRATE_GUIDE.md     # Per-crate public API guide
│       ├── WORKFLOWS.md       # Workflow integration patterns
│       └── PITCH.md           # Positioning summary
```

## Build & Test

```bash
cd causalrust/causalrust
cargo build --workspace            # Build all crates
cargo test --workspace --all-features   # 148 unit tests + 2 doctests
cargo test -p cynepic-core         # Test single crate
cargo test -p cynepic-guardian --no-default-features  # Guardian without rego
cargo test -p cynepic-guardian --features rego         # Guardian with rego
```

The full gate CI enforces (see `.github/workflows/ci.yml`):

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo clippy --workspace --all-targets --no-default-features -- -D warnings
cargo test --workspace --all-features
cargo deny check all                                     # cargo install cargo-deny
cargo hack check --workspace --feature-powerset --no-dev-deps
cargo check --target wasm32-wasip1 -p cynepic-core -p cynepic-causal -p cynepic-bayes
```

**Lint policy** lives in `[workspace.lints]` in `causalrust/Cargo.toml`, and each
crate opts in with `[lints] workspace = true`. Entries set to `allow` there are
tracked debt with a measured violation count and the tier that clears them —
they are not settled policy. Do not add new violations of a debt lint.

- **Rust edition**: 2024, MSRV 1.85
- **Windows note**: `nalgebra` (transitive via `statrs`) may fail with `msvc_spectre_libs` — use `rustup default stable-x86_64-pc-windows-msvc` and install Spectre-mitigated libs via VS Installer, or build on Linux/WSL.

## Crate Dependency Graph

```
cynepic-core (no internal deps)
  ├── cynepic-guardian  (core)
  ├── cynepic-causal    (core)
  ├── cynepic-router    (core)
  ├── cynepic-bayes     (core)
  └── cynepic-graph     (core)
```

No circular dependencies. Each crate re-exports `cynepic-core`.

## Key Types & Traits

| Type | Crate | Purpose |
|------|-------|---------|
| `CynefinDomain` | core | Enum: Clear, Complicated, Complex, Chaotic, Disorder |
| `AnalyticalEngine` | core | Async trait for domain-specific analysis |
| `PolicyDecision` | core | Approve / Reject { reason } / Escalate { target } |
| `AuditEntry` | core | Append-only audit record (UUID, timestamp, decision) |
| `EpistemicState` | core | Unified session provenance (domain, confidence, reasoning chain) |
| `ConfidenceLevel` | core | Discretized confidence: High / Medium / Low / Unknown |
| `ReasoningStep` | core | Single step in the epistemic reasoning chain |
| `CircuitBreaker` | guardian | State machine: Closed → Open → HalfOpen |
| `PolicyChain` | guardian | Sequential evaluator chain, short-circuits on reject |
| `LoopDetector` | guardian | Detects node overvisits and alternation thrashing |
| `RiskAwareEvaluator` | guardian | Bayesian risk score → approve/escalate/reject |
| `RateLimiter` | guardian | Token-bucket rate limiting per action/actor |
| `EscalationManager` | guardian | HITL escalation lifecycle (pending/approved/rejected/timed-out) |
| `BiasAuditor` | guardian | Chi-squared fairness testing on decision distributions |
| `CausalDag` | causal | petgraph-backed DAG with parent/child queries |
| `d_separated` | causal | Bayes-Ball d-separation test on DAG |
| `BackdoorCriterion` | causal | Finds valid adjustment sets for causal identification |
| `FrontDoorCriterion` | causal | Finds front-door adjustment sets via mediators |
| `LinearATEEstimator` | causal | Difference-in-means + full OLS with covariate adjustment |
| `PropensityScoreEstimator` | causal | IPW estimation via logistic regression |
| `IVEstimator` | causal | Two-stage least squares (2SLS) |
| `ATEResult` | causal | Average Treatment Effect + standard error |
| `CounterfactualEngine` | causal | Level-3 counterfactual queries (Pearl's ladder) |
| `CounterfactualQuery` | causal | "What would Y be if T had been t'?" |
| `BetaBinomial` | bayes | Conjugate prior for binary outcomes |
| `DirichletMultinomial` | bayes | Conjugate prior for categorical data |
| `MetropolisHastings` | bayes | 1D MCMC sampler for arbitrary log-densities |
| `MultiDimMH` | bayes | Multi-dimensional MH with diagonal Gaussian proposal |
| `AdaptiveMH` | bayes | Self-tuning MH (Robbins-Monro acceptance targeting) |
| `BeliefTracker` | bayes | Streaming real-time belief updates with typed observations |
| `ToolBelief` | bayes | Beta-prior reliability tracking for tools/services |
| `ToolBeliefSet` | bayes | Multi-tool reliability monitoring |
| `CynefinRouter` | router | Classifier → domain → route target (budget-aware) |
| `BudgetTracker` | router | Cost tracking with tier-based budget enforcement |
| `ClassifierMetrics` | router | Confusion matrix, precision/recall/F1, misrouting cost |
| `DriftDetector` | router | KL-divergence routing distribution drift monitoring |
| `StateGraph<S>` | graph | Typed async workflow graph with conditional edges |
| `Checkpoint<S>` | graph | Serializable execution snapshot for pause/resume |
| `GraphHook` | graph | Event hook trait for observability (NodeStarted/Completed/Failed) |
| `EventCollector` | graph | Collects graph events for testing/analysis |

## Conventions

- **Append-only audit**: `AuditTrail` entries are never mutated or deleted
- **Serde everywhere**: All public types derive `Serialize`/`Deserialize`
- **Async-first**: All engine/policy/node traits are async (tokio)
- **Feature-gated optionals**: `rego` feature on guardian, future `pyo3` features per crate
- **Error types**: Each crate has its own error enum via `thiserror`
- **No unwrap in lib code**: Tests may use `unwrap()`, library code returns `Result`

## Current Status (v0.2.0-dev, release candidate)

| Crate | Status | Tests | Key Capabilities |
|-------|--------|-------|-----------------|
| core | Complete | 13 | Domain enum, engine trait, policy types, audit types, epistemic state |
| guardian | Solid | 30 | Policy chains, Rego v1 (+ explicit v0 opt-in), circuit breaker, loop detection, rate limiting, HITL escalation, bias auditing, audit trail |
| causal | **Provisional** | 30 + 7 guards | DAG, d-separation, backdoor/front-door, OLS (validated), IPW (**broken**), IV/2SLS, 4 refutation tests, counterfactual reasoning |
| router | Solid | 17 | Keyword classifier, entropy scoring, cost-aware routing, budget tracking, drift detection, classifier metrics (F1) |
| bayes | Solid | 20 | 4 conjugate priors, 3 MCMC samplers, belief tracker, tool reliability |
| graph | Solid | 10 | StateGraph, conditional edges, cycle detection, timeout, checkpoint/resume, event hooks |
| testkit | Internal | 21 | Ground-truth DGPs, coverage/bias/RMSE harness, metamorphic relations (`publish = false`) |

**Total: 148 unit tests + 2 doctests + 13 open findings specs, 0 warnings.**

> "Solid" means the feature exists and its tests pass — not that the statistical
> output is trustworthy. Those are different claims and only one of them is now
> measured.
>
> **`cynepic-causal` is provisional.** `ols_adjusted` is validated: interval
> coverage 94.7%–97.3% with bias below 0.02 across the standard DGP grid.
> `PropensityScoreEstimator::ipw` is **broken** — 0.0% coverage, bias +1.8 to
> +9.6 — and identification, refutation and `CausalDag`'s own acyclicity
> invariant all have open defects.
>
> Nine findings, thirteen executable specs, one document:
> **[docs/FINDINGS.md](causalrust/docs/FINDINGS.md)** is the single source of
> truth. Do not restate finding detail anywhere else; link to it.
>
> ```bash
> ./scripts/findings-ratchet.sh    # the CI gate: count down only, all must fail
> cargo run -p cynepic-causal --example coverage_report --release
> ```

### Known Gaps
- **Benchmarks are baselines, not claims.** `benches/` exists (criterion, in
  causal and bayes) and CI builds but does not time them — shared-runner
  variance swamps the signal. Two rules: correctness gates the claim, so the IPW
  group is named `ipw_BROKEN_C5_C13`; and MCMC is measured in **time per
  effective sample**, never samples per second. No `fuzz/` directory.
- **Do not carry over the Python project's numbers.** `projectcarfcynepic`'s
  43/43 benchmark results describe that implementation. This workspace's IPW is
  measured at 0.0% coverage, so those figures are not evidence about these
  crates. See [research.md](causalrust/research.md).
- **WASM is partial.** `wasm32-wasip1` works for core/causal/bayes and is gated
  in CI. `wasm32-unknown-unknown` (browser) builds for nothing yet — `uuid` and
  `rand 0.8` both need a `getrandom` JS backend. guardian/graph/router pull
  `tokio = { features = ["full"] }`, which targets no wasm at all.
- **Not published.** Crates are not on crates.io, so `cargo-semver-checks` has
  no baseline and is deliberately absent from CI until the first publish.

### Future Work (not blocking release)
- **Phase 2**: PyO3 bindings, HTTP API (Axum), MCP tool server, browser WASM
- **Phase 3**: Polars backend, HMC/NUTS, embedding classifier, Cedar policies, parallel graph branches
- See [docs/roadmap.md](causalrust/docs/roadmap.md) for full details
- [research.md](causalrust/research.md) maps the landscape research to modules,
  scores each recommendation against measured status, and specs the three target
  verticals as DGP cells with tests and benchmarks

## What NOT to Do

- Do not add `unsafe` blocks without explicit approval
- Do not introduce circular crate dependencies
- Do not add runtime panics in library code (`unwrap`, `expect` → return `Result`)
- Do not fix a finding without deleting its `#[ignore]` in the same PR — the
  ratchet exists so "fixed" is an arithmetic claim, not an assertion
- Do not restate finding detail outside `docs/FINDINGS.md`
- Do not commit Cargo.lock changes without running `cargo test --workspace`
- Do not import Python/JS/TS code into the Rust crates — bindings go in a separate `bindings/` directory
