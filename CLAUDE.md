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
│   │   └── cynepic-server/    # Axum HTTP API over the crates
│   ├── bindings/
│   │   ├── pyo3/              # Python extension module
│   │   └── mcp/               # MCP server (JSON-RPC 2.0 over stdio)
│   ├── scripts/
│   │   └── findings-ratchet.sh  # CI gate: open findings count down only
│   ├── EXPERIMENTS.md         # Quick-start examples
│   ├── research.md            # Landscape research -> modules, verticals, sequencing
│   └── docs/
│       ├── FINDINGS.md        # SINGLE SOURCE for the C-series correctness findings
│       ├── architecture.md    # Architecture overview
│       ├── integration.md     # Interop guide (Python, TS, Java, MCP)
│       ├── roadmap.md         # Completion roadmap with phases
│       ├── CRATE_GUIDE.md     # Per-crate developer guide
│       ├── WORKFLOWS.md       # Real-world workflow patterns
│       └── PITCH.md           # Positioning and value proposition
```

## Build & Test

```bash
cd causalrust/causalrust
cargo build --workspace            # Build all crates
cargo test --workspace --all-features   # 404 tests (413 with --no-default-features,
                                        # which adds the pyo3 binding suite)
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

# Not in the PR gate — weekly, in .github/workflows/miri.yml. Miri must rebuild
# the whole dependency tree under its interpreter and exceeded 30 minutes even
# scoped to --lib. Low yield anyway: unsafe_code is forbid workspace-wide.
cargo miri test --lib -p cynepic-core -p cynepic-causal -p cynepic-bayes
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
| `CircuitBreaker` | guardian | State machine: Closed → Open → HalfOpen |
| `PolicyChain` | guardian | Sequential evaluator chain, short-circuits on reject |
| `LoopDetector` | guardian | Detects node overvisits and alternation thrashing |
| `RiskAwareEvaluator` | guardian | Bayesian risk score → approve/escalate/reject |
| `RateLimiter` | guardian | Token-bucket rate limiting per action/actor |
| `EscalationManager` | guardian | HITL escalation lifecycle (pending/approved/rejected/timed-out) |
| `CausalDag` | causal | petgraph-backed DAG with parent/child queries |
| `d_separated` | causal | Bayes-Ball d-separation test on DAG |
| `BackdoorCriterion` | causal | Finds valid adjustment sets for causal identification |
| `FrontDoorCriterion` | causal | Finds front-door adjustment sets via mediators |
| `LinearATEEstimator` | causal | Difference-in-means + full OLS with covariate adjustment |
| `PropensityScoreEstimator` | causal | IPW estimation via logistic regression |
| `IVEstimator` | causal | Two-stage least squares (2SLS) |
| `ATEResult` | causal | Average Treatment Effect + standard error |
| `BetaBinomial` | bayes | Conjugate prior for binary outcomes |
| `DirichletMultinomial` | bayes | Conjugate prior for categorical data |
| `MetropolisHastings` | bayes | 1D MCMC sampler for arbitrary log-densities |
| `MultiDimMH` | bayes | Multi-dimensional MH with diagonal Gaussian proposal |
| `AdaptiveMH` | bayes | Self-tuning MH (Robbins-Monro acceptance targeting) |
| `BeliefTracker` | bayes | Streaming real-time belief updates with typed observations |
| `ToolBelief` | bayes | Beta-prior reliability tracking for tools/services |
| `ToolBeliefSet` | bayes | Multi-tool reliability monitoring |
| `CynefinRouter` | router | Classifier → domain → route target (budget-aware) |
| `LexicalClassifier` | router | tf-idf nearest-centroid classifier; trainable, inspectable |
| `BudgetTracker` | router | Cost tracking with tier-based budget enforcement |
| `ClassifierMetrics` | router | Confusion matrix, precision/recall/F1, misrouting cost |
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
| guardian | **Measured** | 35 + 15 guardrails | Policy chains, Rego v1 (+ v0 opt-in), circuit breaker (**real half-open state**), loop detection, rate limiting, HITL escalation, bias auditing, audit trail |
| causal | **Validated** | 94 + 34 regressions | DAG (acyclicity enforced), d-separation, backdoor/front-door (latent-aware), OLS+QR with HC1/HC3, IPW via IRLS, IV/2SLS (LATE-labelled), refutation in SE units, counterfactual |
| router | **Measured** | 26 + 8 accuracy + 8 lexical | Keyword classifier (**macro F1 0.290, 0.000 recall on Chaotic**); `LexicalClassifier` tf-idf (**0.656 / 0.625 cross-validated**), entropy scoring, cost-aware routing, budget tracking, drift detection |
| bayes | **Calibrated** | 30 + 12 calibration | 4 conjugate priors with **exact** quantile intervals, 3 MCMC samplers (SBC-verified), belief tracker, tool reliability |
| graph | **Measured, no findings** | 10 + 15 properties | StateGraph, conditional edges, cycle detection, timeout, checkpoint/resume, event hooks |
| testkit | Internal | 44 | Ground-truth DGPs, coverage harness, metamorphic relations, Bayesian calibration + SBC, 96-query labelled routing corpus (`publish = false`) |

**Total: 404 tests (413 under `--no-default-features`) plus 22 Python tests
against the built wheel, 4 open findings specs across 5 measured crates,
0 warnings.**

> "Solid" means the feature exists and its tests pass — not that the statistical
> output is trustworthy. Those are different claims and only one of them is now
> measured.
>
> **`cynepic-causal` is measured.** `ols_adjusted` achieves nominal coverage on
> all nine DGP cells (93.0%–97.7%, bias below 0.02). **`ipw` is now nominal on
> every estimable cell too** — it was 0.0% on every cell before the Tier 1
> rewrite, 87.3% under strong confounding after it, and 92.3% now, with
> `high-dim` going 89.2% → 94.3%. It refuses outright where overlap is
> insufficient, and refuses to cross-fit where the data cannot support it. The
> **ATT** estimator remains open at 91.0–92.4% (C14).
>
> Thirteen findings closed, two open specs, one document:
> **[docs/FINDINGS.md](causalrust/docs/FINDINGS.md)** is the single source of
> truth. Do not restate finding detail anywhere else; link to it.
>
> Every estimator returns `Result`, and every `ATEResult` carries the
> `Estimand` it computed (ATE / ATT / ATC / LATE), the `StdErrorKind` behind
> its interval, and `Diagnostics` from the fit. `ATEResult` has no public
> constructor, so a number cannot be separated from what it means.
>
> **`cynepic-bayes` is calibrated.** Credible-interval coverage is nominal for
> every conjugate prior, and both MCMC samplers pass simulation-based
> calibration. `BetaBinomial`'s interval was a normal approximation to a Beta
> that under-covered by 8.4 points at p=0.5; it now uses the exact quantile.
>
> ```bash
> ./scripts/findings-ratchet.sh    # the CI gate: count down only, all must fail
> cargo run -p cynepic-causal --example coverage_report --release
> cargo run -p cynepic-bayes  --example calibration_report --release
> cargo run -p cynepic-router --example classifier_report --release
> ```
>
> **`cynepic-router` is measured and improving.** The keyword classifier scores
> macro F1 0.290 against a 0.25 random baseline and misses **all 24**
> live-incident queries; 78% of natural phrasing matches no keyword at all.
> `LexicalClassifier` (tf-idf over unigrams and bigrams, nearest centroid) takes
> that to **0.656 macro F1 and 0.625 Chaotic recall under 4-fold
> cross-validation**, with 14% of queries producing no signal. R1's bar is 0.70
> and 0.80, so the finding stays open and the ratchet is unchanged — but most of
> the distance is covered without a model file, a runtime, or a dependency.
>
> **`cynepic-guardian` is property-tested.** Its circuit breaker had no real
> half-open state — after the reset timeout every caller was admitted, not one
> probe, so a still-down dependency got the whole backed-up load on a timer.
> Fixed with a three-valued state atomic and a compare-and-exchange probe claim.
>
> **`cynepic-graph` has no findings.** Fifteen execution properties hold:
> determinism, `max_steps` as a hard bound, checkpoint resume reproducing an
> uninterrupted run at every cut point and across a JSON round trip, budget
> charged against the original rather than reset on resume, and every started
> node reporting exactly one terminal event. A crate with no findings is a
> result — but only because the properties were written to be capable of
> failing, and four other crates written the same way did fail.

### Known Gaps

- **WASM is partial.** `wasm32-wasip1` works for core/causal/bayes and is gated
  in CI. `wasm32-unknown-unknown` (browser) builds for nothing yet — `uuid` and
  `rand 0.8` both need a `getrandom` JS backend. guardian/graph/router pull
  `tokio = { features = ["full"] }`, which targets no wasm at all.
- **Two findings remain open**, both quantified and ratchet-tracked:
  `att` interval coverage (C14, 91.0–92.4% against a 3-point bar) and the
  router classifier's reach (R1, macro F1 0.656 against a 0.70 bar). Neither
  blocks a release; both are documented with what would close them.
- **Per-call allocation counts are unmeasured.** They need a counting
  `GlobalAlloc`, which `forbid(unsafe_code)` correctly refuses, so it needs an
  external profiler run out-of-band. Peak RSS *is* measured. See
  [roadmap.md](causalrust/docs/roadmap.md#benchmarking-what-still-has-to-be-proven).
- **No `fuzz/` directory.** Nothing takes untrusted binary input, so the yield
  would be low, but it is absent rather than considered and rejected.
- **The Python bindings have no Python-level test.** The Rust side of the
  binding is covered (14 tests), which is where the defects have actually been
  — a circuit breaker with an empty body, a repr hardcoded to zero. What is not
  covered is the built wheel itself: `maturin build` output has never been
  imported and exercised in CI.
- **Do not carry over the Python project's numbers.** `projectcarfcynepic`'s
  43/43 benchmark results describe that implementation, not this one. See
  [research.md](causalrust/research.md).
- **Not published.** Crates are not on crates.io, so `cargo-semver-checks` has
  no baseline and is deliberately absent from CI until the first publish.

### Benchmarks — measured, and two assumptions were wrong

`benches/` (criterion, in causal and bayes) is built by CI but not timed —
shared-runner variance swamps the signal. The comparisons that *are* published
come from committed harnesses run on one machine:

| claim | assumed | measured |
|---|---|---|
| d-separation vs NetworkX | ~1,000x | **9–19x** |
| StateGraph step vs LangGraph | ~10x | **100–191x** |
| policy vs OPA sidecar | ~100x | **57x** (94.5% of it is the network hop) |
| policy vs OPA *engine* | — | **3.1x** |
| Beta credible interval vs scipy | — | **1.2x slower**, then 2.9x faster |

Wrong high, wrong low, and right for the wrong reason. Two rules survive:
correctness gates the claim, and MCMC is measured in **time per effective
sample**, never samples per second. `scripts/compare_{networkx,langgraph,opa}.py`
plus the `*_latency` examples reproduce all of it.

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
