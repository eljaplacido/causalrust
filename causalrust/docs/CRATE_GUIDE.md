# Crate-by-Crate Developer Guide

> What exists in each crate, what it exports, and how to extend it.

---

## cynepic-core

**Role:** Foundation types shared by all crates. No runtime side effects.

### Exports
- `CynefinDomain` — 5-variant enum (Clear, Complicated, Complex, Chaotic, Disorder) with `requires_human()`, `is_emergency()`, Display
- `AnalyticalEngine` — async trait with associated types (Input, Output, Error)
- `PolicyDecision` — Approve / Reject { reason } / Escalate { target, reason }
- `AuditEntry` — UUID + timestamp + action + engine + decision + metadata
- `CynepicError` — unified error enum

### Extension Points
- Add new error variants to `CynepicError` as new crates need them
- `AnalyticalEngine` is the trait all domain engines implement
- All types must remain `Serialize + Deserialize + Send + Sync`

### Tests: 8
Domain properties, serialization round-trips, policy decisions, audit trails.

---

## cynepic-guardian

**Role:** Policy enforcement, safety, governance. The "trust layer."

### Exports
- `PolicyEvaluator` trait — async evaluate(action, context) -> PolicyDecision
- `PolicyChain` — sequential evaluator list, short-circuits on reject
- `RegoPolicyEvaluator` — Rego evaluation via regorus (feature-gated: `rego`)
- `CircuitBreaker` — Closed -> Open (after N failures) -> HalfOpen (after timeout)
- `AuditTrail` — append-only, thread-safe (Arc<Mutex<Vec>>), JSON export
- `LoopDetector` — detects node overvisits and alternation thrashing
- `RiskAwareEvaluator` — Bayesian risk score -> approve/escalate/reject
- `RateLimiter` — token-bucket rate limiting per action/actor
- `EscalationManager` — HITL escalation lifecycle (pending/approved/rejected/timed-out)

### What's Next
| Feature | Effort | Priority |
|---------|--------|----------|
| Cedar policy engine | Low | P2 |
| Persistent audit (sqlx/sled) | Medium | P2 |
| OTel span export from audit trail | Medium | P3 |

### Tests: 19

---

## cynepic-causal

**Role:** Causal inference — the "Complicated" domain engine. Highest ecosystem impact.

### Exports
- `CausalDag` — petgraph wrapper with causal semantics
- `d_separated()` — Bayes-Ball d-separation test
- `BackdoorCriterion::find()` — identifies minimal adjustment sets
- `FrontDoorCriterion::find()` — front-door adjustment sets via mediators
- `LinearATEEstimator` — difference-in-means + OLS with covariate adjustment
- `PropensityScoreEstimator` — IPW estimation via logistic regression
- `IVEstimator` — two-stage least squares (2SLS)
- `ATEResult` — ATE + standard error
- `RefutationResult` — placebo, random common cause, subset validation, bootstrap

### What's Next
| Feature | Effort | Priority |
|---------|--------|----------|
| Polars DataFrame data ingestion | Medium | P1 |
| Sensitivity analysis (Rosenbaum bounds) | Medium | P2 |
| Propensity score matching (not just IPW) | Medium | P2 |
| PyO3 bindings (accelerate DoWhy) | Medium | P1 |

### Tests: 26

---

## cynepic-router

**Role:** Cynefin complexity classification and cost-aware routing.

### Exports
- `QueryClassifier` trait — async classify(query) -> ClassificationResult
- `KeywordClassifier` — bootstrap keyword-based implementation
- `CynefinRouter` — classifier + config -> RoutingDecision
- `RouterConfig` — routes map, confidence threshold, fallback domain
- `RouteTarget` — url + timeout + CostTier
- `BudgetTracker` — cost tracking with tier-based budget enforcement
- `ClassifierMetrics` — confusion matrix, precision/recall/F1, misrouting cost

### What's Next
| Feature | Effort | Priority |
|---------|--------|----------|
| Embedding classifier (candle + sentence transformers) | High | P2 |
| HNSW nearest-neighbor index | Medium | P2 |
| Confidence calibration (Platt scaling) | Medium | P3 |
| A/B routing for model comparison | Medium | P3 |
| Axum HTTP proxy endpoint | Medium | P1 |

### Tests: 13

---

## cynepic-bayes

**Role:** Bayesian inference — the "Complex" domain engine.

### Exports
- `BetaBinomial` — Beta(a, b) conjugate prior for binary outcomes
- `NormalNormal` — Normal conjugate with known precision
- `GammaPoisson` — Gamma(a, b) for count data
- `DirichletMultinomial` — Dirichlet for categorical data
- `BeliefState` — unified enum over all prior types
- `MetropolisHastings` — 1D Gaussian proposal MH sampler
- `AdaptiveMH` — self-tuning MH (Robbins-Monro acceptance targeting)
- `MultiDimMH` — multi-dimensional MH with diagonal Gaussian proposal
- `SamplerResult` — samples vec + acceptance rate
- `BeliefTracker` — streaming real-time belief updates
- `ToolBelief` — Beta-prior reliability tracking for tools/services
- `ToolBeliefSet` — multi-tool reliability monitoring

### What's Next
| Feature | Effort | Priority |
|---------|--------|----------|
| HMC via burn autodiff | High | P2 |
| NUTS (No U-Turn Sampler) | High | P2 |
| Gaussian Process prior | High | P3 |
| PyO3 bindings | Medium | P1 |

### Tests: 20

---

## cynepic-graph

**Role:** Typed workflow orchestration — the "agent graph" layer.

### Exports
- `StateGraph<S>` — generic state workflow with builder pattern
- `Node<S>` trait — async execute(S) -> Result<S, NodeError>
- `FnNode<S>` — closure-based node implementation
- `NodeId` — serializable string identifier
- Fixed edges + conditional edges (fn(&S) -> NodeId routing)
- Max-steps safety limit + per-node timeout
- Cycle detection before execution (DFS on fixed edges)
- `Checkpoint<S>` — serializable execution snapshot for pause/resume
- `GraphHook` trait + `EventCollector` + `TracingHook` — observability hooks
- `GraphError` — NoEntryNode, NodeNotFound, NodeFailed, MaxStepsExceeded, CycleDetected, NodeTimedOut

### What's Next
| Feature | Effort | Priority |
|---------|--------|----------|
| Parallel branch execution (fan-out/fan-in) | High | P2 |
| LLM nodes (async-openai integration) | Medium | P2 |
| Memory nodes (neo4rs for graph memory) | Medium | P3 |
| Checkpointing to persistent storage | Medium | P2 |

### Tests: 10
