# Crate-by-Crate Developer Guide

> Public API summary for each crate.

For full API documentation: `cargo doc --workspace --no-deps --open`

---

## cynepic-core

**Role:** Foundation types shared by all crates.

### Public API
- `CynefinDomain` — 5-variant enum (Clear, Complicated, Complex, Chaotic, Disorder)
- `AnalyticalEngine` — async trait for domain-specific analysis engines
- `PolicyDecision` — Approve / Reject / Escalate
- `AuditEntry` — UUID + timestamp + action + engine + decision + metadata
- `EpistemicState` — Unified session state with domain, confidence, reasoning chain, audit trail
- `ConfidenceLevel` — High / Medium / Low / Unknown (discretized from score)
- `ReasoningStep` — Engine + conclusion + confidence + evidence
- `CynepicError` — Shared error enum

---

## cynepic-guardian

**Role:** Policy enforcement, safety, governance.

### Public API
- `PolicyEvaluator` trait — async evaluate(action, context) → PolicyDecision
- `PolicyChain` — sequential evaluator list, short-circuits on reject
- `RegoPolicyEvaluator` — Rego evaluation (feature-gated: `rego`)
- `CircuitBreaker` — Closed → Open → HalfOpen state machine
- `AuditTrail` — append-only, thread-safe, JSON-exportable
- `LoopDetector` — detects node overvisits and alternation thrashing
- `RiskAwareEvaluator` — risk score → approve/escalate/reject
- `RateLimiter` — token-bucket rate limiting
- `EscalationManager` — HITL escalation lifecycle
- `BiasAuditor` — chi-squared fairness testing on decision distributions

---

## cynepic-causal

**Role:** Causal inference — the "Complicated" domain engine.

### Public API
- `CausalDag` — directed acyclic graph with causal semantics
- `d_separated()` — d-separation test (Bayes-Ball algorithm)
- `BackdoorCriterion` — identifies minimal adjustment sets
- `FrontDoorCriterion` — front-door adjustment via mediators
- `LinearATEEstimator` — difference-in-means and OLS with covariates
- `PropensityScoreEstimator` — IPW estimation
- `IVEstimator` — two-stage least squares (2SLS)
- `ATEResult` — ATE + standard error + sample size
- `RefutationResult` — placebo, random cause, subset, bootstrap tests
- `CounterfactualEngine` — Level-3 counterfactual queries
- `CounterfactualQuery` / `CounterfactualResult` — counterfactual I/O types

---

## cynepic-router

**Role:** Complexity classification and cost-aware routing.

### Public API
- `QueryClassifier` trait — async classify(query) → ClassificationResult
- `KeywordClassifier` — keyword-based bootstrap classifier
- `ClassificationResult` — domain + confidence + entropy + all scores
- `CynefinRouter` — classifier + config → routing decision
- `RoutingDecision` — classification + target + confident + budget status
- `BudgetTracker` — cost tracking with tier-based budget enforcement
- `DriftDetector` — KL-divergence distribution drift monitoring
- `DriftReport` — KL divergence + drift detected flag
- `ClassifierMetrics` — confusion matrix, precision/recall/F1

---

## cynepic-bayes

**Role:** Bayesian inference — the "Complex" domain engine.

### Public API
- `BetaBinomial` — conjugate prior for binary outcomes
- `NormalNormal` — conjugate for continuous data
- `GammaPoisson` — conjugate for count data
- `DirichletMultinomial` — conjugate for categorical data
- `BeliefState` — unified enum over prior types
- `MetropolisHastings` — 1D MH sampler
- `AdaptiveMH` — self-tuning MH (Robbins-Monro)
- `MultiDimMH` — multi-dimensional MH
- `BeliefTracker` — streaming real-time belief updates
- `ToolBelief` / `ToolBeliefSet` — tool reliability tracking

---

## cynepic-graph

**Role:** Typed workflow orchestration.

### Public API
- `StateGraph<S>` — generic state workflow with builder pattern
- `Node<S>` trait — async node execution
- `FnNode<S>` — closure-based node
- `NodeId` — serializable string identifier
- `Checkpoint<S>` — serializable execution snapshot for pause/resume
- `GraphHook` trait — observability hooks
- `EventCollector` / `TracingHook` — built-in hook implementations
- `GraphError` — comprehensive error enum
