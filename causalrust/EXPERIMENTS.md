# Experiments & Hands-On Guide

> Copy-paste these experiments into Rust test files or `main.rs` to understand what each crate does and when to use it.

Each experiment is a self-contained `#[tokio::test]` (or `#[test]`) you can add to a scratch project. Add the relevant `cynepic-*` crate as a dependency.

---

## Setup

Create a scratch project to run these experiments:

```bash
cargo new cynepic-lab
cd cynepic-lab
```

Add to `Cargo.toml`:

```toml
[dependencies]
cynepic-core = { path = "../causalrust/crates/cynepic-core" }
cynepic-guardian = { path = "../causalrust/crates/cynepic-guardian" }
cynepic-causal = { path = "../causalrust/crates/cynepic-causal" }
cynepic-router = { path = "../causalrust/crates/cynepic-router" }
cynepic-bayes = { path = "../causalrust/crates/cynepic-bayes" }
cynepic-graph = { path = "../causalrust/crates/cynepic-graph" }
tokio = { version = "1", features = ["full"] }
ndarray = "0.16"
serde_json = "1"
serde = { version = "1", features = ["derive"] }
async-trait = "0.1"
```

---

## Experiment 1: Causal DAG — Does X Cause Y?

**What you learn:** How to build a causal graph, check d-separation, find adjustment sets, and estimate treatment effects.

**When to use this:** Whenever you need to go beyond correlation — A/B test analysis, feature impact estimation, root cause analysis.

```rust
use cynepic_causal::{CausalDag, BackdoorCriterion, FrontDoorCriterion, d_separated, LinearATEEstimator};
use ndarray::array;
use std::collections::HashSet;

#[test]
fn experiment_causal_dag() {
    // === Scenario: Does a new UI redesign cause more signups? ===
    // Confounders: marketing_spend and seasonality both affect signups
    // AND correlate with when the redesign was deployed.

    let mut dag = CausalDag::new();
    dag.add_variable("redesign");        // treatment
    dag.add_variable("signups");         // outcome
    dag.add_variable("marketing_spend"); // confounder
    dag.add_variable("seasonality");     // confounder

    // Causal edges
    dag.add_edge("redesign", "signups");
    dag.add_edge("marketing_spend", "signups");
    dag.add_edge("marketing_spend", "redesign"); // marketing budget influenced deployment timing
    dag.add_edge("seasonality", "signups");
    dag.add_edge("seasonality", "marketing_spend");

    // Q1: Is the graph acyclic? (Must be for causal inference)
    assert!(dag.is_acyclic());

    // Q2: Are redesign and signups d-separated given {marketing_spend}?
    //     No — there's still the seasonality → signups path not blocked
    let conditioning: HashSet<String> = ["marketing_spend".into()].into();
    let dsep = d_separated(&dag, "redesign", "signups", &conditioning);
    println!("d-separated given {{marketing_spend}}? {}", dsep);

    // Q3: What variables must we condition on to identify the causal effect?
    let adjustment = BackdoorCriterion::find(&dag, "redesign", "signups");
    println!("Backdoor adjustment set: {:?}", adjustment);
    // Should include marketing_spend (and possibly seasonality)

    // Q4: Estimate the treatment effect
    // Simulated data: redesign group vs control
    let treatment = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let outcome   = array![120.0, 115.0, 130.0, 125.0, 118.0, 95.0, 100.0, 92.0, 88.0, 105.0];

    let ate = LinearATEEstimator::difference_in_means(&treatment, &outcome);
    println!("Average Treatment Effect: {:.2} (SE: {:.2})", ate.ate, ate.std_error);
    println!("Is this significant? ATE/SE = {:.2}", ate.ate / ate.std_error);

    // The ATE tells you: "The redesign caused ~X more signups on average"
    // The SE tells you how confident you are in that estimate
}
```

**Try modifying:**
- Add more variables and edges to see how adjustment sets change
- Try `FrontDoorCriterion::find()` by adding a mediator variable
- Compare `difference_in_means` vs `ols_adjusted` with covariates

---

## Experiment 2: Propensity Score & IV Estimation

**What you learn:** Advanced treatment effect estimation methods for observational data.

**When to use this:** When you can't randomize (observational studies), when you have instruments, or when simple difference-in-means is biased.

```rust
use cynepic_causal::{PropensityScoreEstimator, IVEstimator};
use ndarray::{array, Array2};

#[test]
fn experiment_propensity_scores() {
    // IPW estimation: treatment probability varies by covariates
    // Covariates: age, income (affect both treatment assignment AND outcome)
    let covariates = Array2::from_shape_vec((8, 2), vec![
        30.0, 50000.0,  // young, low income → less likely treated
        25.0, 45000.0,
        55.0, 90000.0,  // older, high income → more likely treated
        60.0, 95000.0,
        28.0, 48000.0,
        32.0, 52000.0,
        58.0, 88000.0,
        62.0, 92000.0,
    ]).unwrap();

    let treatment = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let outcome   = array![5.0, 4.0, 15.0, 16.0, 6.0, 5.0, 14.0, 17.0];

    let result = PropensityScoreEstimator::ipw(&treatment, &outcome, &covariates);
    println!("IPW ATE: {:.2} (SE: {:.2})", result.ate, result.std_error);
    // IPW adjusts for the fact that older/richer people were more likely to be treated
}

#[test]
fn experiment_instrumental_variables() {
    // IV/2SLS: When you have an instrument that affects treatment but not outcome directly
    // Instrument: distance to clinic (affects whether you get treatment, but doesn't
    //             directly affect health outcome)
    let instruments = Array2::from_shape_vec((8, 1), vec![
        1.0, 2.0, 0.5, 3.0, 1.5, 0.8, 2.5, 0.3,
    ]).unwrap();
    let treatment = array![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let outcome   = array![5.0, 4.0, 9.0, 3.0, 5.0, 8.0, 4.0, 10.0];

    let result = IVEstimator::two_stage_ls(&treatment, &outcome, &instruments);
    println!("IV (2SLS) ATE: {:.2} (SE: {:.2})", result.ate, result.std_error);
    // 2SLS uses the instrument to isolate exogenous variation in treatment
}
```

---

## Experiment 3: Refutation — Can You Trust Your Estimate?

**What you learn:** How to stress-test causal estimates with placebo, random cause, subset, and bootstrap tests.

**When to use this:** After every ATE estimation — refutation is what separates real causal claims from spurious findings.

```rust
use cynepic_causal::refute::{placebo_treatment, random_common_cause, subset_validation, bootstrap_refutation};
use cynepic_causal::LinearATEEstimator;
use ndarray::array;

#[test]
fn experiment_refutation() {
    // Real effect: treatment increases outcome by ~5 units
    let treatment = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let outcome   = array![10.0, 12.0, 11.0, 9.0, 13.0, 5.0, 6.0, 7.0, 4.0, 8.0];

    let original = LinearATEEstimator::difference_in_means(&treatment, &outcome);
    println!("Original ATE: {:.2}", original.ate);

    // Placebo test: randomly reassign treatment labels
    // If the effect survives, it's probably spurious
    let placebo = placebo_treatment(&outcome, original.ate, 1.0);
    println!("Placebo refutation passed: {}", placebo.passed);
    println!("Placebo effect: {:.4} (should be near 0)", placebo.refuted_effect);

    // Random common cause: add random confounders and re-estimate
    let random_cause = random_common_cause(&treatment, &outcome, original.ate, 50);
    println!("Random cause refutation passed: {}", random_cause.passed);
    println!("Effect change: {:.4}", (random_cause.refuted_effect - original.ate).abs());

    // Subset validation: estimate on random subsets
    let subset = subset_validation(&treatment, &outcome, original.ate, 0.7, 50);
    println!("Subset validation passed: {}", subset.passed);

    // Bootstrap: resample and check stability
    let bootstrap = bootstrap_refutation(&treatment, &outcome, original.ate, 200);
    println!("Bootstrap refutation passed: {}", bootstrap.passed);
    println!("Bootstrap mean effect: {:.2}", bootstrap.refuted_effect);
}
```

---

## Experiment 4: Bayesian Belief Updates — From Uncertainty to Confidence

**What you learn:** How conjugate priors update with evidence, and how credible intervals narrow with more data.

**When to use this:** A/B testing with early stopping, tool reliability tracking, anomaly detection, any decision under uncertainty.

```rust
use cynepic_bayes::priors::{BetaBinomial, NormalNormal, GammaPoisson, DirichletMultinomial};

#[test]
fn experiment_bayesian_ab_test() {
    // === A/B Test: Does variant B convert better? ===

    // Prior: uniform (no opinion) Beta(1,1)
    let mut variant_a = BetaBinomial::new(1.0, 1.0);
    let mut variant_b = BetaBinomial::new(1.0, 1.0);

    // Day 1: A gets 45/500 conversions, B gets 52/500
    variant_a.update(45, 455);
    variant_b.update(52, 448);

    println!("=== After Day 1 ===");
    println!("A: mean={:.4}, 95% CI={:?}", variant_a.mean(), variant_a.credible_interval_95());
    println!("B: mean={:.4}, 95% CI={:?}", variant_b.mean(), variant_b.credible_interval_95());
    // CIs overlap → not enough evidence yet

    // Day 2: More data arrives
    variant_a.update(42, 458);
    variant_b.update(58, 442);

    println!("\n=== After Day 2 ===");
    println!("A: mean={:.4}, 95% CI={:?}", variant_a.mean(), variant_a.credible_interval_95());
    println!("B: mean={:.4}, 95% CI={:?}", variant_b.mean(), variant_b.credible_interval_95());
    // CIs should be narrowing

    // Decision: P(B > A) can be approximated by comparing posteriors
    let b_better = variant_b.mean() > variant_a.mean();
    println!("\nB appears better: {}", b_better);
    println!("B conversion rate: {:.1}%", variant_b.mean() * 100.0);
    println!("A conversion rate: {:.1}%", variant_a.mean() * 100.0);
}

#[test]
fn experiment_anomaly_detection() {
    // === Server request rate anomaly detection ===
    // Prior: typical server handles ~100 requests/hour

    let mut rate_belief = GammaPoisson::new(100.0, 1.0); // alpha=100, beta=1 -> mean=100
    println!("Prior mean rate: {:.1} req/hr", rate_belief.mean());

    // Normal hours: observe 95, 102, 98 requests
    rate_belief.update(&[95, 102, 98]);
    println!("After normal hours: mean={:.1}, var={:.1}", rate_belief.mean(), rate_belief.variance());

    // Suddenly: 15 requests in an hour — anomaly?
    let mut anomaly_test = rate_belief.clone();
    anomaly_test.update(&[15]);
    println!("\nAfter anomalous hour (15 req):");
    println!("  Updated mean: {:.1}", anomaly_test.mean());
    println!("  This is far below the expected ~100 -> likely anomaly");
}

#[test]
fn experiment_categorical_beliefs() {
    // === Classify queries into Cynefin domains ===
    // Track which domain queries fall into with Dirichlet prior

    // Prior: uniform over 4 domains (equal probability)
    let mut domain_belief = DirichletMultinomial::uniform(4);

    // Observe: 20 Clear, 15 Complicated, 8 Complex, 2 Chaotic
    domain_belief.update(&[20, 15, 8, 2]);

    let probs = domain_belief.mean();
    println!("Domain probabilities:");
    println!("  Clear:       {:.1}%", probs[0] * 100.0);
    println!("  Complicated: {:.1}%", probs[1] * 100.0);
    println!("  Complex:     {:.1}%", probs[2] * 100.0);
    println!("  Chaotic:     {:.1}%", probs[3] * 100.0);

    // Most likely domain
    let mode = domain_belief.mode();
    println!("Mode (most likely): {:?}", mode);
}
```

---

## Experiment 5: MCMC Sampling — Fit Custom Models

**What you learn:** How Metropolis-Hastings explores posterior distributions, and how adaptive MH auto-tunes.

**When to use this:** When conjugate priors aren't flexible enough — custom likelihood functions, multi-parameter models.

```rust
use cynepic_bayes::sampler::{MetropolisHastings, AdaptiveMH, MultiDimMH};

#[test]
fn experiment_mcmc_standard_normal() {
    // Sample from a standard normal using MH
    let log_density = |x: f64| -0.5 * x * x; // log N(0,1)

    let sampler = MetropolisHastings::new(0.5, 1000, 5000); // proposal_std, warmup, n_samples
    let result = sampler.sample(log_density, 0.0); // log_density, initial

    let mean: f64 = result.samples.iter().sum::<f64>() / result.samples.len() as f64;
    let variance: f64 = result.samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / result.samples.len() as f64;

    println!("Target: N(0,1)");
    println!("Sample mean: {:.3} (expected: 0)", mean);
    println!("Sample variance: {:.3} (expected: 1)", variance);
    println!("Acceptance rate: {:.1}%", result.acceptance_rate * 100.0);
    // Good acceptance rate for 1D: 30-70%
}

#[test]
fn experiment_adaptive_mcmc() {
    // Adaptive MH auto-tunes proposal to target acceptance rate
    let log_density = |x: f64| {
        // Bimodal: mixture of N(-3, 0.5) and N(3, 0.5)
        let p1 = (-0.5 * ((x + 3.0) / 0.5_f64.sqrt()).powi(2)).exp();
        let p2 = (-0.5 * ((x - 3.0) / 0.5_f64.sqrt()).powi(2)).exp();
        (0.5 * p1 + 0.5 * p2).ln()
    };

    let sampler = AdaptiveMH::new(0.44, 2000, 10000); // target_acceptance, warmup, n_samples
    let result = sampler.sample(log_density, 0.0); // log_density, initial

    println!("Bimodal target: 0.5*N(-3,0.5) + 0.5*N(3,0.5)");
    println!("Acceptance rate: {:.1}% (target: 44%)", result.acceptance_rate * 100.0);
    println!("Sample size: {}", result.samples.len());

    // Check we found both modes
    let near_neg3 = result.samples.iter().filter(|&&x| (x + 3.0).abs() < 1.5).count();
    let near_pos3 = result.samples.iter().filter(|&&x| (x - 3.0).abs() < 1.5).count();
    println!("Samples near -3: {}, near +3: {}", near_neg3, near_pos3);
}

#[test]
fn experiment_multidim_mcmc() {
    // 2D target: bivariate normal N([0,0], I)
    let log_density = |x: &[f64]| -> f64 {
        -0.5 * (x[0] * x[0] + x[1] * x[1])
    };

    let sampler = MultiDimMH::new(vec![0.5, 0.5], 1000, 5000); // proposal_stds, warmup, n_samples
    let result = sampler.sample(log_density, vec![0.0, 0.0]); // log_density, initial

    let n = result.samples.len() as f64;
    let mean_x: f64 = result.samples.iter().map(|s| s[0]).sum::<f64>() / n;
    let mean_y: f64 = result.samples.iter().map(|s| s[1]).sum::<f64>() / n;

    println!("2D Normal target:");
    println!("Mean: ({:.3}, {:.3}) (expected: (0, 0))", mean_x, mean_y);
    println!("Acceptance rate: {:.1}%", result.acceptance_rate * 100.0);
}
```

---

## Experiment 6: Policy Guardrails — Protect Your System

**What you learn:** How to chain policies, use circuit breakers, detect loops, rate-limit actions, and manage HITL escalation.

**When to use this:** Any production system where automated decisions need governance, safety limits, and audit trails.

```rust
use cynepic_guardian::*;
use cynepic_core::{PolicyDecision, AuditEntry, EscalationTarget};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn experiment_policy_chain() {
    // === Chain multiple policies: all must approve ===

    // Policy 1: only allow admin actions
    struct AdminOnly;
    #[async_trait::async_trait]
    impl PolicyEvaluator for AdminOnly {
        async fn evaluate(&self, _action: &str, context: &serde_json::Value)
            -> Result<PolicyDecision, cynepic_guardian::policy::GuardianError>
        {
            if context.get("role").and_then(|r| r.as_str()) == Some("admin") {
                Ok(PolicyDecision::Approve)
            } else {
                Ok(PolicyDecision::Reject { reason: "Admin role required".into() })
            }
        }
        fn name(&self) -> &str { "admin_only" }
    }

    // Policy 2: block during maintenance windows
    struct MaintenanceCheck;
    #[async_trait::async_trait]
    impl PolicyEvaluator for MaintenanceCheck {
        async fn evaluate(&self, _action: &str, context: &serde_json::Value)
            -> Result<PolicyDecision, cynepic_guardian::policy::GuardianError>
        {
            if context.get("maintenance").and_then(|m| m.as_bool()) == Some(true) {
                Ok(PolicyDecision::Reject { reason: "System in maintenance".into() })
            } else {
                Ok(PolicyDecision::Approve)
            }
        }
        fn name(&self) -> &str { "maintenance_check" }
    }

    let chain = PolicyChain::new()
        .add(Arc::new(AdminOnly))
        .add(Arc::new(MaintenanceCheck));

    // Test: admin during normal operation -> approve
    let result = chain.evaluate("deploy", &json!({"role": "admin", "maintenance": false})).await.unwrap();
    assert!(matches!(result, PolicyDecision::Approve));

    // Test: admin during maintenance -> reject (short-circuits at policy 2)
    let result = chain.evaluate("deploy", &json!({"role": "admin", "maintenance": true})).await.unwrap();
    assert!(matches!(result, PolicyDecision::Reject { .. }));

    // Test: non-admin -> reject (short-circuits at policy 1)
    let result = chain.evaluate("deploy", &json!({"role": "viewer", "maintenance": false})).await.unwrap();
    assert!(matches!(result, PolicyDecision::Reject { .. }));
}

#[test]
fn experiment_loop_detection() {
    // === Detect runaway loops in agent execution ===
    let mut detector = LoopDetector::new(
        5,    // max visits per node before flagging
        3,    // alternation threshold (A->B->A->B... detected after 3 cycles)
    );

    // Normal execution
    assert!(detector.record_visit("fetch_data").is_none());
    assert!(detector.record_visit("process").is_none());
    assert!(detector.record_visit("validate").is_none());

    // Repeated visits to same node -> overvisit detection
    for _ in 0..4 {
        let _ = detector.record_visit("retry_api");
    }
    let violation = detector.record_visit("retry_api");
    println!("Loop violation: {:?}", violation);
    assert!(matches!(violation, Some(LoopViolation::NodeOvervisited { .. })));
}

#[test]
fn experiment_hitl_escalation() {
    // === Human-in-the-loop escalation workflow ===
    let mut mgr = EscalationManager::new(Duration::from_secs(300)); // 5 min timeout

    // Create an escalation request
    let event = mgr.create_escalation(
        "high_risk_deployment".into(),
        "Risk score exceeds threshold".into(),
        EscalationTarget::Role { name: "senior_oncall".into() },
        json!({"service": "payments", "risk_score": 0.87}),
    );
    let event_id = event.id;
    println!("Escalation created: {}", event_id);
    assert!(matches!(event.status, EscalationStatus::Pending));

    // Check pending escalations
    let pending = mgr.pending();
    assert_eq!(pending.len(), 1);

    // Human approves
    let approved = mgr.approve(&event_id, "jane@example.com".into()).unwrap();
    assert!(matches!(approved.status, EscalationStatus::Approved { .. }));
}

#[test]
fn experiment_audit_trail() {
    // === Append-only audit trail ===
    let trail = AuditTrail::new();

    trail.record(AuditEntry::new("deploy_v2.1", "deployment_agent", PolicyDecision::Approve));
    trail.record(AuditEntry::new(
        "modify_database",
        "migration_agent",
        PolicyDecision::Reject { reason: "Schema change during freeze".into() },
    ));

    // Export as JSON
    let json = trail.to_json().unwrap();
    println!("Audit trail:\n{}", json);

    // Query entries
    assert_eq!(trail.len(), 2);
    let entries = trail.entries();
    assert_eq!(entries[0].action, "deploy_v2.1");
    assert_eq!(entries[1].action, "modify_database");
}
```

---

## Experiment 7: Cynefin Router — Classify and Route

**What you learn:** How queries are classified into Cynefin domains and routed to appropriate handlers with cost awareness.

**When to use this:** MCP tool selection, model routing (cheap vs expensive), incident severity classification.

```rust
use cynepic_router::*;
use cynepic_router::config::{RouterConfig, RouteTarget, CostTier};
use cynepic_router::classifier::KeywordClassifier;
use cynepic_core::CynefinDomain;
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::test]
async fn experiment_routing() {
    // Build route configuration
    let mut routes = HashMap::new();
    routes.insert(CynefinDomain::Clear, RouteTarget {
        url: "http://cache/lookup".into(),
        timeout_ms: 100,
        cost_tier: CostTier::Free,
    });
    routes.insert(CynefinDomain::Complicated, RouteTarget {
        url: "http://causal-engine/analyze".into(),
        timeout_ms: 5000,
        cost_tier: CostTier::Medium,
    });
    routes.insert(CynefinDomain::Complex, RouteTarget {
        url: "http://bayes-engine/update".into(),
        timeout_ms: 3000,
        cost_tier: CostTier::Low,
    });
    routes.insert(CynefinDomain::Chaotic, RouteTarget {
        url: "http://emergency/respond".into(),
        timeout_ms: 500,
        cost_tier: CostTier::High,
    });

    let config = RouterConfig {
        routes,
        confidence_threshold: 0.5,
        fallback_domain: CynefinDomain::Disorder,
    };

    let classifier = KeywordClassifier::default_patterns();
    let router = CynefinRouter::new(Arc::new(classifier), config);

    // Route different queries
    let queries = vec![
        "What caused the revenue drop last quarter?",           // -> Complicated (causal)
        "System is down, customers can't pay!",                 // -> Chaotic (emergency)
        "What is the probability of churn for this segment?",   // -> Complex (probabilistic)
        "Look up user ID 12345",                                // -> Clear (deterministic)
    ];

    for query in queries {
        let decision = router.route(query).await.unwrap();
        println!("'{}'\n  -> {:?} (confidence: {:.2})\n  -> {:?}\n",
            &query[..50.min(query.len())],
            decision.classification.domain,
            decision.classification.confidence,
            decision.target.map(|t| t.url),
        );
    }
}

#[test]
fn experiment_budget_tracking() {
    let cost_map = CostMap {
        free: 0.0,
        low: 0.01,
        medium: 0.10,
        high: 1.00,
    };

    let mut tracker = BudgetTracker::new(100.0); // $100 budget

    // Record costs
    tracker.record(&CostTier::Free, &cost_map);
    tracker.record(&CostTier::Low, &cost_map);
    tracker.record(&CostTier::Medium, &cost_map);
    tracker.record(&CostTier::High, &cost_map);

    println!("Spent: ${:.2}", tracker.total_spent);
    println!("Remaining: ${:.2}", tracker.remaining());

    // Check budget decisions
    match tracker.check(&CostTier::High, &cost_map) {
        BudgetDecision::WithinBudget { remaining } => println!("Approved, ${:.2} left", remaining),
        BudgetDecision::OverBudget { overage, suggested_tier } =>
            println!("Over by ${:.2}, suggest {:?}", overage, suggested_tier),
    }
}

#[test]
fn experiment_classifier_metrics() {
    let mut metrics = ClassifierMetrics::new();

    // Record predictions vs ground truth (predicted, actual)
    metrics.record(CynefinDomain::Clear, CynefinDomain::Clear);       // correct
    metrics.record(CynefinDomain::Clear, CynefinDomain::Clear);       // correct
    metrics.record(CynefinDomain::Complicated, CynefinDomain::Complicated); // correct
    metrics.record(CynefinDomain::Complicated, CynefinDomain::Clear); // misclassified
    metrics.record(CynefinDomain::Complex, CynefinDomain::Complex);   // correct
    metrics.record(CynefinDomain::Chaotic, CynefinDomain::Chaotic);   // correct
    metrics.record(CynefinDomain::Clear, CynefinDomain::Complicated); // misclassified

    println!("Accuracy: {:.1}%", metrics.accuracy() * 100.0);
    for domain in CynefinDomain::all() {
        println!("{:?} — precision: {:.2}, recall: {:.2}, F1: {:.2}",
            domain,
            metrics.precision(domain),
            metrics.recall(domain),
            metrics.f1(domain),
        );
    }
}
```

---

## Experiment 8: Workflow Graph — Orchestrate Multi-Step Pipelines

**What you learn:** How to build typed workflow graphs with conditional routing, timeouts, checkpointing, and event hooks.

**When to use this:** Agent orchestration, CI/CD pipelines, multi-step data processing, any workflow that needs safety limits.

```rust
use cynepic_graph::{StateGraph, FnNode, NodeId, Checkpoint, EventCollector, GraphEvent};
use std::sync::Arc;
use serde::{Serialize, Deserialize};

// Define a typed workflow state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct PipelineState {
    data: Vec<f64>,
    validated: bool,
    result: Option<f64>,
    error: Option<String>,
}

#[tokio::test]
async fn experiment_pipeline_graph() {
    // === Data processing pipeline with conditional routing ===

    let validate = Arc::new(FnNode::new("validate", |mut s: PipelineState| async move {
        if s.data.is_empty() {
            s.error = Some("No data provided".into());
        } else {
            s.validated = true;
        }
        Ok(s)
    }));

    let compute_mean = Arc::new(FnNode::new("compute_mean", |mut s: PipelineState| async move {
        let sum: f64 = s.data.iter().sum();
        s.result = Some(sum / s.data.len() as f64);
        Ok(s)
    }));

    let handle_error = Arc::new(FnNode::new("handle_error", |mut s: PipelineState| async move {
        s.result = Some(f64::NAN);
        Ok(s)
    }));

    let collector = Arc::new(EventCollector::new());

    let graph = StateGraph::new()
        .add_node(validate)
        .add_node(compute_mean)
        .add_node(handle_error)
        .set_entry(NodeId::new("validate"))
        .add_conditional_edge(NodeId::new("validate"), |s: &PipelineState| {
            if s.validated {
                NodeId::new("compute_mean")
            } else {
                NodeId::new("handle_error")
            }
        })
        .add_hook(collector.clone());

    // Happy path
    let initial = PipelineState {
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        validated: false,
        result: None,
        error: None,
    };

    let result = graph.execute(initial, 10).await.unwrap();
    assert!(result.validated);
    assert_eq!(result.result, Some(3.0));

    // Check execution events
    let events = collector.events();
    println!("Execution trace:");
    for event in &events {
        match event {
            GraphEvent::NodeStarted { node, step } =>
                println!("  [{step}] Started: {node}"),
            GraphEvent::NodeCompleted { node, step, duration_ms } =>
                println!("  [{step}] Completed: {node} ({duration_ms}ms)"),
            GraphEvent::RouteDecision { from, to, .. } =>
                println!("  Route: {from} -> {to}"),
            GraphEvent::ExecutionCompleted { total_steps, total_ms } =>
                println!("  Done: {total_steps} steps in {total_ms}ms"),
            _ => {}
        }
    }

    // Error path
    let empty = PipelineState {
        data: vec![],
        validated: false,
        result: None,
        error: None,
    };
    let result = graph.execute(empty, 10).await.unwrap();
    assert!(!result.validated);
    assert!(result.result.unwrap().is_nan());
}

#[tokio::test]
async fn experiment_checkpoint_resume() {
    // === Pause and resume execution (for HITL workflows) ===

    let step1 = Arc::new(FnNode::new("step1", |x: i32| async move { Ok(x + 10) }));
    let step2 = Arc::new(FnNode::new("step2", |x: i32| async move { Ok(x * 2) }));

    let graph = StateGraph::new()
        .add_node(step1)
        .add_node(step2)
        .set_entry(NodeId::new("step1"))
        .add_edge(NodeId::new("step1"), NodeId::new("step2"));

    // Simulate: step1 ran (5 + 10 = 15), now we checkpoint before step2
    let checkpoint = Checkpoint::with_reason(
        15i32,
        NodeId::new("step2"),
        1,
        "awaiting_human_approval",
    );

    // Serialize checkpoint (could save to DB/file)
    let json = checkpoint.to_json().unwrap();
    println!("Checkpoint JSON:\n{}", json);

    // Later: human approves, resume from checkpoint
    let restored: Checkpoint<i32> = Checkpoint::from_json(&json).unwrap();
    let result = graph.resume(restored, 10).await.unwrap();
    assert_eq!(result, 30); // 15 * 2 = 30
}

#[tokio::test]
async fn experiment_timeout_safety() {
    // === Per-node timeout prevents hanging ===

    let fast = Arc::new(FnNode::new("fast", |x: i32| async move { Ok(x + 1) }));
    let slow = Arc::new(FnNode::new("slow", |x: i32| async move {
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        Ok(x)
    }));

    let graph = StateGraph::new()
        .add_node(fast)
        .add_node(slow)
        .set_entry(NodeId::new("fast"))
        .add_edge(NodeId::new("fast"), NodeId::new("slow"));

    let result = graph
        .execute_with_timeout(5, 10, std::time::Duration::from_millis(100))
        .await;

    match result {
        Err(cynepic_graph::GraphError::NodeTimedOut { node_id, timeout_ms }) => {
            println!("Node '{}' timed out after {}ms (as expected)", node_id, timeout_ms);
        }
        other => panic!("Expected timeout, got: {:?}", other),
    }
}
```

---

## Experiment 9: Tool Reliability Tracking

**What you learn:** How to track success/failure of external tools and services using Bayesian belief updates.

**When to use this:** Online learning, real-time monitoring, streaming anomaly detection, agent confidence tracking.

```rust
use cynepic_bayes::tool_belief::{ToolBelief, ToolBeliefSet};

#[test]
fn experiment_tool_reliability() {
    // Track reliability of external tools/APIs
    let mut tools = ToolBeliefSet::new();
    tools.add_tool("llm_api", None);     // default Beta(1,1) prior
    tools.add_tool("search_api", None);

    // Stream of observations
    tools.record_success("llm_api");
    tools.record_success("llm_api");
    tools.record_failure("llm_api");    // 2 successes, 1 failure

    tools.record_success("search_api");
    tools.record_success("search_api");
    tools.record_success("search_api"); // 3 successes, 0 failures

    // Query reliability beliefs
    let llm = tools.get("llm_api").unwrap();
    println!("LLM reliability: {:.3}", llm.reliability());
    println!("Should circuit-break? {}", llm.should_circuit_break(0.3));

    let search = tools.get("search_api").unwrap();
    println!("Search reliability: {:.3}", search.reliability());

    // Check confidence intervals
    println!("LLM CI:    {:?}", llm.confidence_interval());
    println!("Search CI: {:?}", search.confidence_interval());
}
```

---

## Experiment 10: Risk-Aware Decisions — Combining Bayes + Guardian

**What you learn:** How to use Bayesian risk scores to drive policy decisions.

**When to use this:** Any automated decision where the action's risk depends on current belief state.

```rust
use cynepic_guardian::RiskAwareEvaluator;
use cynepic_guardian::policy::PolicyEvaluator;
use cynepic_core::{PolicyDecision, EscalationTarget};
use serde_json::json;

#[tokio::test]
async fn experiment_risk_aware_policy() {
    // Risk thresholds: escalate if risk >= 0.3, reject if >= 0.7
    // The evaluator reads the risk score from a field in the JSON context
    let evaluator = RiskAwareEvaluator::new(
        "risk_score",                                       // field name in context JSON
        0.3,                                                 // escalate threshold
        0.7,                                                 // reject threshold
        EscalationTarget::Role { name: "oncall".into() },   // escalation target
    );

    // Low risk action
    let decision = evaluator.evaluate("deploy", &json!({"risk_score": 0.15})).await.unwrap();
    println!("Risk 0.15 -> {:?}", decision);
    assert!(matches!(decision, PolicyDecision::Approve));

    // Medium risk -> needs human approval
    let decision = evaluator.evaluate("deploy", &json!({"risk_score": 0.5})).await.unwrap();
    println!("Risk 0.50 -> {:?}", decision);
    assert!(matches!(decision, PolicyDecision::Escalate { .. }));

    // High risk -> blocked
    let decision = evaluator.evaluate("deploy", &json!({"risk_score": 0.85})).await.unwrap();
    println!("Risk 0.85 -> {:?}", decision);
    assert!(matches!(decision, PolicyDecision::Reject { .. }));
}
```

---

## What to Try Next

After running these experiments, try combining crates into end-to-end workflows:

1. **Causal A/B Test Pipeline**: Build a DAG -> identify confounders -> estimate ATE -> refute -> policy gate on CI
2. **Tool Reliability Monitor**: Track tool success/failure with `ToolBeliefSet` -> circuit break when reliability drops -> escalate to human
3. **Governed Agent Loop**: `StateGraph` orchestrating classify -> analyze -> validate -> act, with `PolicyChain` checking every transition
4. **Anomaly Response Workflow**: `GammaPoisson` streaming anomaly detection -> `CynefinRouter` severity classification -> `EscalationManager` HITL

Each crate is designed to compose with the others. The `cynepic-core` types (`PolicyDecision`, `CynefinDomain`, `AuditEntry`) are the shared vocabulary that ties everything together.
