# How cynepic-rs Fits Into Real Workflows

> Concrete integration patterns for software development, agentic AI, and DevOps/MLOps pipelines.

---

## Part 1: Software Development Process

### The Core Insight

Most software teams already make decisions that are causal, probabilistic, and policy-governed — they just do it informally. cynepic-rs makes these decisions explicit, auditable, and fast enough to automate.

---

### 1.1 Feature Flag Decisions with Causal Inference

**The problem:** Your team ships a feature behind a flag. After 2 weeks, dashboards show revenue is up. PM says "ship it." But revenue also went up in the control group — because of a seasonal campaign nobody accounted for.

**How cynepic-causal changes this:**

```
              ┌─────────────────────┐
  Feature     │    cynepic-causal   │
  Rollout ──► │  Build causal DAG:  │
  Data        │  flag → revenue     │
              │  season → revenue   │
              │  season → flag      │  (users opt-in more in Q4)
              │                     │
              │  BackdoorCriterion  │
              │  → adjust: {season} │
              │                     │
              │  ATE = +2.1%        │
              │  SE  = ±0.8%        │
              │  Refute: PASS       │
              └─────────────────────┘
```

Instead of "revenue went up, ship it," the team gets: "The causal effect of this feature on revenue is +2.1% (±0.8%), controlling for seasonality, and the estimate survives placebo and subset refutation tests."

**Where it runs:** CI pipeline as a post-experiment analysis step. Data comes from the warehouse (Parquet via Polars), DAG comes from a config file checked into the repo, result is posted as a PR comment or Slack message.

**Integration surface:** Python (PyO3) for data science notebooks, HTTP API for dashboard backends, MCP tool for an agent that answers "should we ship this feature?"

---

### 1.2 Code Review Risk Assessment

**The problem:** A PR touches the payment processing path. How risky is it? Today this is a gut-feel judgment by the reviewer.

**How cynepic-router + cynepic-bayes change this:**

```
  PR Diff ──► cynepic-router
              │ Classify change complexity:
              │ - Touches 2 files, both in /payments → "Complicated"
              │ - Modifies error handling logic → confidence 0.78
              │
              ▼
              cynepic-bayes
              │ Prior: Beta(α=12, β=3) — this path has had
              │        12 successful deploys, 3 incidents
              │
              │ Historical incident rate for payment changes:
              │ posterior mean = 0.20 (20% incident probability)
              │ 95% CI = [0.08, 0.36]
              │
              ▼
              cynepic-guardian
              │ Policy: "payment path changes with incident
              │          probability > 0.15 require senior review"
              │ Decision: ESCALATE → senior-oncall
              │ AuditEntry logged
```

**Where it runs:** GitHub Actions workflow triggered on PR open/update. The bot comments with the risk assessment and tags the appropriate reviewer if escalation is needed.

**What the developer sees:**
```
🔍 Change Risk Assessment
Domain: Complicated (payment path, confidence: 0.78)
Historical incident rate: 20% [8%-36% CI]
Policy: Senior review required (threshold: 15%)
→ @senior-oncall tagged for review
Audit: entry-7f3a2b logged
```

---

### 1.3 Dependency Update Impact Analysis

**The problem:** Dependabot opens 15 PRs. Which ones are safe to merge and which need testing? Today: merge them all and hope, or manually review each one.

**How cynepic-causal + cynepic-guardian change this:**

```
  Dependency   ┌────────────────────────┐
  Update   ──► │    cynepic-causal      │
  PRs          │                        │
               │  Causal DAG:           │
               │  dep_update → build    │
               │  dep_update → tests    │
               │  dep_major → breaking  │
               │  breaking → incidents  │
               │                        │
               │  Identify: major bump  │
               │  with transitive deps  │
               │  → needs full test     │
               └──────────┬─────────────┘
                          ▼
               ┌────────────────────────┐
               │   cynepic-guardian     │
               │                       │
               │  Policy chain:        │
               │  1. patch → auto-merge │
               │  2. minor, tests pass  │
               │     → auto-merge       │
               │  3. major → human      │
               │     review required    │
               │                       │
               │  Circuit breaker:     │
               │  if 3+ dep updates    │
               │  fail in 24h → stop   │
               │  auto-merging         │
               └───────────────────────┘
```

**Where it runs:** GitHub Action that watches Dependabot PRs. cynepic-guardian policies are Rego files in the repo (`policies/dependency-updates.rego`). The circuit breaker prevents cascading failures from a bad batch of updates.

---

### 1.4 Incident Severity Classification

**The problem:** PagerDuty fires. Is this a P1 (all hands) or a P4 (can wait until morning)? Humans are bad at this at 3am.

**How cynepic-router + cynepic-bayes change this:**

```
  Alert ──► cynepic-router
            │ Classify:
            │ - error_rate spike + payment path → "Chaotic"
            │ - latency increase + non-critical path → "Complicated"
            │ - single user report → "Clear"
            │
            ▼
            cynepic-bayes
            │ Maintain per-service belief state:
            │ GammaPoisson prior for error counts
            │ Update with current window observations
            │ If posterior mean > 3σ above baseline → anomalous
            │
            ▼
            cynepic-guardian
            │ Policy:
            │ - Chaotic + anomalous → page oncall (P1)
            │ - Complicated + anomalous → Slack alert (P2)
            │ - Clear → auto-ticket, don't page
            │
            AuditEntry: every classification + routing decision logged
```

**Where it runs:** Sidecar alongside your alerting system (PagerDuty, OpsGenie, Grafana OnCall). HTTP API ingests alerts, returns severity + routing decision. Sub-millisecond — faster than the alert pipeline itself.

---

## Part 2: Agentic Workflows

### The Core Insight

AI agents today are LLM loops that generate text and call tools. They lack three capabilities that separate a useful demo from a production system: **causal reasoning** (what will happen if I do X?), **calibrated confidence** (how sure am I?), and **governance** (am I allowed to do this?). cynepic-rs provides all three as tools the agent can invoke.

---

### 2.1 The Reasoning Agent Pattern

An AI agent equipped with cynepic MCP tools follows a fundamentally different decision loop than a standard ReAct agent:

```
Standard ReAct Agent:
  Observe → Think (LLM) → Act → Observe → ...

cynepic-Enhanced Agent:
  Observe → Classify (router) → Reason (causal/bayes) → Validate (guardian) → Act → Audit → ...
```

The difference: between "Think" and "Act," three structured reasoning steps replace the LLM's ungrounded intuition with formal methods. The LLM still orchestrates — but the heavy analytical lifting happens in Rust at microsecond latency.

---

### 2.2 Agentic Data Analysis

**Scenario:** A product manager asks an AI agent: "Why did our signup rate drop last week?"

```
Agent receives question
│
├─► MCP: classify_query("why did signup rate drop")
│   → domain: Complicated, confidence: 0.91
│   → route: causal engine
│
├─► Agent fetches data from warehouse (tool call)
│   → signups, marketing_spend, site_changes, competitor_events
│
├─► MCP: build_causal_dag(
│     variables: [signups, marketing_spend, site_redesign,
│                 competitor_launch, seasonality],
│     edges: [[marketing_spend, signups],
│             [site_redesign, signups],
│             [competitor_launch, signups],
│             [seasonality, signups],
│             [seasonality, marketing_spend]],
│     treatment: "site_redesign",
│     outcome: "signups"
│   )
│   → adjustment_set: {seasonality, marketing_spend}
│
├─► MCP: estimate_treatment_effect(
│     treatment: [redesign exposure data],
│     outcome: [signup counts]
│   )
│   → ate: -12.3%, std_error: ±3.1%
│
├─► MCP: bayesian_update(
│     prior_type: "normal_normal",
│     prior_params: {mu: 0, sigma: 10},  // weak prior
│     observations: {mean: -12.3, se: 3.1, n: 5000}
│   )
│   → posterior: {mean: -12.2, ci_95: [-18.3, -6.1]}
│
└─► Agent synthesizes:
    "The site redesign caused a 12.2% drop in signups
     (95% credible interval: 6-18% drop), after controlling
     for seasonality and marketing spend changes.
     This is a causal estimate, not just a correlation.
     The effect survived placebo refutation."
```

**What's different from a standard agent:** The agent didn't just find a correlation in a dashboard. It built a causal model, identified confounders, estimated the treatment effect with proper adjustment, quantified uncertainty with Bayesian credible intervals, and can defend every step with formal methodology. The PM gets an answer they can bring to the exec meeting, not a "it looks like maybe the redesign had something to do with it."

---

### 2.3 Autonomous Deployment Agent

**Scenario:** A CI/CD agent decides whether to promote a build from staging to production.

```
Build passes tests
│
├─► cynepic-router: classify deployment risk
│   │ inputs: files changed, services affected, time of day, recent incidents
│   │ → domain: Complicated (multiple services, business hours)
│   │ → confidence: 0.83
│
├─► cynepic-causal: estimate deployment impact
│   │ Historical DAG: deploy → error_rate, deploy → latency,
│   │                 traffic_load → error_rate
│   │ BackdoorCriterion → adjust for: {traffic_load}
│   │ ATE on error_rate: +0.02% (within tolerance)
│   │ Refutation: PASS
│
├─► cynepic-bayes: deployment confidence
│   │ Prior: Beta(α=47, β=3) — 47 successful deploys, 3 rollbacks
│   │ Posterior success probability: 0.94 [0.86, 0.98]
│
├─► cynepic-guardian: policy evaluation
│   │ Policy chain:
│   │ ✅ "error_rate_impact < 0.1%" → PASS
│   │ ✅ "success_probability > 0.85" → PASS
│   │ ✅ "not_during_freeze_window" → PASS
│   │ ✅ "no_critical_alerts_active" → PASS
│   │ → PolicyDecision::Approve
│   │
│   │ AuditEntry: {
│   │   id: "7f3a...",
│   │   action: "promote-to-prod",
│   │   engine: "deployment-agent",
│   │   decision: "approve",
│   │   metadata: {ate: 0.02%, confidence: 0.94, policies_passed: 4}
│   │ }
│
├─► cynepic-graph: execute deployment workflow
│   │ StateGraph<DeployState>:
│   │   validate → canary(5%) → monitor(10min)
│   │             ├── healthy → rollout(25%) → monitor → rollout(100%)
│   │             └── degraded → rollback → alert-oncall
│   │
│   │ Conditional edge at "monitor":
│   │   inspect canary metrics → route to "healthy" or "degraded"
│
└─► Agent reports:
    "Deployed build #4521 to production.
     Causal impact estimate: +0.02% error rate (within 0.1% tolerance).
     Deployment confidence: 94% based on 50 prior deployments.
     4/4 policies passed. Canary healthy after 10min monitoring.
     Full audit trail: entry-7f3a..."
```

**Why this matters for DevOps:** Every deployment decision is now backed by causal evidence (not just "tests passed"), quantified confidence (not just "it worked before"), enforced policy (not just "someone eyeballed it"), and a complete audit trail (not just "it was deployed at 2pm"). When the postmortem happens, every decision is traceable.

---

### 2.4 Multi-Agent Research System

**Scenario:** A research team uses multiple specialized agents that collaborate through cynepic's shared reasoning infrastructure.

```
Research Question: "What drives customer churn in enterprise accounts?"

┌──────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                     │
│                  (cynepic-graph StateGraph)               │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │  Data     │    │ Analysis │    │ Policy   │           │
│  │  Agent    │    │ Agent    │    │ Agent    │           │
│  └─────┬────┘    └─────┬────┘    └─────┬────┘           │
│        │               │               │                 │
│        ▼               ▼               ▼                 │
│  Pulls data      Uses cynepic-    Checks cynepic-       │
│  from sources    causal to build  guardian: "is this     │
│  (warehouse,     churn DAG,       analysis approved      │
│  CRM, support    identify         for sharing with       │
│  tickets)        confounders,     client? PII scrubbed?  │
│                  estimate ATEs    data retention ok?"     │
│                  for each driver                          │
│                       │                                  │
│                       ▼                                  │
│                 Uses cynepic-bayes                        │
│                 to quantify confidence                    │
│                 per causal claim                          │
│                                                          │
│  Conditional routing via cynepic-router:                 │
│  - High confidence claims → include in report            │
│  - Low confidence → flag for human review                │
│  - Contradictory evidence → route to "Complex" domain    │
│    → deeper Bayesian analysis with wider priors           │
└──────────────────────────────────────────────────────────┘
```

Each agent speaks to cynepic through MCP tools. The orchestrator is itself a cynepic-graph `StateGraph<ResearchState>` with conditional edges that route based on confidence levels and policy decisions. The entire research process is auditable — every causal claim, every confidence interval, every policy check.

---

### 2.5 Coding Agent with Impact Awareness

**Scenario:** A coding agent (Claude Code, Cursor, Copilot) is asked to refactor a database query.

```
Developer: "Optimize the slow query in user_analytics.py"

Agent plans refactoring
│
├─► MCP: classify_query("optimize database query in analytics path")
│   → domain: Complicated
│   → rationale: deterministic optimization, but with downstream dependencies
│
├─► MCP: build_causal_dag(
│     variables: [query_change, response_time, cache_hit_rate,
│                 downstream_dashboard, report_accuracy],
│     edges: [[query_change, response_time],
│             [query_change, cache_hit_rate],
│             [cache_hit_rate, downstream_dashboard],
│             [query_change, report_accuracy]],
│     treatment: "query_change",
│     outcome: "report_accuracy"
│   )
│   → adjustment_set: {} (direct effect)
│   → WARNING: query_change has path to report_accuracy
│              — verify column semantics are preserved
│
├─► MCP: evaluate_policy(
│     action: "modify_analytics_query",
│     context: {
│       path: "user_analytics.py",
│       affects: ["dashboard", "weekly_report"],
│       estimated_rows_affected: 50000
│     },
│     policy: "package policy\n
│              default allow = false\n
│              allow { not input.affects[_] == \"weekly_report\" }\n
│              allow { input.estimated_rows_affected < 100000 }"
│   )
│   → REJECT: change affects weekly_report
│   → Agent adjusts: adds backward-compatible view, migrates in 2 steps
│
└─► Agent proceeds with safe refactoring plan:
    Step 1: Create optimized query as new view
    Step 2: Validate output matches existing query
    Step 3: Swap weekly_report to new view
    Step 4: Remove old query after validation period
```

**What changed:** The coding agent didn't just optimize the query. It understood the *causal downstream impact* (the query feeds a report), checked policy (report-affecting changes need extra care), and adjusted its plan accordingly. The agent went from "make it faster" to "make it faster without breaking the weekly report, in a reversible way."

---

## Part 3: DevOps & MLOps Pipelines

### The Core Insight

DevOps pipelines are decision graphs with policy gates. MLOps pipelines add statistical validation. cynepic-rs provides native primitives for both — and the composition between them.

---

### 3.1 The cynepic-Enhanced CI/CD Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline with cynepic                   │
│                                                                 │
│  ┌──────┐   ┌──────┐   ┌──────────┐   ┌─────────┐   ┌──────┐ │
│  │Build │──►│ Test │──►│ Validate  │──►│  Gate   │──►│Deploy│ │
│  └──────┘   └──────┘   └──────────┘   └─────────┘   └──────┘ │
│                              │               │           │      │
│                              ▼               ▼           ▼      │
│                        cynepic-causal  cynepic-     cynepic-    │
│                        cynepic-bayes   guardian     graph       │
│                                                                 │
│  VALIDATE STAGE:                                                │
│  ├─ causal: does this change preserve causal assumptions?       │
│  │  (DAG edges still valid after schema change?)                │
│  ├─ bayes: statistical validation of model performance          │
│  │  (posterior probability of regression > threshold?)          │
│  └─ causal: refutation tests on any ML model changes            │
│                                                                 │
│  GATE STAGE:                                                    │
│  ├─ guardian: policy chain evaluation                            │
│  │  ├─ "all tests pass" → check                                │
│  │  ├─ "no critical CVEs in deps" → check                      │
│  │  ├─ "performance regression < 5%" → check                   │
│  │  ├─ "deployment window open" → check                        │
│  │  └─ "human approval if touching payment path" → check       │
│  ├─ guardian: circuit breaker                                   │
│  │  └─ if 3+ failed deploys in 24h → block all deploys         │
│  └─ guardian: audit entry for every gate decision               │
│                                                                 │
│  DEPLOY STAGE:                                                  │
│  └─ graph: StateGraph<DeployState> orchestrates                 │
│     canary → monitor → progressive rollout → verify             │
└─────────────────────────────────────────────────────────────────┘
```

**GitHub Actions integration:**

```yaml
# .github/workflows/deploy.yml
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Causal validation: check DAG assumptions still hold
      - name: Validate causal assumptions
        run: |
          curl -X POST http://cynepic-server:8080/causal/validate \
            -d @causal_dag.json \
            -H "Content-Type: application/json"

      # Bayesian validation: model performance regression check
      - name: Statistical validation
        run: |
          curl -X POST http://cynepic-server:8080/bayes/regression-test \
            -d '{"prior": {"alpha": 50, "beta": 2},
                 "new_failures": 1, "new_successes": 48}'

  gate:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      # Policy gate: evaluate all deployment policies
      - name: Policy evaluation
        run: |
          curl -X POST http://cynepic-server:8080/guardian/evaluate \
            -d '{"action": "deploy",
                 "context": {
                   "branch": "${{ github.ref }}",
                   "tests_passed": true,
                   "files_changed": ${{ steps.changes.outputs.count }},
                   "touches_payment": ${{ steps.changes.outputs.payment }}
                 }}'
```

---

### 3.2 ML Model Validation Pipeline

**The problem no one talks about:** ML models go through training → evaluation → deployment. But evaluation is typically "accuracy > 0.95, ship it." Nobody asks *why* the model performs well, whether the performance is *caused* by the features or confounded by data leakage, or how *confident* we should be in the evaluation metrics given the test set size.

```
┌────────────────────────────────────────────────────────────────┐
│               ML Model Validation with cynepic                  │
│                                                                 │
│  Model Training Complete                                        │
│  │                                                              │
│  ├─► cynepic-causal: Feature Causality Audit                   │
│  │   │ Build DAG: features → target                            │
│  │   │ For each feature:                                       │
│  │   │   - Is the feature causally upstream of the target?     │
│  │   │   - Or is it a proxy for a confounder?                  │
│  │   │   - BackdoorCriterion: what needs adjustment?           │
│  │   │                                                          │
│  │   │ Refutation tests:                                       │
│  │   │   - Placebo treatment (random feature → should show     │
│  │   │     no effect)                                          │
│  │   │   - Data subset validation (effect stable across        │
│  │   │     subpopulations?)                                    │
│  │   │                                                          │
│  │   └─► Report: "Feature X is confounded with Z.             │
│  │        Remove or adjust before production."                  │
│  │                                                              │
│  ├─► cynepic-bayes: Metric Confidence                          │
│  │   │ Test set accuracy: 0.947                                │
│  │   │ Test set size: 200 samples                              │
│  │   │                                                          │
│  │   │ BetaBinomial prior: Beta(1, 1) (uninformative)         │
│  │   │ Update with: 189 correct, 11 wrong                      │
│  │   │ Posterior: Beta(190, 12)                                │
│  │   │ Mean: 0.941                                             │
│  │   │ 95% CI: [0.901, 0.970]                                 │
│  │   │                                                          │
│  │   └─► Report: "True accuracy is between 90.1% and 97.0%   │
│  │        with 95% confidence. The point estimate of 94.7%     │
│  │        is plausible but the lower bound (90.1%) is below    │
│  │        the 92% production threshold."                        │
│  │                                                              │
│  ├─► cynepic-guardian: Deployment Policy                       │
│  │   │ Policy chain:                                           │
│  │   │ 1. accuracy_lower_ci > 0.92? → FAIL (90.1% < 92%)     │
│  │   │    → REJECT: "confidence interval lower bound below     │
│  │   │      production threshold. Need more test data or       │
│  │   │      better model."                                     │
│  │   │                                                          │
│  │   │ Circuit breaker: 2 model rejections this week           │
│  │   │ → flag for team review of data pipeline                 │
│  │   │                                                          │
│  │   └─► AuditEntry: model_v2.3 rejected, reason: CI below    │
│  │        threshold, reviewer: automated                        │
│  │                                                              │
│  └─► cynepic-router: Route to appropriate next action          │
│      │ Model rejected → Complicated domain                     │
│      │ → route to: "retrain with larger test set" workflow     │
│      │ (not "deploy anyway" or "ask a human at 2am")           │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**The key difference from standard MLOps:** Standard pipelines check thresholds on point estimates. cynepic checks the *causal validity* of features, the *statistical confidence* of metrics (not just the point values), and enforces *policies on the confidence intervals* — not the point estimates. A model with 94.7% accuracy and 200 test samples is treated very differently from one with 94.7% accuracy and 50,000 test samples. Standard pipelines can't tell the difference. cynepic can.

---

### 3.3 Data Pipeline Integrity Monitoring

```
Scheduled data pipeline runs
│
├─► cynepic-bayes: Streaming anomaly detection
│   │ Per-table, per-column belief states:
│   │
│   │ users.signup_count:
│   │   GammaPoisson(α=120, β=4)  → expected ~30/day
│   │   Today: 3 signups
│   │   Posterior update → P(rate < 10) = 0.97
│   │   → ANOMALY: signup rate collapsed
│   │
│   │ orders.total_amount:
│   │   NormalNormal(μ=$47.20, σ²=$12.50)
│   │   Today mean: $46.80
│   │   → NORMAL: within expected range
│   │
│   │ events.row_count:
│   │   GammaPoisson(α=500, β=0.5) → expected ~1000/hr
│   │   This hour: 0
│   │   → ANOMALY: zero events (pipeline may be broken)
│
├─► cynepic-causal: Root cause analysis
│   │ DAG of pipeline dependencies:
│   │   auth_service → signup_events → users table
│   │   payment_gateway → order_events → orders table
│   │
│   │ Signup anomaly detected:
│   │   Trace causal path: signup_events → auth_service
│   │   Check: auth_service error rate spiked at same time
│   │   → Root cause: auth_service, not data pipeline
│
├─► cynepic-guardian: Alert routing policy
│   │ Policy:
│   │   - P(anomaly) > 0.95 AND revenue-affecting → P1 page
│   │   - P(anomaly) > 0.95 AND non-revenue → P2 Slack
│   │   - P(anomaly) > 0.80 → P3 ticket
│   │   - Auth service root cause → page SRE, not data team
│   │
│   │ Circuit breaker:
│   │   If > 5 anomalies in 1 hour → suppress individual alerts,
│   │   send single "systemic issue" alert
│   │
│   └─► AuditTrail: every anomaly detection, root cause analysis,
│        and alert routing decision logged with full context
│
└─► cynepic-graph: Remediation workflow
    │ StateGraph<PipelineState>:
    │   detect → classify → root_cause → remediate
    │   │
    │   Conditional edges:
    │   - auth_service root cause → restart auth pods
    │   - pipeline stale → trigger backfill
    │   - unknown cause → escalate to human
    │
    │ Each step: guardian policy check before action
    │ "Can this agent restart auth pods?" → evaluate policy → approve/deny
```

---

### 3.4 Infrastructure Cost Optimization

**How cynepic-causal transforms FinOps from correlation to causation:**

```
Monthly cloud spend analysis
│
├─► cynepic-causal: What CAUSED the cost increase?
│   │
│   │ Variables: compute_cost, storage_cost, traffic,
│   │            new_feature_A, team_size, data_retention_policy
│   │
│   │ DAG:
│   │   traffic → compute_cost
│   │   data_retention_policy → storage_cost
│   │   new_feature_A → traffic
│   │   new_feature_A → storage_cost (new cache layer)
│   │   team_size → compute_cost (more dev environments)
│   │
│   │ Question: "What's the causal effect of feature A on total cost?"
│   │ BackdoorCriterion → adjust for: {team_size}
│   │ ATE: feature A caused +$12,400/month in costs
│   │       (direct: +$3,200 compute, +$9,200 storage)
│   │
│   │ Refutation: effect stable across regions (PASS)
│   │
│   └─► "Feature A's cache layer is the primary cost driver.
│        The traffic increase is secondary. Optimize cache TTL
│        before scaling down compute."
│
├─► cynepic-bayes: Forecast with uncertainty
│   │ NormalNormal prior on monthly cost growth rate
│   │ 12 months of data → posterior
│   │ Next month forecast: $147K ± $8K (95% CI)
│   │ P(exceeding $160K budget) = 0.06
│
└─► cynepic-guardian: Budget policy enforcement
    │ "If P(exceeding budget) > 0.10 → freeze non-critical provisioning"
    │ Current: 0.06 → APPROVE (but close to threshold)
    │ → Recommend proactive optimization
```

---

### 3.5 Continuous Experimentation Platform

**The holy grail of data-driven organizations: every change is an experiment, every experiment has causal evidence, every decision has an audit trail.**

```
┌─────────────────────────────────────────────────────────────┐
│            Continuous Experimentation Platform                │
│                                                              │
│  Every feature flag, config change, and deployment           │
│  automatically becomes a causal experiment:                  │
│                                                              │
│  1. DESIGN (cynepic-causal)                                 │
│     - Auto-generate causal DAG from system topology          │
│     - Identify confounders that need controlling             │
│     - Recommend experiment duration based on effect size     │
│                                                              │
│  2. MONITOR (cynepic-bayes)                                 │
│     - Real-time posterior updates as data arrives            │
│     - Bayesian stopping: stop early when evidence is clear  │
│     - No peeking problem — Bayesian doesn't inflate α       │
│                                                              │
│  3. ANALYZE (cynepic-causal + cynepic-bayes)               │
│     - Estimate causal effect with proper adjustment          │
│     - Credible intervals (not just p-values)                │
│     - Automated refutation (placebo, subset, bootstrap)     │
│                                                              │
│  4. DECIDE (cynepic-guardian + cynepic-router)              │
│     - Policy: "ship if ATE > 0 with 95% probability"       │
│     - Route: Clear (auto-ship) / Complicated (human review) │
│     - Audit: complete decision trail for every experiment   │
│                                                              │
│  5. EXECUTE (cynepic-graph)                                 │
│     - Progressive rollout workflow                           │
│     - Automatic rollback if guardian circuit breaker trips   │
│     - Event emission to experiment tracking dashboard       │
│                                                              │
│  Integration: Runs as HTTP sidecar or embedded Rust library │
│  Connects to: feature flag system, data warehouse, alerting │
│  Latency: <5ms end-to-end for the decision loop             │
│  Output: every experiment decision → audit trail + OTel     │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 4: Integration Patterns Summary

### By Consumption Method

| Method | Best For | Latency | Setup |
|--------|---------|---------|-------|
| **Rust library** (`cargo add`) | Rust services needing embedded reasoning | μs | Cargo dependency |
| **Python extension** (`pip install cynepic`) | Data science notebooks, ML pipelines | μs (in-process) | pip install |
| **HTTP API** (cynepic-server) | Polyglot microservices, CI/CD pipelines | ~1ms (network) | Docker container |
| **MCP tools** (cynepic-mcp) | AI agents (Claude, GPT, local models) | ~1ms (stdio) | Binary + config |
| **WASM** (wasm-pack) | Browser dashboards, edge functions | μs (in-process) | npm package |
| **JNI** (jni-rs) | Spark UDFs, Flink processors | μs (in-process) | JAR + native lib |

### By Workflow Stage

| Stage | Primary Crate | Supporting Crates | What It Does |
|-------|--------------|-------------------|-------------|
| Classification | router | core | Determine complexity, route to right engine |
| Analysis | causal, bayes | core | Causal reasoning, uncertainty quantification |
| Validation | guardian | causal, bayes | Policy enforcement, compliance check |
| Orchestration | graph | all | Multi-step workflow execution |
| Audit | guardian | core | Append-only decision trail |

### By Persona

| Role | Primary Touch Point | Key Value |
|------|-------------------|-----------|
| **SWE** | MCP tools in IDE, CI/CD HTTP API | Impact-aware code changes, automated risk assessment |
| **Data Scientist** | Python bindings, notebooks | Fast causal inference, Bayesian A/B tests |
| **ML Engineer** | Python bindings, CI/CD pipeline | Model validation with causal + statistical rigor |
| **SRE / DevOps** | HTTP sidecar, alerting integration | Anomaly detection with calibrated confidence, policy-gated deployments |
| **Platform Engineer** | Rust library, Docker deployment | Embedded reasoning in platform services |
| **Product Manager** | Agent-mediated (MCP) | "Why did X happen?" answered with causal evidence |
| **Compliance / Legal** | Audit trail exports | Every automated decision traceable and policy-justified |
