# Integration guide

How to call cynepic-rs from the place you actually work, and why you would.

Every example here has been run. Where an API does not exist yet, this document
says so rather than showing you what it might look like.

---

## Which surface do you want?

| You are | Use | Because |
|---|---|---|
| Building an **AI agent** | [MCP tool server](#mcp--tools-for-an-ai-agent) | The agent calls these as tools and gets back reasons, not just answers |
| Running an **MLOps platform** | [HTTP API](#http--a-service-your-platform-calls) | Language-agnostic, one deployment, no per-service dependency |
| Doing **data science / analysis** | [Python](#python--for-analysis-and-notebooks) | `pip install`, then it is a function call |
| Writing **Rust** | [The crates directly](#rust--embedded-in-your-service) | No serialisation, no process boundary |

All four call the same code. Pick by where your work already lives.

---

## The one idea worth understanding first

Most of these components will **refuse to answer** rather than return a number
they cannot support. That is the point of them, and it changes how you integrate.

```python
>>> cynepic.estimate_ate([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
ValueError: treatment arm 'control' is empty (3 treated, 0 control);
            no contrast is defined
```

Given the same shape of input, `numpy.linalg.lstsq` returns `-0.000562` with no
error and no warning. That reads as *"no effect"*, which is a finding somebody
might act on. There is no effect to report — the data cannot identify one.

**So: handle the error path.** It is not an edge case, it is the feature. In the
HTTP API these arrive as `422` with a machine-readable `error` kind; in Python as
`ValueError`; in MCP as a JSON-RPC error.

---

## MCP — tools for an AI agent

A stdio JSON-RPC server. Point any MCP client at the binary.

```bash
cargo build --release -p cynepic-mcp
```

```json
{
  "mcpServers": {
    "cynepic": { "command": "/path/to/target/release/cynepic-mcp" }
  }
}
```

### The tools

| Tool | What an agent uses it for |
|---|---|
| `classify_domain` | "Is this question a lookup, an analysis, an experiment, or an emergency?" — and route accordingly |
| `estimate_ate` | "Did this change actually cause that outcome?" |
| `run_counterfactual` | "What would have happened to this specific unit instead?" |
| `check_policy` | "Am I allowed to do this?" — before acting, not after |
| `update_belief` | "How reliable has this tool been?" — with an honest interval |
| `detect_loop` | "Am I stuck repeating myself?" |
| `monitor_drift` | "Has my routing distribution shifted?" |
| `audit_trail` | "Show me every decision I made and why" |

### Why an agent benefits from these specifically

**Classification comes with its reasoning.** An agent that routes a query can
show why, and a reviewer can check whether it was sensible:

```json
{
  "domain": "Chaotic",
  "confidence": 0.66,
  "entropy": 0.71,
  "abstained": false,
  "why": ["now", "is", "corrupting", "right now"]
}
```

That query was *"the pipeline is corrupting rows right now"*. It contains none of
`emergency`, `crisis`, `outage` or `urgent` — a keyword matcher abstains on it,
and answering a live incident from cache is the exact failure this is for.

**Abstention is a first-class answer.** When the classifier cannot read a query
it returns `Disorder` at **exactly zero** confidence with high entropy. Escalate
on that. It is a designed contract, not an accident of scoring:

```json
{ "domain": "Disorder", "confidence": 0.0, "entropy": 1.0, "abstained": true }
```

**Every call is audited.** `audit_trail` returns what the agent did, including
the calls that failed — an audit trail holding only successes answers the wrong
question:

```json
{
  "total": 1,
  "returned": 1,
  "entries": [{
    "action": "classify_domain",
    "decision": "Approve",
    "engine": "cynepic-mcp",
    "id": "52ecf660-b145-4169-b382-bfbf3582a90a",
    "timestamp": "2026-08-21T19:23:50.405980636Z",
    "metadata": { "query": "the pipeline is corrupting rows right now" }
  }]
}
```

The trail lives for the process, which for a stdio server is one client session.
For durable audit, record the entries yourself — they are `Serialize`.

### A worked pattern: route, then act, then justify

1. `classify_domain` on the user's request.
2. If `abstained`, ask a human — do not guess.
3. If `Chaotic`, act to stabilise first and analyse afterwards.
4. `check_policy` before any action with consequences.
5. `audit_trail` at the end to explain what happened.

---

## HTTP — a service your platform calls

```bash
cargo run --release -p cynepic-server     # 127.0.0.1:4310, or set CYNEPIC_BIND
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness, and which version is deployed |
| `/router/classify` | POST | Query → domain, confidence, entropy |
| `/causal/estimate` | POST | Treatment effect with full provenance |
| `/bayes/update` | POST | Conjugate belief update with an exact interval |
| `/guardian/evaluate` | POST | Policy verdict |

### Estimating an effect

```bash
curl -s localhost:4310/causal/estimate -H 'content-type: application/json' -d '{
  "treatment":  [1, 0, 1, 0, 1, 0, 1, 0],
  "outcome":    [3.0, 1.0, 3.5, 1.2, 2.8, 0.9, 3.1, 1.1],
  "covariates": [[0.2],[0.1],[0.4],[0.3],[0.5],[0.2],[0.3],[0.1]],
  "method": "ols"
}'
```

`method` is one of `ols`, `ipw`, `att`. The response carries the provenance, not
just the number:

```json
{
  "ate": 2.038709677419354,
  "std_error": 0.1520066278642411,
  "confidence_interval": [1.7407821611538667, 2.336637193684841],
  "estimand": "ATE",
  "population": "the whole population",
  "std_error_kind": "Hc1",
  "n_obs": 8,
  "method": "ols",
  "diagnostics": {
    "rank": [3, 3],
    "arm_sizes": [4, 4],
    "effective_n": null,
    "propensity_range": null,
    "variance_dof": null,
    "converged": null
  }
}
```

`rank` is `[found, expected]` — equal means the design was full rank. The
`null`s are diagnostics that only a weighted estimator produces; switch
`"method"` to `"ipw"` and `effective_n`, `propensity_range`, `variance_dof` and
`converged` all fill in.

### Errors are part of the API

Data problems are **422**, not 500 — a 500 pages someone at 3am, a 422 tells the
caller what to fix. Each carries a stable `error` kind you can branch on:

```json
{ "error": "insufficient_overlap",
  "detail": "412 of 2000 units have propensity outside [0.02, 0.98]" }
```

Kinds include `length_mismatch`, `empty_arm`, `insufficient_data`,
`rank_deficient`, `not_converged`, `separation`, `insufficient_overlap`,
`weak_instrument`, `constant_treatment`, `ragged_covariates`, and
`unknown_method` (a **400** — the request is malformed, not the data in it).

### Where this fits in an MLOps platform

- **Before promoting a model**: `/causal/estimate` on the shadow-traffic outcome,
  adjusting for the covariates that differ between arms.
- **On every inference request**: `/guardian/evaluate` as an admission check.
- **Per model or per tool**: `/bayes/update` to track a reliability estimate that
  reports how sure it is, so a new model with three observations does not look
  identical to one with three thousand.
- **On a schedule**: watch `variance_dof` and `effective_n` in estimate responses.
  Falling numbers mean your populations are drifting apart.

---

## Python — for analysis and notebooks

```bash
pip install maturin
maturin build --release --features extension-module --manifest-path bindings/pyo3/Cargo.toml
pip install target/wheels/*.whl
```

### Estimating an effect

```python
import cynepic

effect = cynepic.estimate_ate(treatment, outcome, covariates)
print(effect)
# CausalEffect(ATE=1.9933, 95% CI [1.8991, 2.0875], n=2000)

effect.ate                  # 1.9933
effect.confidence_interval  # (1.8991, 2.0875)
effect.estimand             # 'ATE'
effect.population           # 'the whole population'
effect.std_error_kind       # 'Hc1'
effect.significant          # True — or None if no interval could be formed
```

Three estimators, answering three different questions:

| Call | Answers | Use when |
|---|---|---|
| `estimate_ate(t, y, x)` | Effect over everyone, adjusting for `x` | The outcome is plausibly linear in the covariates |
| `estimate_ate_weighted(t, y, x)` | Effect over everyone, by reweighting | *Assignment* is easier to model than the outcome |
| `estimate_att(t, y, x)` | Effect **on the treated** | "Did the campaign work?" — usually this one |

`estimate_ate` with no covariates is a plain difference in means.

### The two fields to read before trusting a weighted interval

```python
w = cynepic.estimate_ate_weighted(t, y, x)
w.effective_n      # 1695.4  — of 2000. Kish effective sample size
w.variance_dof     # 329.9   — effective degrees of freedom
w.propensity_range # (0.069, 0.923)
```

**`effective_n` far below `n_obs` means the estimate rests on a handful of rows**
however large your dataframe is. `variance_dof` in the single digits on a large
dataset means the same thing about the interval.

And the counterintuitive part, which is measured rather than folklore: under
heavy weighting, a **narrow** interval is the one to distrust. In the worst
validation cell, *every single interval that missed the truth was one reporting
below-average uncertainty*. The rare heavily-weighted rows are what remove the
bias; a sample that happens to miss them looks confident and is wrong.

### The rest of the Python surface

```python
dag = cynepic.CausalDag()
dag.add_edge("season", "sales")
dag.add_edge("season", "promo")
dag.add_edge("promo", "sales")
dag.find_backdoor_adjustment("promo", "sales")   # ['season']
dag.d_separated("promo", "sales", ["season"])    # False — a direct edge remains

belief = cynepic.BetaBinomial()
belief.update(successes=47, failures=3)
belief.mean            # a property, not a method

breaker = cynepic.CircuitBreaker(failure_threshold=5, recovery_timeout_secs=30)
breaker.record_failure()
breaker.is_open        # also a property

tools = cynepic.ToolBeliefSet()
tools.add_tool("search")
tools.record_failure("search")
tools.reliability("search")
tools.should_circuit_break("search", 0.8)
```

### What Python does *not* have yet

Front-door identification, instrumental variables, refutation tests and the
MCMC samplers are Rust-only for now. They are reachable over HTTP or by writing
a small Rust shim. This is a real gap, not an oversight to be discovered later.

### Interop with what you already use

There is **no zero-copy numpy path yet** — inputs are Python lists, so a
`DataFrame` needs `df["col"].tolist()`. For the array sizes this is aimed at
(thousands to hundreds of thousands of rows) the conversion is not the
bottleneck; the estimator still beats `statsmodels` on like-for-like work. If it
becomes your bottleneck, that is worth reporting as an issue with the shape of
your data.

```python
import polars as pl
df = pl.read_parquet("experiment.parquet")
effect = cynepic.estimate_ate(
    df["treated"].cast(pl.Float64).to_list(),
    df["revenue"].to_list(),
    df.select(["tenure", "plan_tier"]).rows(),
)
```

---

## Rust — embedded in your service

```toml
[dependencies]
cynepic-causal   = "0.3"
cynepic-guardian = "0.3"
```

```rust
use cynepic_causal::estimate::linear::LinearATEEstimator;

let result = LinearATEEstimator::ols_adjusted(&treatment, &outcome, &covariates)?;
println!("{} = {:.4} ± {:.4}", result.estimand().label(), result.ate(), result.std_error());
```

`ATEResult` has no public constructor. You cannot build one by hand, which means
a number in your system always arrived with its estimand, its standard-error
kind and its diagnostics attached.

The always-on guardrails are cheap enough to leave on: a circuit-breaker check is
**16ns**, a rate-limit check **112ns**, a full Rego policy evaluation **5.6µs**.
That matters because a guardrail applied to one call in ten is not a guardrail,
and the usual reason people sample them is cost.

---

## Performance, honestly

Measured on one machine against the tools you would otherwise use. Full table and
method in [roadmap.md](roadmap.md#benchmarking--what-still-has-to-be-proven).

| Task | Alternative | cynepic-rs | |
|---|---|---|---|
| OLS + robust SE, n=100k | statsmodels 16.05ms | **9.08ms** | 1.8x faster |
| OLS + robust SE, n=10k **p=25** | statsmodels **10.51ms** | 14.40ms | **1.37x slower** |
| IPW, like for like, n=1k | statsmodels 361µs | **160µs** | 2.3x faster |
| Policy eval vs sidecar | OPA 320µs | **5.6µs** | 57x faster |
| Policy eval vs engine | OPA 17.6µs | **5.6µs** | 3.1x faster |
| Workflow step | LangGraph 58µs | **0.57µs** | 102x faster |

Three things this table is careful about:

- **We lose at 25 covariates.** LAPACK's blocked QR beats ours and the gap widens
  with `p`. This crate suits *low-dimensional, high-volume* estimation — many
  small estimates rather than one wide one.
- **`ipw` as shipped is ~3.6x slower than statsmodels**, because it fits the
  propensity model out-of-fold — six fits instead of one — which is what made its
  intervals correct. The "like for like" row is the same estimator without that.
  You can have the speed via `fit_propensity` + `ipw_with_model`, and you are
  trading away the coverage fix to get it.
- **The policy row is two rows** because 94.5% of the sidecar figure is deleting
  an HTTP hop, not a better engine. Both are real; only one is about this code.

---

## What is not here

Named so you do not go looking:

- **No WASM in the browser.** Server-side `wasm32-wasip1` works for core, causal
  and bayes. Browser targets build for nothing yet.
- **No JVM bindings.** Use the HTTP API.
- **No Arrow or Polars backend.** Lists in, results out.
- **No streaming or incremental estimation.** Estimators take a complete dataset.
- **The router is not accurate enough to run unsupervised** — 0.656 macro F1
  cross-validated. Deploy it behind a human or an escalation rule. It abstains
  reliably, which is what makes that safe; it does not classify reliably enough
  to be left alone, and no cost-saving figure is claimed because none is measured.

---

## Where to look next

- [FINDINGS.md](FINDINGS.md) — every correctness defect found, with its
  measurement, including the two still open
- [CRATE_GUIDE.md](CRATE_GUIDE.md) — per-crate API tour
- [WORKFLOWS.md](WORKFLOWS.md) — longer end-to-end patterns
- [roadmap.md](roadmap.md) — what is measured, what is assumed, and what it would
  take to prove the rest
