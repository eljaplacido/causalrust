# cynepic-rs v0.3.0 — what was built, and what it is proven to do

**Status: merged to `main`, unpublished. 21 August 2026.**

Six Rust libraries for making decisions under uncertainty, plus three ways to
call them. Everything below that carries a number has an instrument behind it —
and most of the original numbers turned out to be wrong.

---

## The short version

The work took a codebase whose statistical claims were untested, built
instruments to test them, and then fixed what the instruments found. Thirteen
correctness defects were confirmed and closed. Two remain open, both measured
and documented. The library is ready to publish; two decisions before that are
yours.

| | |
|---|---|
| **Confirmed defects** | 13 closed, **2 open** — each open one quantified with what would close it |
| **Tests** | **404** passing (413 in the second build configuration), plus **22 Python tests against the built wheel**. Green on Linux, macOS and Windows |
| **The worst thing found** | One estimator's confidence intervals contained the true answer **0% of the time** while claiming 95% — and thirty unit tests passed throughout |
| **Ready to publish** | **Yes.** All six crate names are free, packaging is verified, the runbook is written. Nothing has been sent anywhere |

Shipping with known limits is normal. Shipping without naming them is not.

---

## What each component is for

Read the **Don't** line as carefully as the others. It is the part that most
software omits, and the part that decides whether you get burned.

### `cynepic-causal` — validated · 151 tests

**Use it for.** Measuring the effect of something you could not randomise. Did
the price change actually cause the churn, or did the customers who churned
differ to begin with? Works on observational data — rollouts, policy changes,
campaigns.

**The evidence.** Confidence intervals contain the true answer 93.0–97.7% of the
time across nine deliberately nasty simulated scenarios, against a 95% target.
Agrees with numpy, scipy and networkx across 258 cases — worst disagreement
5.5e-12. Given data that cannot support an answer, it **refused all seven times**
and invented nothing.

**Speed, against the tool you already have.** Faster than `statsmodels` on seven
of nine measured tasks — 1.8x at n=100k, 2.3x on like-for-like weighting, 3.4x on
a two-sample contrast. Slower on two, both in the table below.

**Don't.** Publish a confidence interval from the ATT estimator yet — see
concerns below. And treat a *narrow* interval under heavily skewed data as a
warning sign, not reassurance.

### `cynepic-bayes` — calibrated · 45 tests

**Use it for.** A running estimate of a rate that updates as evidence arrives,
and knows how sure it is. How reliable is this API? What is the true conversion
rate after 40 trials? Cheap enough to update per request.

**The evidence.** Credible intervals land at 94.7–95.7% against a 95% target
across eight configurations, including the hard cases — very few observations,
probabilities near 0 and 1. Both samplers pass simulation-based calibration, the
strictest available check.

**Don't.** Expect it to replace a full probabilistic programming language. It
covers conjugate models — the common shapes — exactly and fast, not arbitrary
models approximately.

### `cynepic-guardian` — property-tested · 44 tests

**Use it for.** Stopping an AI agent before it does damage. Policy rules, a
circuit breaker for failing dependencies, rate limits, loop detection, and an
append-only audit trail of every decision.

**The evidence.** Fast enough to sit on every call, which is the property that
decides whether a guardrail gets applied or quietly sampled: a breaker check is
16ns, a rate-limit check 112ns, a full policy evaluation 5.6µs — 57× faster than
the same policy through an OPA sidecar.

**Don't.** Assume an unconfigured policy chain is protecting you. An empty chain
approves everything, deliberately and visibly.

### `cynepic-graph` — no defects found · 25 tests

**Use it for.** Multi-step workflows that must survive interruption — an agent
pipeline, an approval chain, a batch job. Typed, async, with checkpoint and
resume.

**The evidence.** Fifteen execution properties hold, including the ones that
matter after a crash: resuming from a checkpoint reproduces an uninterrupted run
at every cut point, the step budget is a hard bound rather than a suggestion, and
every node that starts reports exactly one ending.

**Worth knowing.** This is the only component where the instruments found
*nothing*. That counts as a result only because the same style of testing broke
four other components.

### `cynepic-router` — the weakest link · 35 tests

**Use it for.** Judging how hard a question is, so simple ones go to a cheap
model and hard ones to an expensive one. The intended payoff is LLM cost
reduction.

**The evidence.** Accuracy score 0.656 against 0.25 for random guessing — a real
improvement on the 0.290 it replaced, and **not accurate enough to route without
supervision.** It catches 62% of urgent "production is on fire" queries, up from
**zero**.

**Don't.** Put it in front of a cost-sensitive pipeline unsupervised, and don't
quote a savings figure — none has been measured. What makes it safe to deploy is
that it *abstains*: input it cannot read returns "unknown" at exactly zero
confidence, which an escalation rule can trigger on.

### `cynepic-core` — foundation · 24 tests

**Use it for.** Shared vocabulary — the domain types, the decision types, and the
special mathematical functions the statistics rest on. You get it automatically
with any of the others.

**The evidence.** Its quantile functions agree with scipy to 5.5e-12 across 116
cases, using independently derived algorithms — which is what makes the agreement
evidence rather than coincidence.

### Three ways to call them

An **HTTP server** (13 tests) exposing classification, estimation, belief updates
and policy checks. An **MCP tool server** (19 tests) so an AI assistant can use
these directly as tools — all eight tools verified callable, every call audited.
And **Python bindings** (14 Rust tests plus 22 run against the actual built
wheel), so a data team can use the fast paths without writing Rust.

Python can now estimate causal effects directly — `estimate_ate`,
`estimate_ate_weighted`, `estimate_att` — each returning a result that carries
its estimand, population, standard-error kind, interval and weighting
diagnostics. That was Rust-only until this round, which meant the flagship
capability was unreachable from where most analysts work.

[`docs/integration.md`](causalrust/docs/integration.md) is the guide, organised
by which surface you should pick, with every example executed against a running
server or a built wheel before being written down.

> All three of these surfaces had zero tests when this work started. Two of them
> had real defects — including a Python circuit breaker that could never trip.

---

## What measuring changed

This is the most useful section, because it shows what the testing was worth.
Every row is something believed to be true that measurement contradicted.

### Correctness — before and after

| What was believed | What it actually was | Now |
|---|---|---|
| Effect estimator worked; 30 tests passed | Intervals covered the truth **0% of the time** | nominal |
| Bayesian intervals were exact | Under-covered by **8.4 points** at the most common case | calibrated |
| Circuit breaker had a half-open recovery state | It didn't — a recovering service got the **entire backlog** at once | fixed |
| Graph independence check was sound | Returned "independent" for questions that have **no answer** | fixed |
| Python circuit breaker worked | Recorded nothing. **Could never trip.** | fixed |
| Drift detection was a router feature | **The module was never compiled in.** Advertised in the docs and the tool manifest; the file existed and was never declared, so the type was unreachable and its four tests had never run | fixed |
| The agent tool server implemented its manifest | `audit_trail` was advertised with a schema and returned **"Unknown tool"** | implemented |

### Speed against the tools you would otherwise use

| Task | Alternative | cynepic-rs | |
|---|---|---|---|
| Two-sample contrast, n=100k | statsmodels 875.6µs | **256.7µs** | 3.4x faster |
| OLS + robust SE, n=100k | statsmodels 16.05ms | **9.08ms** | 1.8x faster |
| OLS + robust SE, n=10k **p=25** | statsmodels **10.51ms** | 14.40ms | **1.37x slower** |
| Weighted estimate, like for like | statsmodels 361µs | **160µs** | 2.3x faster |
| Weighted estimate **as shipped** | statsmodels 1.86ms | 6.60ms | **3.6x slower** |
| Policy check vs sidecar | OPA 320µs | **5.6µs** | 57x faster |
| Workflow step | LangGraph 58µs | **0.57µs** | 102x faster |

**Both losses are deliberate and in the public table.** The weighted estimator
is slower because it fits its propensity model six times instead of once — that
is what made its confidence intervals correct. The like-for-like row is the same
estimator without that correction, and there it is faster; a caller who wants
the speed can have it, knowing what they trade. The OLS loss at 25 covariates is
a genuine one: LAPACK's blocked linear algebra beats ours and the gap widens
with the number of covariates. Closing it means a BLAS dependency, which would
cost the "no system libraries, runs anywhere" property. Recorded, not optimised
away.

### Earlier speed claims — assumed against measured

| Comparison | Claimed | Measured | Verdict |
|---|---|---|---|
| Graph analysis vs NetworkX | ~1,000× | 9–19× | **50× over** |
| Workflow step vs LangGraph | ~10× | 100–191× | **15× under** |
| Policy vs OPA sidecar | ~100× | 57× | right, wrong reason |
| Policy vs OPA's own engine | — | 3.1× | newly measured |
| Credible interval vs scipy | — | 2.9× | was **1.2× slower** |

Two things worth taking from that table.

First, the estimates were wrong in *both* directions — so they were not cautious
or optimistic, they were simply uncorrelated with reality, which is what
un-instrumented numbers look like in bulk.

Second, the OPA row was one number doing two jobs. **94.5% of the apparent
speedup was deleting a network hop**, not the policy engine being better.
Removing a sidecar is worth 57× and is a genuine operational win; the engine
itself is worth 3.1×. Quoting the combined figure as an engine comparison would
have overstated it thirty-fold.

Two further claims were **withdrawn rather than measured**, because no fair
version of them exists: the comparison against PyMC (which has no equivalent
operation) and against a Python circuit breaker (an atomic instruction against an
attribute lookup). They are kept in the docs, marked, so anyone who met the claim
elsewhere finds out here that it does not hold up.

---

## Remaining concerns

Ordered by how likely each is to cause you a problem. None blocks publishing; all
four affect how you should describe the project.

### 1. The router is not good enough to run unsupervised — *highest impact*

> **Impact.** At 0.656 accuracy it will misroute roughly a third of queries. The
> "cut your LLM bill" use case — probably the most commercially attractive one —
> cannot be claimed on this evidence, and a "saves 40–60%" line was removed from
> the pitch document because nothing measured it.

It is deployable behind a human or an escalation rule, because it reliably admits
when it does not know. It is not deployable as an autonomous cost optimiser.
Eight different configurations were tried and all landed between 0.579 and 0.661,
so this is a ceiling of the approach rather than a tuning gap. Closing it means an
embedding model — which adds a model file, a runtime and a dependency the design
currently avoids on purpose.

### 2. One estimator's intervals are still slightly too narrow — *medium*

> **Impact.** The ATT estimator's intervals contain the truth 91–92% of the time
> while claiming 95%. In practice: it will look more certain than it is, roughly
> one time in thirty.

Its sibling estimators are fine, so the practical guidance is to prefer those.
Closing this needs a genuinely different estimator (an outcome model layered on
top), which is new capability rather than a bug fix. It is documented, and the
spec that fails is checked on every commit so it cannot be quietly forgotten.

### 3. A small error bar can be a danger signal — *counterintuitive*

> **Impact.** This inverts normal statistical intuition, so anyone using the
> weighting estimators on skewed data needs to be told. In the worst test case,
> **100% of the intervals that missed the truth were the ones reporting
> below-average uncertainty.**

The reason is that the rare, heavily-weighted observations are what correct the
bias. A sample that happens not to contain them produces both a confident-looking
small error bar *and* an answer that is off. It is confident *because* it is
wrong. The library reports the diagnostics that reveal this; a user has to know to
read them.

### 4. Everything is validated against simulated data — *epistemic limit*

> **Impact.** The evidence is strong on data where the true answer is known by
> construction — which is the only way to measure whether an interval covers. It
> says nothing about messy real-world data with unmeasured confounders.

This is inherent, not an oversight: you cannot measure coverage without knowing
the truth. But it means the honest claim is "the mathematics is correct and the
uncertainty is honest", not "it will give you the right answer about your
business".

**If you have a real dataset with a known answer — a past experiment where you
also hold observational data — running it through would be the single most
valuable next piece of evidence.** It is the one thing simulated validation
structurally cannot give you.

### 5. Eight dependency updates are waiting, and one must not be taken yet

> **Impact.** None are security fixes — the advisory scan is clean. But
> **`rand` 0.9 → 0.10 would change every seeded random stream**, invalidating
> the recorded determinism values and requiring every coverage and calibration
> number in this report to be re-measured.

Sequence it as: **release 0.3.0 on the current pins, then take the updates, then
re-run the three measurement artifacts** and confirm the numbers still hold. The
reverse order means publishing figures nobody has re-checked.

`regorus` 0.5 → 0.11 is the other one to take deliberately rather than
automatically — it is the policy engine, and a jump of six minor versions could
change how a Rego policy evaluates. The CI action bumps and `criterion` are
routine and safe whenever.

### Smaller items

- **Nothing is published yet**, so there is no automated check that a future
  release doesn't break someone's code. That check needs a first published
  version to compare against.
- **Python has the estimators but not everything.** Front-door identification,
  instrumental variables, refutation tests and the MCMC samplers are Rust-only.
  Reachable over HTTP; named in the integration guide so it is not discovered
  later.
- **No zero-copy numpy path.** Inputs are Python lists, so a DataFrame column
  needs `.tolist()`. Not the bottleneck at the sizes this targets — the
  estimator still beats statsmodels on like-for-like work — but it would be at
  much larger scale.
- **Browser support is absent.** Server-side WebAssembly works for the three
  mathematical crates; running in a browser does not work for any of them.
- **No fuzz testing.** Low risk here — nothing parses untrusted binary input —
  but it is absent rather than considered and dismissed.
- **The quality regime needs maintaining.** The mechanism that keeps defect count
  honest only works if contributors keep using it; it is documented, but it is a
  habit, not an automation.

---

## Open questions I can't answer for you

These change what the project becomes, not how it is built. My reasoning is
included so you can disagree with it specifically.

### Who is this for?

The components serve two quite different audiences: Rust engineers embedding
decision logic in a service, and Python data teams wanting faster causal tools.
The current documentation tries to speak to both, which risks convincing neither.

*Why it matters now:* it determines whether the next work is more Python surface
area or more Rust ergonomics — and whether the README leads with speed or with
correctness.

### Publish now, or soak privately first?

Publishing claims the six names permanently and starts the clock on other people
depending on you. It cannot be undone — a version can be withdrawn from new use,
but never deleted.

*My read:* the names are all still free, which is the main argument for moving.
The argument against is support burden: once published, breaking changes cost
other people time, and there are still two open findings. A middle path is to
publish and label it clearly as pre-1.0, which the version number already does.

### How much support are you willing to commit to?

An open-source release with a security policy and a contributing guide implies
someone answers issues. Right now that is one person.

*Why it matters:* it is easier to set a modest expectation on day one — "best
effort, no SLA" — than to walk one back later. The documents currently do not say
either way.

### What do you intend to do about the trademarks?

The code is Apache-2.0, which grants no trademark rights, and the CARF/CYNEPIC
marks are reserved in the notice file. That is a deliberate and defensible split —
the code is free, the name is not.

*Worth deciding explicitly:* whether you actually intend to enforce that, and
whether the marks are registered anywhere. A reserved-but-unenforced mark tends to
become an unreserved one.

### Which of the two open findings is worth funding?

Both need real work. The embedding classifier would make the router commercially
usable but compromises the "no dependencies, runs anywhere" property. The
doubly-robust estimator would close the interval gap and is a well-understood
piece of statistics with no such tradeoff.

*My read:* the estimator is the cheaper and cleaner win; the classifier is the one
that unlocks a use case. They serve different goals, so the answer depends on the
audience question above.

---

## If you want to publish

Verified as far as it can be without your credentials. All six names were
confirmed available, packaging was tested, and the one confusing error you will
hit is documented below.

**1. Re-run the checks.** Roughly three minutes; everything should be green.

```bash
cd causalrust
cargo test --workspace --all-features
./scripts/findings-ratchet.sh
```

**2. Log in** with a token from your crates.io account settings.

```bash
cargo login
```

**3. Publish in this order.** It is not optional — each one must exist before the
next can resolve it.

```bash
cargo publish -p cynepic-core
cargo publish -p cynepic-guardian
cargo publish -p cynepic-router
cargo publish -p cynepic-causal
cargo publish -p cynepic-bayes
cargo publish -p cynepic-graph
```

**4. Tag the release.**

```bash
git tag -a v0.3.0 -m "cynepic-rs v0.3.0"
git push origin v0.3.0
```

### One error that looks fatal and isn't

Testing any crate other than the first will fail with `no matching package named
cynepic-core`. That is expected — the others look for it on the public registry,
where it does not exist *yet*. It resolves the moment you publish the first one,
which is why the order matters.

**Do not "fix" it** by removing the version numbers from the internal
dependencies. That turns them into wildcards, which the security tooling rejects
and which cannot be published at all.

---

Every figure in this report is reproducible from a seeded command in the
repository. The full defect ledger, including the hypotheses that measurement
disproved, is in [`causalrust/docs/FINDINGS.md`](causalrust/docs/FINDINGS.md).
The release runbook is in
[`causalrust/docs/RELEASING.md`](causalrust/docs/RELEASING.md).

*Status as of 21 August 2026 — v0.3.0 merged to `main`, unpublished.*
