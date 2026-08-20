# Completion Roadmap — cynepic-rs

## Current State (v0.2.0-dev)

**317 tests** across 10 crates. Five of six publishable crates have a validation
harness capable of failing; four of them did. PyO3, MCP and HTTP surfaces exist
and compile.

| Crate | Status | Evidence |
|---|---|---|
| `cynepic-causal` | Measured | OLS nominal on 9/9 coverage cells; IPW nominal on 8/8 estimable |
| `cynepic-bayes` | Calibrated | Exact conjugate intervals; both samplers pass simulation-based calibration |
| `cynepic-guardian` | Property-tested | 15 guardrail properties |
| `cynepic-graph` | Property-tested | 15 execution properties, no findings |
| `cynepic-router` | Measured, weak | macro F1 0.290 vs a 0.25 random baseline |
| `cynepic-core` | Types + `special` | Checked against closed forms |

**The roadmap is subordinate to [FINDINGS.md](FINDINGS.md).** Two findings are
open and both gate a 1.0:

- **C14** — IPW/ATT intervals under-cover under strong confounding (ATT 89.3%).
- **R1** — the keyword classifier has 0.000 recall on Chaotic; 78% of natural
  phrasing matches no keyword. A reach problem, not a tuning one.

Closing those matters more than anything below.

---

## Phase 1: Hardening & CI (v0.2.0) — COMPLETE

### 1.1 Build & CI
- [x] CI that actually runs. Every workflow from March to August failed in under
      10 seconds on an action reference that does not resolve
      (`dtolnay/rust-action`; it is `dtolnay/rust-toolchain`), so nothing had
      ever been built or tested. 11 jobs now, gated behind one `ci-ok` check
- [x] 3-OS x 2-toolchain test matrix (ubuntu, macOS, windows; stable and 1.85)
- [x] Lint: fmt, clippy on both feature configurations, rustdoc link checking
- [x] Feature powerset via `cargo hack`
- [x] MSRV verified, and the manifest checked against the workflow's pin
- [x] Supply chain: `cargo deny` over advisories, licences, bans, sources
- [x] Findings ratchet across five suites
- [x] `wasm32-wasip1` gated for core, causal, bayes
- [x] Coverage, miri, and package verification (LICENSE + NOTICE in every tarball)
- [x] `msvc_spectre_libs` no longer blocks Windows — the whole matrix is green
- [ ] `rust-toolchain.toml` pinning MSRV

### 1.2 Completed Implementations (was "stubs")
- [x] `cynepic-causal`: Full OLS with covariate adjustment
- [x] `cynepic-causal`: d-separation (Bayes-Ball)
- [x] `cynepic-causal`: Front-door criterion
- [x] `cynepic-causal`: Propensity score IPW estimation
- [x] `cynepic-causal`: Instrumental variable 2SLS estimation
- [x] `cynepic-causal`: Multiple refutation tests (placebo, random cause, subset, bootstrap)
- [x] `cynepic-graph`: Per-node timeout via `tokio::time::timeout`
- [x] `cynepic-graph`: Cycle detection before execution (DFS)
- [x] `cynepic-graph`: Checkpointing with serialize/deserialize
- [x] `cynepic-graph`: Event hooks (GraphHook trait + EventCollector)
- [x] `cynepic-bayes`: Multi-dimensional MH sampler
- [x] `cynepic-bayes`: Adaptive MH (Robbins-Monro acceptance targeting)
- [x] `cynepic-bayes`: Dirichlet-Multinomial conjugate prior
- [x] `cynepic-bayes`: BeliefTracker streaming updates
- [x] `cynepic-bayes`: ToolBelief / ToolBeliefSet reliability tracking
- [x] `cynepic-guardian`: LoopDetector (overvisit + alternation detection)
- [x] `cynepic-guardian`: RiskAwareEvaluator (Bayesian risk scoring)
- [x] `cynepic-guardian`: RateLimiter (token-bucket)
- [x] `cynepic-guardian`: EscalationManager (HITL lifecycle)
- [x] `cynepic-router`: BudgetTracker (cost-aware routing)
- [x] `cynepic-router`: ClassifierMetrics (confusion matrix, precision/recall/F1)
- [x] Test coverage: 99 tests across workspace

### 1.3 API Surface Polish
- [x] `Result`-returning API across `cynepic-causal`; no panics on caller input
- [x] `[workspace.lints]` enforcing the stated conventions, with remaining debt
      recorded as measured counts rather than omitted
- [x] Provenance on every estimate: `Estimand`, `StdErrorKind`, `Diagnostics`,
      with `ATEResult` unconstructible without them
- [ ] `missing_docs = "deny"` — 86 violations, tracked in `[workspace.lints]`
- [ ] Clear the panic-freedom debt set (`indexing_slicing` 104, `unwrap_used` 7,
      `expect_used` 4)
- [ ] `#[must_use]` on all `Result`-returning functions

---

## Phase 2: Integration Interfaces (v0.3.0) — IN PROGRESS

**Goal:** Each crate is usable from Python, HTTP, and MCP.

All three surfaces exist and compile. They were merged in a non-compiling state
— 32 errors, written against an API that did not exist — and were fixed once CI
could see them.

### 2.1 PyO3 Bindings (`bindings/pyo3/`)
- [x] `cynepic` module wrapping the core types, DAG, priors, circuit breaker
- [x] `extension-module` as an opt-in feature, so `cargo build --workspace`
      links on macOS; maturin turns it on
- [ ] Full surface: estimators returning `Result` with the estimand attached
- [ ] `maturin` release build, publish to PyPI as `cynepic`
- [ ] NumPy interop (ndarray <-> numpy, ideally zero-copy)

### 2.2 HTTP API (`crates/cynepic-server/`)
- [x] Axum server: classify, estimate, bayes update, policy evaluate
- [x] Estimation errors map to 422 rather than 500 — they describe the caller's
      data, not a service fault
- [x] Responses carry the estimand, standard-error kind, interval and diagnostics
- [ ] OpenAPI spec via `utoipa`
- [ ] Docker image
- [ ] Integration tests against a running server

### 2.3 MCP Tool Server (`bindings/mcp/`)
- [x] JSON-RPC 2.0 over stdio, with the version field actually validated
- [x] Tools: `classify_domain`, `estimate_ate`, `check_policy`, `update_belief`,
      `detect_loop`, `run_counterfactual`
- [ ] `monitor_drift` — currently returns `not_implemented` with what it would
      need. A monitor that always answers "no drift" is worse than an absent one
- [ ] Verified against a real MCP client

### 2.4 WASM Target
- [ ] `cynepic-core`, `cynepic-bayes`, `cynepic-causal` compilable to `wasm32-unknown-unknown`
- [ ] `wasm-pack` build + npm package

---

## Phase 3: Advanced Capabilities (v0.4.0)

### 3.1 cynepic-causal
- [ ] Polars DataFrame integration for data ingestion
- [ ] Sensitivity analysis (Rosenbaum bounds)
- [ ] Propensity score matching (not just IPW)

### 3.2 cynepic-bayes
- [ ] Hamiltonian Monte Carlo (HMC) via `burn` autodiff
- [ ] No U-Turn Sampler (NUTS)
- [ ] Gaussian Process prior

### 3.3 cynepic-router
- [ ] Embedding-based classifier via `candle` (sentence transformers)
- [ ] HNSW nearest-neighbor index
- [ ] Confidence calibration (Platt scaling)
- [ ] A/B routing for model comparison

### 3.4 cynepic-graph
- [ ] Parallel branch execution (fan-out/fan-in)
- [ ] LLM nodes (`async-openai` integration)
- [ ] Memory nodes (Neo4j via `neo4rs`)

### 3.5 cynepic-guardian
- [ ] Cedar policy engine support
- [ ] Persistent audit trail (PostgreSQL via `sqlx` or `sled` embedded)
- [ ] Audit trail export as OpenTelemetry spans

---

## Phase 4: Ecosystem & Publishing (v1.0.0)

### 4.1 Crates.io Publication
- [ ] Stabilize public API
- [ ] Publish all 6 crates + `cynepic` umbrella crate
- [ ] Semantic versioning enforcement

### 4.2 Examples
- [x] `examples/coverage_report`, `calibration_report`, `classifier_report` —
      the three measurement artifacts, published by CI on every run
- [ ] End-to-end demos per vertical (supply chain, clinical, financial risk;
      specced in [research.md](../research.md))

---

## <a id="benchmarking-what-still-has-to-be-proven"></a>Benchmarking — what still has to be proven

The README carries a table of **assumed** speedups over NetworkX, PyMC, OPA and
LangGraph. Nothing measures them. This section is what would have to exist
before any of them may be quoted.

### The ordering problem

Speed is the least important thing on this list, and it is the only thing the
table currently claims.

For causal inference the product is **a number someone will act on**. A
treatment effect that arrives in 10µs and is wrong is worth less than one that
takes 10ms and is right — and worth much less than one that takes 10ms and
*declines to answer* because the data cannot support it. The same holds for a
policy verdict and for a routing decision.

So the benchmark suite is ordered by what a wrong answer costs, not by what is
easy to measure:

| # | Claim | Instrument | Status |
|---|---|---|---|
| 1 | **Agreement** — same answer as a reference implementation on the same data | Golden-file parity against numpy/scipy/networkx | **Built**, 258 cases |
| 2 | **Coverage** — intervals deliver the confidence they claim | `cynepic-testkit` harness | **Built**, published per run |
| 3 | **Refusal** — declines when the data cannot support an answer | Adversarial corpus, false-answer rate | **Built**, 0 false answers |
| 4 | **Reproducibility** — same seed, same answer, every platform | Determinism suites + CI's 3-OS matrix | **Built**, 15 checks |
| 5 | **Tail latency** — p99, not mean | criterion distributions | **Not built** |
| 6 | **Footprint** — allocations and peak memory | counting allocator | **Not built** |
| 7 | **Throughput** — the table's claim | criterion vs a Python harness | **Not built** |

### 1. Agreement (the claim that matters most)

A cross-language speedup claim is only interesting if both sides compute the
same thing. Before any "1000x faster than NetworkX" is meaningful, we need
"identical to NetworkX", and today nothing checks that.

- [x] Golden-file fixtures generated by `scripts/generate_parity_fixtures.py`
      and committed, with the library versions that produced them recorded
- [x] `crates/cynepic-causal/tests/parity.rs` — 258 cases, all passing

| Quantity | Here | Reference | Cases | Worst disagreement |
|---|---|---|---|---|
| Least squares ATE | pivoted Householder QR | numpy SVD (`lstsq`) | 5 | < 1e-9 |
| Classical SE | closed form | numpy `sigma^2 (X'X)^-1` | 4 | < 1e-8 |
| Welch SE | closed form | recovered from `scipy.stats.ttest_ind` | 3 | < 1e-9 |
| Beta / Gamma / t quantiles | Lentz CF + bisection | scipy (Boost) | 116 | **5.5e-12** |
| D-separation | Bayes-Ball | `networkx.is_d_separator` | 134 | none |

The pairs use genuinely different algorithms, which is what makes agreement
evidence rather than tautology. The 134 d-separation verdicts include the
collider cases — conditioning on a collider *opens* a path, the direction an
implementation is most likely to get backwards and the one a unit test written
by the same author is least likely to catch.

No Python at test time: references are computed once and committed, so CI stays
dependency-free.

- [ ] Extend to IPW and 2SLS once a reference implementation is pinned
- [ ] Disagreement is a **finding**, not a benchmark result — it goes in
      FINDINGS.md with the measurement

### 3. Refusal

The property no Python equivalent has, and the one most worth advertising.

- [x] `tests/refusal.rs` and `examples/refusal_report` — an adversarial corpus of
      inputs that should be refused, plus well-posed inputs that must not be.
      Without the second half, an estimator that refuses everything would score
      perfectly.
- [x] **False-answer rate: 0** across `difference_in_means`, `ols_adjusted`,
      `ipw` and `att`. False refusals on well-posed data: 0 for `ols_adjusted`.
- [x] The comparison, measured rather than asserted. On the exactly-collinear
      design:

      numpy.linalg.lstsq  ->  ATE = -0.000562, rank 4 of 5,
                              smallest singular value 1.8e-15,
                              no error and no warning
      cynepic-causal      ->  RankDeficient { aliased: ["covariate[2]"] }

      `-0.000562` does not read as "undefined". It reads as "no effect", which
      is a finding somebody might publish. numpy is not wrong — a minimum-norm
      solution is the documented behaviour for an underdetermined system — but
      it is wrong for a causal estimate.

- [ ] Extend to a DoWhy/statsmodels comparison once those are pinned in the
      fixture generator

### 4. Reproducibility

"Reproducible" is usually said as though it were one property. It is two, and
quoting the strong one when only the weak one holds is how the claim goes bad.

**Replay** — same seed, same binary — must be *bit-identical*. Anything weaker
and a stored seed cannot reconstruct a reported figure, which is the whole point
of storing it.

**Portability** — same seed, different platform — is bit-identical only for the
operations IEEE-754 requires to be correctly rounded (`+ - * /`, `sqrt`). It is
not, and cannot be, for `exp`, `ln`, `lgamma` or `erf`: those are libm, and
glibc, macOS and MSVC each round them differently in the last ulp.

So the claims are split along the line the standard draws:

| Path | Uses | Claim |
|---|---|---|
| `difference_in_means`, `ols_adjusted` | `+ - * / sqrt` only | **bit-identical on every platform** |
| `ipw` (IRLS, so `exp`/`ln`) | libm | agreement to 1e-9, deviation reported |
| MCMC chains | libm on every proposal | summary within Monte Carlo error |

- [x] `cynepic-causal/tests/determinism.rs` — committed bit patterns for the
      arithmetic-only estimators, checked on ubuntu / macos / windows by the
      existing CI matrix. Inputs are integers over 4, so they are exactly
      representable and identical everywhere *by construction*; generating them
      with a seeded RNG would have been shorter and would have destroyed the
      point, because `rand`'s normal ziggurat calls `ln` in its tail.
- [x] `cynepic-bayes/tests/determinism.rs` — chain replay for all three
      samplers, including the adaptive one, where a single differing draw
      changes the proposal scale and so every draw after it.
- [x] **Interleaving tests.** Two seeded streams consumed *alternately*, not one
      after the other. A thread-local generator reproduces perfectly when a
      chain runs alone and leaks the moment a caller fits two models in one
      process — which no single-chain test can see.
- [x] **Controls.** An unseeded chain must differ from another unseeded chain,
      and two seeds must produce different data. Without these, a sampler that
      ignored its seed entirely would score perfectly on every replay test.

#### The check the suite could not make, found by mutation

Every test above compared two runs *inside one process*. Perturbing the stored
seed by the process id — the exact defect that makes "reproduce it from the
recorded seed" false — left the entire suite green.

`a_seed_replays_across_processes_not_only_within_one` closes it by re-executing
the test binary and comparing chain digests. It fails under that mutation; the
other seven do not.

The lesson generalises past this file: **a determinism test that never crosses a
process boundary is testing that a function is a function.** It is worth
recording because the suite looked complete before the mutation was tried.

- [ ] Extend cross-process replay to the causal estimators (they are pure
      functions of their inputs, so the gap is smaller, but "smaller" is not
      "measured")
- [ ] A recorded-seed corpus: fix seeds now, assert the same answers after
      dependency bumps. This is the check that catches a `rand` or `statrs`
      upgrade silently changing a published number.

### 5. Tail latency, not throughput

For an embedded decision layer the p99 is what a caller feels; the mean is what
a marketing table quotes.

- [x] `examples/latency_report` — per-call p50/p95/p99/max, not criterion's
      central estimate. criterion is the right tool for detecting *regressions*
      in throughput; it times batches, so a per-call tail is not recoverable
      from it. Hence a separate harness.
- [x] Timer overhead measured and printed. For the sub-microsecond rows it is a
      material fraction of the reading, and quoting them without it overstates
      their cost.
- [x] The CPU is printed with the table. A latency number without the machine is
      not reproducible.
- [x] Refusal paths timed alongside the happy path. If declining cost more than
      answering, a caller under load would be tempted to skip the check.
      A length mismatch is rejected at the timer floor; an empty arm is **not**
      free, because discovering the arm is empty costs a pass over the column.
- [ ] Latency under adversarial *numerics* — near-singular designs, extreme
      propensity scores — where the iterative paths do the most work
- [ ] Not run for time in CI, and should stay that way: shared runners have
      noisy neighbours whose variance exceeds the differences worth detecting

### 6. Footprint

The claim that "no GC pauses, embeddable in any service" rests on, and is
partly measured.

- [x] Peak RSS reported by `examples/latency_report`, read from
      `/proc/self/status`
- [x] Tails are tight — p99 within a few percent of p50 on every row — which is
      the evidence that these paths are not allocation-dominated, and that is
      the property that makes them safe in a request path
- [ ] **Per-call allocation counts — blocked, deliberately.** The direct way is
      a counting `GlobalAlloc`. The workspace sets `unsafe_code = "forbid"`, and
      `forbid` cannot be downgraded by an `#[allow]` at the use site — which is
      the whole reason for choosing it over `deny`. Weakening a real guarantee
      so a benchmark could print a nicer number is the wrong trade, so this
      needs an **external profiler** (`heaptrack`, or `valgrind --tool=massif`)
      run out-of-band and recorded here. It is not going to be smuggled into the
      example.
- [ ] Confirm no allocation in the circuit-breaker and rate-limiter hot paths
      (same blocker, same resolution)

### 7. Throughput — and the first measured cell, which was 50x off

Last, deliberately. One row of the README's table has now been measured, and it
is the most important result in this section:

| nodes | `networkx.is_d_separator` (p50) | `cynepic_causal::d_separated` (p50) | measured |
|---|---|---|---|
| 10 | 12.1µs | 0.62µs | **19x** |
| 100 | 49.3µs | 4.6µs | **11x** |
| 500 | 213.2µs | 24.2µs | **9x** |

Both sides on one machine (aarch64, 2026-08-20; networkx 3.6.1, Python 3.12.3),
p50 of 2000 calls, same chain graphs, same conditioning set.

The README assumed **~1,000x** against a "~10ms NetworkX baseline". Both halves
were wrong, and in the same direction — NetworkX is roughly *three orders of
magnitude* faster than the assumed baseline, so the ratio was inflated at both
ends. The real figure is **9–19x**.

Two things follow, and the second matters more than the first:

1. **The speedup falls as the graph grows.** 19x at 10 nodes, 9x at 500. That
   shape says most of the win is per-call Python interpreter overhead, not the
   algorithm. Extrapolating from the small-graph number to a production-sized
   graph would be wrong in the optimistic direction, which is the direction that
   gets noticed late.
2. **9–19x is a good result that nobody needed to exaggerate.** The assumed
   number was not a lie anyone told deliberately; it was a plausible figure that
   never met an instrument. That is exactly the failure mode `docs/FINDINGS.md`
   exists for, and it is why the remaining rows stay marked assumed.

#### The second measured cell, where we started out losing

Measuring `cynepic-bayes` produced a row that is not in the README's table at
all, and it went the wrong way:

| Operation | scipy | cynepic-bayes | |
|---|---|---|---|
| Beta 95% credible interval (a pair of quantiles) | 42.4µs | **52.1µs** | **1.2x slower** |

A Rust crate losing to a Python one — and `scipy.stats.beta.ppf` is Boost
underneath, so this was Rust losing to C++ with a Python wrapper on top, on the
operation that dominates the cost of using conjugate priors at all.

No test could see it. The answers agreed with scipy to 5.5e-12 the whole time;
correctness and cost are independent, and only one of them had an instrument.

The cause was a bisection loop running a flat **200 iterations**. Bisection
halves its bracket each step, so on `[0, 1]` it exhausts `f64` in about 60 — the
remaining ~140 each evaluated an incomplete beta and then could not move `lo` or
`hi`, because no float remained between them. Stopping when the midpoint stops
moving returns **bit-identical values** and does 3.6x less work:

| | before | after |
|---|---|---|
| Beta 95% credible interval | 52.1µs | **14.4µs** |
| versus scipy | 1.2x slower | **2.9x faster** |

Bit-identity is checked rather than argued: the 116 quantile parity cases from
item 1 pin the output against scipy, and they are what made the change safe to
land in one step. That is the payoff from having built item 1 first — a
correctness harness turned an optimisation from a risk into a mechanical edit.

The same fix applies to `gamma_quantile` and `t_quantile`, which shared the
loop; the t quantile is on the OLS confidence-interval path, so this is on the
hot path of the coverage harness too.

Still outstanding:

- [x] A committed Python harness pinning the versions compared against —
      `scripts/compare_networkx.py`, which prints its `networkx` and Python
      versions and refuses to be read as a cross-machine result
- [x] Same data, same task, both sides. Writing this is what surfaced the
      d-separation disjointness defect: NetworkX *raises* on overlapping query
      sets where we returned `Ok(true)`, so the two sides were not answering the
      same question until that was fixed.
- [x] **LangGraph (graph) — measured, and understated by an order of
      magnitude.** `scripts/compare_langgraph.py` against
      `crates/cynepic-graph/examples/latency_report`, one machine,
      langgraph 1.2.11:

      | nodes | langgraph/step | cynepic/step | speedup |
      |---|---|---|---|
      | 5 | 66.3µs | 550ns | **120x** |
      | 20 | 58.3µs | 572ns | **102x** |
      | 100 | 74.9µs | 392ns | **191x** |

      The README assumed ~10x. So of the two rows that were both well-posed
      and measurable, one was overstated by 50x and the other understated by
      ~15x. Two plausible assumptions, wrong in opposite directions — which is
      what un-instrumented figures look like in aggregate: not biased, just
      noise dressed as a claim. It is also why "our guess was conservative" is
      not a defence; the guesses were simply uncorrelated with the facts.

      The shapes differ as well. LangGraph's per-step cost is flat in `n`,
      ours falls — we amortise fixed setup over more steps, theirs is
      genuinely per-step. And LangGraph carries checkpointing, a channel-based
      reducer model and interrupt support through every step, so part of the
      gap is machinery rather than waste. A dispatch win is not a claim that
      one replaces the other.
- [ ] OPA (policy) — not measured; the binary is not installed here. Still
      marked assumed, and see the audit below for why the number would not mean
      what it appears to.
- [ ] PyMC (sampler) — not a fair comparison at all. See the audit below.
- [x] Publish the cells where we are slower. The first one found is above, and
      publishing it is what led to the fix.
- [ ] Publish the cells where we are *slower*. A table with no losses is
      advertising, and at 500 nodes the trend line is already pointing at one.
- [ ] Run on a quiet machine with the CPU recorded, never on a shared CI runner

#### Are the remaining rows even well-posed questions?

Before measuring the other three, each was checked for whether a fair comparison
exists. Two do not, and manufacturing a number for those would be worse than
leaving them unmeasured.

**Beta conjugate update vs PyMC — kept, marked, and not to be measured.** PyMC has no conjugate-update
primitive. Every candidate comparison measures different work: against
`pm.sample()`, an exact closed-form posterior is being compared to a thousands-
of-draws MCMC approximation of the same thing, which would yield an enormous and
meaningless ratio; against `scipy.stats.beta(a, b)`, the Python side performs no
update at all; against `a += s; b += f` in plain Python, the measurement is of
CPython's interpreter loop and PyMC is not involved. A conjugate update is two
additions and, as measured above, sits below the timer floor.

**Decision (2026-08-20): the row stays in the README, marked "not a fair
comparison".** Deleting it would erase the fact that it was ever claimed, and
that fact is the useful part — a reader who has seen the claim elsewhere needs
to find out here that it does not hold up. The credible-interval row is what to
quote instead: it does real work, the scipy comparison is fair, and it is
measured.

**Circuit breaker vs Python — kept and marked, same reasoning.** Fair in kind,
and a foregone conclusion: an atomic load against a Python attribute access. It
also measures the least interesting property of a guardrail. Whether the breaker
*opens when it should* is the question, and `tests/guardrails.rs` answers it —
including the half-open state that finding G1 showed did not exist.

**Policy evaluation vs OPA sidecar — well-posed, but it will not be measuring
what it appears to.** ~100x is plausible, and almost all of it is deleting a
network round trip, not regorus outperforming OPA's evaluator. Against OPA *as
a library* the gap would be far smaller. If measured, both configurations must
be reported, or the row credits the policy engine for a win that belongs to the
deployment topology.

**StateGraph step vs LangGraph — well-posed.** Same task, same graph shape, both
sides doing dispatch. This is the one of the four worth measuring as written,
and it needs `langgraph` pinned in a committed harness alongside
`compare_networkx.py`.

Only when 1, 3 and 5 exist does 7 become a claim rather than a hope. For
d-separation they now do, and the claim is 9–19x. For everything else the README
says "assumed", which remains accurate.

### 4.3 Documentation Site
- [ ] mdBook or similar for user-facing docs
- [ ] Tutorials per crate

---

## Priority Matrix

| Item | Impact | Effort | Priority |
|------|--------|--------|----------|
| CI/CD pipeline | High | Low | **Done** |
| OLS covariate adjustment | High | Medium | **Done** |
| d-separation, front-door | High | Medium | **Done** |
| IPW + IV estimation | High | Medium | **Done** |
| MCMC variants (adaptive, multi-dim) | High | Medium | **Done** |
| Guardian safety (loops, HITL, rate limit) | High | Medium | **Done** |
| Router metrics + budget | Medium | Low | **Done** |
| Graph (timeout, cycles, hooks, checkpoint) | High | Medium | **Done** |
| PyO3 bindings (causal + bayes) | Very High | Medium | P1 |
| HTTP API server | High | Medium | P1 |
| MCP tool server | High | Medium | P1 |
| Embedding classifier | High | High | P2 |
| HMC/NUTS sampler | High | High | P2 |
| Cedar policies | Medium | Low | P2 |
| WASM target | Medium | Medium | P3 |
| Crates.io publish | High | Low | P3 (after API stable) |
