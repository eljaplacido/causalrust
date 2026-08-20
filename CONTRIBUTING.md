# Contributing to cynepic-rs

Thank you for considering it. This document is short on ceremony and long on the
two or three things that are genuinely different about working here.

## The one idea

**A claim without a measurement is not a claim.**

This project computes numbers that people may act on — treatment effects,
credible intervals, routing decisions, policy verdicts. A wrong number that
looks reasonable is worse than a crash, because a crash is noticed. So the
standard here is not "the tests pass". It is "there is a measurement, it can
come back bad, and here is what it said".

Two consequences you will meet immediately:

- Statistical code needs a **coverage** or **calibration** measurement, not just
  a unit test with a tolerance. `docs/FINDINGS.md` exists because thirty passing
  unit tests coexisted with an estimator at 0.0% interval coverage.
- Performance claims need a benchmark that produced them. The figures in
  `causalrust/README.md` are marked as **assumed**; `docs/roadmap.md` breaks
  down what would have to exist to prove them. Do not add more, and do not quote
  the marked ones.
- Cross-implementation claims need a **parity** test first. "1000x faster than
  NetworkX" is not a claim until "identical to NetworkX" is true.

## Getting set up

```bash
git clone https://github.com/eljaplacido/causalrust
cd causalrust/causalrust
cargo test --workspace --all-features
```

Rust 1.85 or later (edition 2024). Nothing else is required.

Optional, and what CI runs:

```bash
cargo install cargo-deny cargo-hack cargo-llvm-cov
```

## The full local gate

CI runs nine jobs, plus Miri weekly in a separate workflow. This is all of them, in the order that fails fastest:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features    -- -D warnings
cargo clippy --workspace --all-targets --no-default-features -- -D warnings
cargo test --workspace --all-features
cargo test --workspace --no-default-features
cargo doc --workspace --no-deps --all-features           # broken doc links fail
./scripts/findings-ratchet.sh
cargo deny check
cargo hack check --workspace --feature-powerset --no-dev-deps
cargo check --target wasm32-wasip1 -p cynepic-core -p cynepic-causal -p cynepic-bayes
```

`ci-ok` is the single required status check, so adding a job does not mean
editing branch protection.

## The findings ratchet

This is the part most likely to surprise you.

Confirmed correctness defects live as **failing tests** — real specs that assert
the corrected behaviour, marked `#[ignore]` so the default suite stays green.
`scripts/findings-ratchet.sh` counts them and enforces three things:

1. The count may only go **down**.
2. The specs still **compile** against the current API. Otherwise an API change
   could orphan the lot while the count kept reporting a reassuring number about
   dead code.
3. Every open spec still **fails**.

That third check is the one that earns its keep. It has caught five specs that
had quietly started passing — some because the defect was fixed and nobody
lowered the count, some because the spec never witnessed its defect in the first
place. Both look identical to outstanding work until something checks.

### Fixing a finding

1. Fix it.
2. Delete its `#[ignore]` **in the same pull request**.
3. Lower `BASELINE` in `scripts/findings-ratchet.sh`.
4. Mark it closed in `docs/FINDINGS.md`.

A fix that arrives without deleting its `#[ignore]` has not been demonstrated,
and a reviewer should ask why.

### Reporting a new finding

The count is allowed to go up — but deliberately. Add the finding to
`docs/FINDINGS.md` with its failure mode and measurement, add a failing spec,
and raise `BASELINE` in the same commit. Use the **Correctness finding** issue
template.

The count has gone up twice: once when a fix for one defect created a smaller
one, and once when two more crates were measured for the first time. **A ledger
that only ever shrinks is a ledger that has stopped looking.**

## Parity fixtures

`crates/cynepic-causal/tests/fixtures/parity.json` holds reference answers
computed by numpy, scipy and networkx. It is committed, so the test suite needs
no Python.

Regenerate only when adding cases:

```bash
python3 scripts/generate_parity_fixtures.py   # needs numpy, scipy, networkx
```

**If a regenerated value differs from what is committed, that is a finding to
investigate — not a fixture to overwrite.** Either our implementation drifted or
the reference library changed behaviour, and both are worth knowing before the
evidence is quietly replaced.

## Naming examples and benchmarks

**Example binaries must have workspace-unique names.** Cargo writes every
example in the workspace to one `target/debug/examples/` directory, keyed by
name and not by crate. Four crates each had an example called `latency_report`;
on Linux that silently works, and on Windows the second linker to reach
`latency_report.exe` fails with `LNK1104: cannot open file` because the first
still holds it.

It is a race, so it is intermittent: CI stayed green with two colliding
examples and failed with three. Prefix with the crate — `causal_latency`,
`bayes_latency`, `graph_latency`, `guardian_latency`.

## Comparison harnesses

Cross-implementation numbers live in `scripts/compare_*.py`, one per reference
implementation, each printing the versions it used. Three rules:

1. **Both halves on one machine, or neither.** A cross-machine ratio is not a
   result, and every harness says so in its own output.
2. **Assert the work actually happened.** The LangGraph fixture checks the node
   count; a framework that skipped nodes would otherwise look wonderfully fast.
3. **Publish the cells where we lose.** The Beta credible interval was measured
   at 1.2x *slower* than scipy, and publishing that is what led to the fix. A
   table with no losses is advertising.

Nothing is installed into a contributor's environment: langgraph runs from a
throwaway venv, OPA from a downloaded binary.

## Lint policy

`[workspace.lints]` in `causalrust/Cargo.toml` turns this project's conventions
into build failures. `unsafe_code` is forbidden; `missing_debug_implementations`,
`unreachable_pub` and `clippy::correctness` are denied.

Some lints are set to `allow`. **Those are tracked debt, not settled policy.**
Each carries a measured violation count and the work that clears it:

```toml
# DEBT — the panic-freedom set. Measured violations:
#   indexing_slicing  104   unwrap_used  7   expect_used  4   panic  0
indexing_slicing = "allow"
```

Do not add new violations of a debt lint. If you clear one, flip it to `deny`
and delete the comment — that is a very welcome pull request.

## Style

The codebase is deliberately comment-dense in one specific way: comments explain
**why**, and especially why an obvious-looking alternative was rejected. Compare:

```rust
// Bisection rather than Newton because a tool with 200 successes and no
// failures gives Beta(201, 1), where the density lives in the last thousandth
// of the range and a Newton step leaves [0, 1].
```

against `// use bisection`. The first survives contact with a future reader who
is about to "optimise" it.

Beyond that: `cargo fmt` decides formatting, and library code returns `Result`
rather than panicking.

## Commit messages

Conventional-commit prefixes (`feat:`, `fix:`, `test:`, `ci:`, `docs:`,
`chore:`). Beyond the subject line, say what you measured and what it said.
`git log` here is meant to be readable as an account of what was learned, not
only of what changed.

## Pull requests

- One coherent change. If a diff is dominated by formatting or line-ending
  churn, split that into its own commit so the real change is reviewable.
- State what you measured. "Tests pass" is not a measurement.
- If you were wrong about something along the way, say so in the PR. Several of
  this project's most useful findings came from a hypothesis that measurement
  refuted, and recording that is worth more than appearing to have been right.

## Reporting security issues

Do not open a public issue. See [SECURITY.md](SECURITY.md).

## Licence

Apache-2.0. By contributing you agree your contributions are licensed under it.
See [NOTICE](NOTICE) — Apache-2.0 grants no trademark rights, and the
CARF/CYNEPIC marks are reserved.
