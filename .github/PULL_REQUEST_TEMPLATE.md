## What this changes

<!-- One or two sentences. -->

## What you measured

<!-- Required for anything touching statistical or routing behaviour.
     "Tests pass" is not a measurement. Paste the numbers.

     cargo run -p cynepic-causal --example coverage_report      --release
     cargo run -p cynepic-bayes  --example calibration_report   --release
     cargo run -p cynepic-router --example classifier_report    --release

     If a hypothesis you started with turned out wrong, say so — that is
     worth more than appearing to have been right. -->

## Findings ledger

- [ ] No change to the open findings count
- [ ] **Closes** a finding — `#[ignore]` deleted, `BASELINE` lowered,
      `docs/FINDINGS.md` updated, all in this PR
- [ ] **Opens** a finding — spec added, `docs/FINDINGS.md` entry written,
      `BASELINE` raised, all in this commit

## Checklist

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [ ] `cargo clippy --workspace --all-targets --no-default-features -- -D warnings`
- [ ] `cargo test --workspace --all-features`
- [ ] `cargo doc --workspace --no-deps --all-features` (broken links fail CI)
- [ ] `./scripts/findings-ratchet.sh`
- [ ] No new violations of a lint marked DEBT in `[workspace.lints]`
- [ ] No new unverified performance claims

## Breaking changes

<!-- Public API changes, with before/after. Delete if none. -->
