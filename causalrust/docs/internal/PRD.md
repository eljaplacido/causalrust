# PRD: Public Launch Readiness — cynepic-rs

## Objective
Make cynepic-rs repo fully consistent, correct, and secure before public introduction.

## Requirements

### R1: Fix All Broken Code Examples
Every code example in READMEs and EXPERIMENTS.md must compile against the actual API.

### R2: Correct All Numbers
Test counts, LOC counts, and version references must be accurate across all docs.

### R3: Remove Phantom Dependencies
Remove unused workspace deps (polars, async-openai) and unused crate deps (axum in router).

### R4: Fix Export Gaps
All types referenced in docs must be re-exported from crate roots.

### R5: Fix Security Issues (Medium+)
- CausalDag deserialization roundtrip
- CircuitBreaker atomic ordering
- MCMC sampler panic-on-invalid-input
- NormalNormal NaN on empty input
- Replace assert! with Result in library code

### R6: Update Stale Docs
- PITCH.md status section
- architecture.md file trees
- integration.md false Polars claim
- CLAUDE.md repository layout

### R7: .gitignore Hygiene
Add .env, *.pem, *.key, credentials.*, .claude/ to .gitignore.

## Success Criteria
- `cargo test -p cynepic-core -p cynepic-guardian --no-default-features -p cynepic-router -p cynepic-graph` passes
- All code examples in root README.md match actual API signatures
- Test counts in all docs match actual counts
- No phantom dependencies in workspace Cargo.toml or crate Cargo.toml files
- Security issues documented in STATE.md are resolved
