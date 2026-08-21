# Releasing

The runbook for cutting a version and publishing to crates.io.

## What gets published

Six crates. The dependency order below is also the **publish order** — crates.io
resolves each against what is already there, so a dependent published before its
dependency fails with "no matching package".

```
1. cynepic-core        (no internal deps)
2. cynepic-guardian    ┐
3. cynepic-router      │ all depend on core only
4. cynepic-causal      │
5. cynepic-bayes       │
6. cynepic-graph       ┘
```

**Not published**, and each for a reason:

| Crate | Why not |
|---|---|
| `cynepic-testkit` | Internal validation machinery. It exists so the published crates can prove their claims; a consumer has no use for it, and publishing it would imply a stability promise nobody wants to keep. |
| `cynepic-server` | A binary. A Rust consumer cannot depend on it. |
| `cynepic-mcp` | A binary. |
| `cynepic-pyo3` | Ships as a Python wheel via maturin. Publishing a `cdylib` to a Rust registry offers something nobody can link. |

All four carry `publish = false`, and the `package` CI job derives the
publishable set from `cargo metadata` rather than a hardcoded list, so a new
crate cannot escape the checks by accident.

## Before you tag

Everything below must be green. CI enforces all of it, but a failed publish is
irreversible in a way a failed CI run is not — crates.io permits yanking, never
deleting or replacing.

```bash
cd causalrust

cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features    -- -D warnings
cargo clippy --workspace --all-targets --no-default-features -- -D warnings
cargo test  --workspace --all-features
cargo test  --workspace --no-default-features
cargo doc   --workspace --no-deps --all-features
cargo deny  check
./scripts/findings-ratchet.sh
```

Then the three measurement artifacts. **Read them; do not just run them.** They
are the evidence behind every claim in the README, and a release that quotes a
number nobody re-measured is how the last set of wrong numbers shipped.

```bash
cargo run -p cynepic-causal --example coverage_report    --release
cargo run -p cynepic-bayes  --example calibration_report --release
cargo run -p cynepic-router --example classifier_report  --release
```

### Release checklist

- [ ] The gate above is green
- [ ] `docs/FINDINGS.md` matches what the measurement artifacts actually printed
- [ ] `CHANGELOG.md` has an entry for this version, with what changed **and what
      it measured**
- [ ] Breaking changes are listed with before/after code
- [ ] Open findings are named in the release notes. Shipping with known defects
      is fine; shipping without saying so is not
- [ ] No new unverified performance claims, and the existing ones are still
      marked
- [ ] `version` in `[workspace.package]` bumped
- [ ] Internal dependency `version` fields bumped to match — they are pinned
      exactly, so a stale one publishes a crate that cannot resolve
- [ ] `rust-version` still accurate; the `msrv` CI job checks the manifest and
      the workflow agree

## Version policy

Semantic versioning, with the pre-1.0 caveat that **minor versions may break the
API**. This is stated in `CHANGELOG.md` and should stay stated until 1.0.

Bump the workspace version in one place:

```toml
# causalrust/Cargo.toml
[workspace.package]
version = "0.3.0"
```

and every internal dependency that pins it:

```toml
cynepic-core = { path = "crates/cynepic-core", version = "0.3.0" }
```

`version` alongside `path` is not optional. A path-only dependency cannot be
published — `cargo publish` rejects it and `cargo deny` reports it as a
wildcard.

## Pre-flight, verified 2026-08-21

Checked before the 0.3.0 release. Re-check the first two if time has passed.

| Check | Result |
|---|---|
| All six names free on crates.io | ✅ `cynepic-core`, `-guardian`, `-router`, `-causal`, `-bayes`, `-graph` — and `cynepic` itself, if an umbrella crate is ever wanted |
| `cargo publish --dry-run -p cynepic-core` | ✅ packages 14 files; README, LICENSE and NOTICE all included |
| `cynepic-testkit` does not block anything | ✅ it is a **path-only dev-dependency with no `version`**, which `cargo publish` strips. Had it carried a version, every dependent would fail to publish because the testkit is `publish = false` |
| Every publishable crate has description, keywords, categories, docs.rs metadata | ✅ |

### The one error you will see and should ignore

`cargo publish --dry-run` on anything except `cynepic-core` fails with:

```
no matching package named `cynepic-core` found
location searched: crates.io index
```

That is the chicken-and-egg, not a manifest fault: the dependents resolve
`cynepic-core = { version = "0.3.0" }` against crates.io, where it does not
exist *yet*. It resolves itself the moment core is published, which is why the
order below is not optional. Do not "fix" it by removing the version from the
path dependency — a path-only dependency is a wildcard, which `cargo-deny`
rejects and `cargo publish` cannot resolve at all.

## Publishing

```bash
# Dry run first. `--no-verify` is deliberately NOT passed here: verification
# builds the packaged tarball from scratch, which is the only way to catch a
# file that is in your working tree but excluded from the package.
cargo publish -p cynepic-core --dry-run

# Then for real, in dependency order, waiting for the index between each.
cargo publish -p cynepic-core
cargo publish -p cynepic-guardian
cargo publish -p cynepic-router
cargo publish -p cynepic-causal
cargo publish -p cynepic-bayes
cargo publish -p cynepic-graph
```

crates.io takes a few seconds to index a new crate. If the next publish fails
with "no matching package named `cynepic-core`", wait and retry — it is a
propagation delay, not a manifest error.

### Tag and release

```bash
git tag -a v0.3.0 -m "cynepic-rs v0.3.0"
git push origin v0.3.0
gh release create v0.3.0 --notes-file <(sed -n '/## \[0.3.0\]/,/## \[/p' ../CHANGELOG.md)
```

## The Python wheel

Separate from the Rust release, and built with `extension-module` turned on —
it is off by default so `cargo build --workspace` links on macOS.

```bash
cd causalrust/bindings/pyo3
maturin build --release --features extension-module
```

## After the first publish

Two things become possible that are deliberately absent today:

- **`cargo-semver-checks`** in CI. It compares against a baseline published on
  crates.io; with nothing published it has nothing to compare to, and a job that
  cannot run is worse than no job. Add it to the release workflow immediately
  after the first publish.
- **A yank procedure.** If a released version is found to compute wrong numbers,
  yank it (`cargo yank --version X.Y.Z`), publish the fix, and add the finding to
  `docs/FINDINGS.md` with its measurement. Yanking does not remove the crate; it
  stops new dependents resolving to it.

## If you publish something wrong

You cannot delete or overwrite a published version. The recovery is:

1. `cargo yank --version X.Y.Z -p <crate>`
2. Fix it, add a regression spec, bump the patch version, publish.
3. Record it in `docs/FINDINGS.md` and `CHANGELOG.md`.
4. If it computed wrong numbers rather than failing to build, treat it as a
   security-class integrity issue: see [SECURITY.md](../../SECURITY.md).
