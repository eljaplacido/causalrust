# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| < 0.2   | No        |

Pre-1.0. The API changes between minor versions; see [CHANGELOG.md](CHANGELOG.md).

## Reporting a vulnerability

**Do not open a public issue.**

Use GitHub's private reporting:
[Report a vulnerability](https://github.com/eljaplacido/causalrust/security/advisories/new)

Or email **eljailari.suhonen@gmail.com** with `SECURITY` in the subject.

Please include what you can of: affected crate and version, reproduction steps
or a proof of concept, the impact you see, and any suggested fix. A minimal
reproducing test is the single most useful thing you can send.

**Response targets.** Acknowledgement within 3 working days; an initial
assessment within 10. Fix timelines depend on severity and will be communicated
in the thread rather than promised here.

Disclosure is coordinated: a fix ships first, then an advisory, with credit
unless you prefer otherwise.

## What counts as a vulnerability here

The usual — memory unsafety, denial of service, injection through a policy or
query surface, dependency advisories reachable from a published crate.

And one that is specific to this project:

> **A statistically wrong number that a caller cannot detect is treated as a
> security issue, not merely a bug**, when it can be triggered by attacker- or
> user-controlled input.

An estimator that reports a confident effect on data that cannot support one is
an integrity failure. It reached production once already — an inverse-probability
estimator with 0.0% confidence-interval coverage, recorded in
[docs/FINDINGS.md](causalrust/docs/FINDINGS.md) — and the whole validation
harness exists because of it. If you find another, we want to hear about it
through this channel.

Correctness defects that are *not* input-triggered belong in a public issue
under the **Correctness finding** template. They are tracked openly, with
measurements, in `docs/FINDINGS.md`.

## Hardening notes

- `unsafe_code` is `forbid` across the workspace.
- `cargo deny check` runs in CI: advisories, licences, banned crates, sources.
- Library code returns `Result` rather than panicking. The remaining
  `indexing_slicing` / `unwrap_used` sites are counted as tracked debt in
  `[workspace.lints]`.
- No crate reads the network or the filesystem at runtime except
  `cynepic-server`, which binds a socket, and `cynepic-mcp`, which speaks
  JSON-RPC over stdio.
- Rego policy sources are treated as sensitive: `RegoPolicyEvaluator`'s `Debug`
  impl is deliberately opaque so authorization logic cannot leak into logs.

## Not yet published

These crates are not on crates.io. Anything claiming to be `cynepic-*` on a
registry today did not come from this project.
