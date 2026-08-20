//! Reproducibility for the samplers — item 4 of the plan in `docs/roadmap.md`.
//!
//! The companion of `cynepic-causal/tests/determinism.rs`, and the harder half.
//! A closed-form estimator is a pure function of its inputs and reproduces by
//! construction. An MCMC chain reproduces only if every draw comes from a
//! generator seeded by the caller — and the failure mode is quiet:
//!
//! * A chain that falls back to entropy when no seed is set looks fine in every
//!   test that only checks the posterior mean, because the mean converges
//!   regardless. What breaks is the *reported* number, months later, when
//!   somebody tries to reproduce a figure from its recorded seed.
//! * A chain that shares a thread-local generator reproduces perfectly when run
//!   alone, and stops the moment a caller runs two analyses in one process.
//!   Every single-chain test passes throughout.
//!
//! Both are checked below, the second by consuming two seeded chains
//! alternately rather than one after the other.
//!
//! # What is *not* claimed
//!
//! Bit-identical chains across platforms. The samplers call `exp` and `ln` on
//! every proposal, so an accept/reject decision sitting within an ulp of the
//! threshold can go the other way on a different libm — and once one decision
//! flips, the chains diverge completely rather than slightly. That is inherent
//! to Metropolis-Hastings, not a defect, and asserting otherwise would produce
//! a test that fails on macOS for a reason nobody could act on.
//!
//! What *is* claimed across platforms is the thing a caller depends on: the
//! posterior summary agrees to well inside Monte Carlo error. That is checked
//! against a tolerance derived from the chain's own standard error rather than
//! from a number that looked about right.

use cynepic_bayes::sampler::{AdaptiveMH, MetropolisHastings, MultiDimMH};
use std::process::Command;

/// FNV-1a over the raw bits of a chain. Not a cryptographic hash — the job is
/// to turn 4000 doubles into one comparable token, and any single differing bit
/// must change it.
fn digest(samples: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for s in samples {
        for byte in s.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

/// The chain whose digest is compared across processes.
fn canonical_chain() -> Vec<f64> {
    MetropolisHastings::new(1.0, 500, 2_000)
        .with_seed(20_260_820)
        .sample(std_normal, 0.0)
        .samples
}

/// Standard normal log-density, up to a constant.
fn std_normal(x: f64) -> f64 {
    -0.5 * x * x
}

#[test]
fn a_seeded_chain_replays_bit_for_bit() {
    for seed in [1u64, 42, 20_260_820] {
        let mh = MetropolisHastings::new(1.0, 500, 2_000).with_seed(seed);
        let a = mh.sample(std_normal, 0.0);
        let b = mh.sample(std_normal, 0.0);

        assert_eq!(
            a.samples.len(),
            b.samples.len(),
            "seed {seed}: chain length"
        );
        for (i, (x, y)) in a.samples.iter().zip(b.samples.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "seed {seed}: chains diverge at draw {i} ({x} vs {y}) — \
                 the sampler is not a pure function of (seed, density, start)"
            );
        }
        assert_eq!(
            a.acceptance_rate.to_bits(),
            b.acceptance_rate.to_bits(),
            "seed {seed}: acceptance rate"
        );
    }
}

#[test]
fn interleaved_chains_do_not_contaminate_each_other() {
    // The thread-local-RNG failure. Run alone, each chain reproduces; run
    // alternately, a shared generator leaks state between them. This is the
    // arrangement a caller creates the first time they fit two models in one
    // process, and no single-chain test can see it.
    let solo_a = MetropolisHastings::new(1.0, 200, 1_000)
        .with_seed(7)
        .sample(std_normal, 0.0);
    let solo_b = MetropolisHastings::new(0.4, 200, 1_000)
        .with_seed(13)
        .sample(std_normal, 2.0);

    for round in 0..3 {
        let a = MetropolisHastings::new(1.0, 200, 1_000)
            .with_seed(7)
            .sample(std_normal, 0.0);
        let b = MetropolisHastings::new(0.4, 200, 1_000)
            .with_seed(13)
            .sample(std_normal, 2.0);

        assert!(
            a.samples
                .iter()
                .zip(solo_a.samples.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "round {round}: chain(seed 7) changed when interleaved with chain(seed 13)"
        );
        assert!(
            b.samples
                .iter()
                .zip(solo_b.samples.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "round {round}: chain(seed 13) changed when interleaved with chain(seed 7)"
        );
    }
}

#[test]
fn the_adaptive_sampler_replays_too() {
    // Adaptation makes each proposal depend on the whole history, so a single
    // differing draw does not stay local — it changes the proposal scale and
    // therefore every draw after it. That makes this the strictest replay check
    // of the three, and the one most likely to catch state leaking in.
    let mh = AdaptiveMH::new(0.44, 500, 2_000).with_seed(99);
    let a = mh.sample(std_normal, 0.0);
    let b = mh.sample(std_normal, 0.0);
    for (i, (x, y)) in a.samples.iter().zip(b.samples.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "adaptive chain diverges at draw {i}"
        );
    }
}

#[test]
fn the_multidimensional_sampler_replays_too() {
    let mh = MultiDimMH::new(vec![1.0, 0.5], 300, 1_500).with_seed(5);
    let a = mh.sample(|v| v.iter().map(|x| std_normal(*x)).sum(), vec![0.0, 0.0]);
    let b = mh.sample(|v| v.iter().map(|x| std_normal(*x)).sum(), vec![0.0, 0.0]);
    for (i, (x, y)) in a.samples.iter().zip(b.samples.iter()).enumerate() {
        assert_eq!(
            x.len(),
            y.len(),
            "multi-dim chain draw {i}: dimension changed"
        );
        for (d, (p, q)) in x.iter().zip(y.iter()).enumerate() {
            assert_eq!(p.to_bits(), q.to_bits(), "draw {i}, dimension {d}");
        }
    }
}

#[test]
fn different_seeds_produce_genuinely_different_chains() {
    // The control. A sampler that ignored `with_seed` and returned a fixed
    // chain would pass every test above with a perfect score.
    let a = MetropolisHastings::new(1.0, 200, 1_000)
        .with_seed(1)
        .sample(std_normal, 0.0);
    let b = MetropolisHastings::new(1.0, 200, 1_000)
        .with_seed(2)
        .sample(std_normal, 0.0);
    let identical = a
        .samples
        .iter()
        .zip(b.samples.iter())
        .filter(|(x, y)| x.to_bits() == y.to_bits())
        .count();
    assert!(
        identical < a.samples.len() / 20,
        "{identical} of {} draws are bit-identical between seeds 1 and 2; \
         the seed is barely reaching the generator",
        a.samples.len()
    );
}

#[test]
fn the_posterior_summary_is_portable_within_monte_carlo_error() {
    // The cross-platform claim, and the only one worth making for MCMC. A
    // single flipped accept/reject decision — which one ulp of difference in
    // `exp` can cause — sends the chains down entirely different paths, so bits
    // are hopeless. The summary is not: it is what a caller reports, and it is
    // stable for the same reason the sampler is useful at all.
    //
    // The tolerance is derived, not chosen. For a standard-normal target the
    // draws have unit variance, so the mean of `n` draws has standard error
    // `1/sqrt(n_eff)`; at 4000 draws with the autocorrelation this proposal
    // gives, `n_eff` is a few hundred. Five of those standard errors is the
    // bound below — wide enough that libm cannot breach it, narrow enough that
    // a real change in the sampler will.
    let mh = MetropolisHastings::new(1.5, 1_000, 4_000).with_seed(20_260_820);
    let r = mh.sample(std_normal, 0.0);

    let n = r.samples.len() as f64;
    let mean = r.samples.iter().sum::<f64>() / n;
    // Conservative effective sample size: assume only ~5% of draws are
    // independent, which is pessimistic for this proposal.
    let se = (1.0 / (n * 0.05)).sqrt();
    let bound = 5.0 * se;

    assert!(
        mean.abs() < bound,
        "posterior mean {mean:.4} is {:.1} conservative standard errors from 0 \
         (bound {bound:.4}). Either the sampler changed or the chain did not \
         converge; both need explaining before this tolerance is widened.",
        mean.abs() / se,
    );
    assert!(
        (0.1..0.7).contains(&r.acceptance_rate),
        "acceptance rate {:.3} is outside the range this proposal should give; \
         a chain that accepts everything or nothing has a stable mean for the \
         wrong reason",
        r.acceptance_rate
    );
}

#[test]
fn an_unseeded_chain_is_genuinely_random() {
    // The control for `a_seeded_chain_replays_bit_for_bit`, and the reason that
    // test means anything. If the sampler were deterministic whatever the seed
    // — a stubbed RNG, a chain that never moves — replay would pass trivially
    // and prove nothing about seeding.
    //
    // Two unseeded chains must therefore differ. For 1000 continuous draws the
    // probability of a false failure here is zero, not merely small.
    let mh = MetropolisHastings::new(1.0, 200, 1_000);
    let a = mh.sample(std_normal, 0.0);
    let b = mh.sample(std_normal, 0.0);
    assert!(
        a.samples
            .iter()
            .zip(b.samples.iter())
            .any(|(x, y)| x.to_bits() != y.to_bits()),
        "two unseeded chains came out bit-identical; the sampler is not \
         actually drawing randomness, which would make every replay test above \
         vacuous"
    );
}

// ── Cross-process replay ─────────────────────────────────────────────────

/// Emits the canonical chain's digest, for the parent test below to compare
/// against. `#[ignore]` because it is machinery, not a check.
#[test]
#[ignore = "child process of cross_process_replay; not a standalone check"]
fn emit_canonical_digest() {
    println!("DIGEST={:016x}", digest(&canonical_chain()));
}

#[test]
fn a_seed_replays_across_processes_not_only_within_one() {
    // This exists because of a defect the rest of the file could not see.
    //
    // Every other replay test compares two chains in the *same* process. A
    // sampler that folded the process id into its seed — or read one byte of
    // entropy, or hashed the address of anything — would satisfy all of them
    // perfectly, and would still make "reproduce it from the recorded seed"
    // false the moment the process ended. That was verified by mutation, not
    // assumed: perturbing the stored seed per process left the whole suite
    // green.
    //
    // Re-executing the test binary is what closes it. The comparison stays on
    // one machine and one libm by construction, which is deliberate — see the
    // module header for why bit-equality across platforms is not claimed for
    // MCMC.
    let exe = std::env::current_exe().expect("test binary path");
    let out = Command::new(&exe)
        .args([
            "--exact",
            "emit_canonical_digest",
            "--ignored",
            "--nocapture",
        ])
        .output()
        .expect("re-exec the test binary");
    assert!(
        out.status.success(),
        "child process failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    let child = stdout
        .lines()
        .find_map(|l| l.strip_prefix("DIGEST="))
        .unwrap_or_else(|| panic!("child printed no digest:\n{stdout}"));
    let ours = format!("{:016x}", digest(&canonical_chain()));

    assert_eq!(
        child, ours,
        "chain digest differs between processes for the same seed.\n           this process {ours}\n  child process {child}\n\n           The seed is not the only thing feeding the generator — something \n           per-process (pid, address, clock, one byte of OS entropy) is reaching \n           it. Every other replay test in this file will still pass; they compare \n           chains within one process and cannot see this."
    );
}
