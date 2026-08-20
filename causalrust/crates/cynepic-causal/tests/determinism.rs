//! Reproducibility: item 4 of the benchmarking plan in `docs/roadmap.md`.
//!
//! # Two claims, deliberately not conflated
//!
//! "Reproducible" is usually said as if it were one property. It is two, they
//! have different strengths, and quoting the strong one when only the weak one
//! holds is how a reproducibility claim becomes false.
//!
//! **Replay** — same seed, same binary, same machine — must be *bit-identical*.
//! Anything less and an audit trail cannot be replayed: a regulator re-running
//! a stored seed would get a number that differs from the one on the record,
//! and no amount of "within tolerance" makes that a satisfying answer.
//!
//! **Portability** — same seed, different platform — is bit-identical only for
//! operations IEEE-754 requires to be correctly rounded: `+ - * /` and `sqrt`.
//! It is *not* bit-identical for `exp`, `ln`, `lgamma` or `erf`, because those
//! are libm routines and libm is a different implementation on glibc, on macOS
//! and on MSVC. Each is free to be off by an ulp or two in a different place.
//!
//! So this file draws the line where the standard draws it:
//!
//! | Path | Uses | Claim asserted here |
//! |---|---|---|
//! | `difference_in_means` | `+ - * / sqrt` | bit-identical everywhere |
//! | `ols_adjusted` (pivoted QR) | `+ - * / sqrt` | bit-identical everywhere |
//! | `ipw` (logistic IRLS) | `exp`, `ln` | agreement to a stated tolerance |
//!
//! The CI matrix runs ubuntu / macos / windows, so the bit-identical rows are
//! not an argument — they are checked on three libms every push. **If one of
//! them fails, that is a finding for `docs/FINDINGS.md`, not a tolerance to
//! widen.** Widening it would convert a discovered fact into a hidden one.
//!
//! # Why the inputs contain no random numbers
//!
//! Every fixture below is built from integer arithmetic divided by a power of
//! two, so each input is exactly representable and identical on every platform
//! by construction. Generating them with a seeded RNG would have been less
//! code and would have quietly undermined the point: `rand`'s normal
//! distribution uses a ziggurat whose tail calls `ln`, so the *inputs* would
//! already differ across libms and the test could no longer tell an input
//! difference from an output one.

use cynepic_causal::estimate::linear::LinearATEEstimator;
use cynepic_causal::estimate::propensity::PropensityScoreEstimator;
use cynepic_testkit::Dgp;
use ndarray::{Array1, Array2};

/// A design with no random numbers in it.
///
/// Values are small integers over 4, so every entry is exact in binary
/// floating point and byte-identical on every target.
fn fixed_design(n: usize, p: usize) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let treatment = Array1::from_shape_fn(n, |i| f64::from(u8::from(i % 3 != 0)));
    let covariates = Array2::from_shape_fn((n, p), |(i, j)| {
        f64::from(u8::try_from((i * 7 + j * 13) % 23).unwrap_or(0)) / 4.0 - 2.75
    });
    let outcome = Array1::from_shape_fn(n, |i| {
        let x0 = covariates[(i, 0)];
        let t = treatment[i];
        // Deliberately arithmetic-only: no exp, no ln, nothing from libm.
        1.5 * t + 0.75 * x0 + f64::from(u8::try_from(i % 11).unwrap_or(0)) / 8.0
    });
    (treatment, outcome, covariates)
}

/// Report a bit-level mismatch in a form that says what to do about it.
fn assert_bits(label: &str, got: f64, want_bits: u64) {
    let want = f64::from_bits(want_bits);
    assert_eq!(
        got.to_bits(),
        want_bits,
        "{label} is not bit-reproducible on this target.\n  \
         got  {got:.17e}  (bits 0x{:016x})\n  \
         want {want:.17e}  (bits 0x{want_bits:016x})\n  \
         relative difference {:.3e}\n\n  \
         This path uses only + - * / and sqrt, which IEEE-754 requires to be \n  \
         correctly rounded, so a difference here is a real finding: either the \n  \
         implementation acquired a libm call, or the toolchain is contracting \n  \
         to FMA. Record it in docs/FINDINGS.md. Do NOT relax this to a \n  \
         tolerance — that hides the fact rather than fixing it.",
        got.to_bits(),
        ((got - want) / want).abs(),
    );
}

// ── Portability: bit-identical across platforms ──────────────────────────

#[test]
fn difference_in_means_is_bit_identical_on_every_platform() {
    let (t, y, _) = fixed_design(400, 3);
    let r = LinearATEEstimator::difference_in_means(&t, &y).expect("well-posed");
    assert_bits("difference_in_means ATE", r.ate(), 0x3ff7_2754_5179_60c9);
    assert_bits(
        "difference_in_means SE",
        r.std_error(),
        0x3fc2_44e5_008c_2a97,
    );
}

#[test]
fn ols_is_bit_identical_on_every_platform() {
    let (t, y, x) = fixed_design(400, 3);
    let r = LinearATEEstimator::ols_adjusted(&t, &y, &x).expect("well-posed");
    assert_bits("ols_adjusted ATE", r.ate(), 0x3ff8_10dd_19d8_84b3);
    assert_bits("ols_adjusted SE", r.std_error(), 0x3fa5_91aa_6003_9a7e);
}

// ── Portability: tolerance, because libm is involved ─────────────────────

#[test]
fn ipw_agrees_across_platforms_to_a_stated_tolerance() {
    let (t, y, x) = fixed_design(400, 3);
    let r = PropensityScoreEstimator::ipw(&t, &y, &x).expect("well-posed");

    // Recorded on linux/aarch64/glibc. The tolerance is not a guess about how
    // wrong libm might be — it is a bound on how far an IRLS fixed point can
    // move when its inputs differ in the last ulp, and it is loose enough to
    // survive that and tight enough to catch an actual change in the
    // estimator.
    // Written as bits, not decimal: the shortest decimal that round-trips is
    // what clippy insists on, and rewriting a recorded value to satisfy a
    // formatting lint is how a golden quietly stops being the thing that was
    // measured.
    const GOLDEN_ATE: f64 = f64::from_bits(0x3ff8_16a4_15bb_212e);
    const TOL: f64 = 1e-9;

    let dev = (r.ate() - GOLDEN_ATE).abs();
    assert!(
        dev < TOL,
        "ipw ATE moved by {dev:.3e}, past the {TOL:.0e} portability tolerance.\n  \
         got {:.17e}, recorded {GOLDEN_ATE:.17e}\n\n  \
         A deviation of order 1e-16 is libm and is expected. A deviation of \n  \
         order 1e-6 or larger is the estimator changing, and the golden value \n  \
         should not be updated until it is understood which.",
        r.ate(),
    );
}

// ── Replay: bit-identical for the same seed on the same machine ──────────

#[test]
fn seeded_data_generation_replays_bit_for_bit() {
    // The weak claim, and the one an audit trail actually rests on. This uses
    // the RNG path deliberately — it is the path where non-determinism would
    // hide.
    for seed in [1u64, 7, 99] {
        let a = Dgp::new().with_n(500).with_p(4).sample(seed);
        let b = Dgp::new().with_n(500).with_p(4).sample(seed);
        for (i, (x, y)) in a.outcome.iter().zip(b.outcome.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "seed {seed}, outcome[{i}]");
        }
        for (i, (x, y)) in a.covariates.iter().zip(b.covariates.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "seed {seed}, covariate[{i}]");
        }
    }
}

#[test]
fn estimates_replay_bit_for_bit_from_a_seed() {
    let d = Dgp::new().with_n(2_000).with_p(5).sample(2026);
    let first =
        PropensityScoreEstimator::ipw(&d.treatment, &d.outcome, &d.covariates).expect("well-posed");
    for _ in 0..3 {
        let again = PropensityScoreEstimator::ipw(&d.treatment, &d.outcome, &d.covariates)
            .expect("well-posed");
        assert_eq!(
            first.ate().to_bits(),
            again.ate().to_bits(),
            "ipw is not a pure function of its inputs"
        );
        assert_eq!(first.std_error().to_bits(), again.std_error().to_bits());
    }
}

#[test]
fn interleaving_two_seeded_runs_does_not_change_either() {
    // The failure this exists to catch: a global or thread-local RNG. Under one
    // it, each run alone still reproduces perfectly — `sample(1)` twice in a
    // row matches — and the tests above all pass. It is only when two seeded
    // streams are consumed *alternately* that the shared state leaks between
    // them, which is exactly what happens once a caller runs two analyses in
    // one process.
    let solo_a = Dgp::new().with_n(300).sample(11);
    let solo_b = Dgp::new().with_n(300).sample(22);

    let mut interleaved_a = Vec::new();
    let mut interleaved_b = Vec::new();
    for _ in 0..3 {
        interleaved_a.push(Dgp::new().with_n(300).sample(11));
        interleaved_b.push(Dgp::new().with_n(300).sample(22));
    }

    for (k, d) in interleaved_a.iter().enumerate() {
        assert!(
            d.outcome
                .iter()
                .zip(solo_a.outcome.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "seed 11 changed when interleaved with seed 22 (round {k}) — \
             the generator is sharing state between streams"
        );
    }
    for (k, d) in interleaved_b.iter().enumerate() {
        assert!(
            d.outcome
                .iter()
                .zip(solo_b.outcome.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "seed 22 changed when interleaved with seed 11 (round {k})"
        );
    }
}

#[test]
fn different_seeds_actually_produce_different_data() {
    // The control for every test above. A generator that ignored its seed would
    // pass all of them — perfectly reproducible, and perfectly useless.
    let a = Dgp::new().with_n(300).sample(11);
    let b = Dgp::new().with_n(300).sample(22);
    let differing = a
        .outcome
        .iter()
        .zip(b.outcome.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert!(
        differing > 290,
        "only {differing}/300 outcomes differ between seeds 11 and 22; \
         the seed is barely reaching the generator"
    );
}

#[test]
#[ignore = "prints golden values; run with --ignored --nocapture to re-record"]
fn record_golden_values() {
    let (t, y, x) = fixed_design(400, 3);
    let d = LinearATEEstimator::difference_in_means(&t, &y).expect("well-posed");
    let o = LinearATEEstimator::ols_adjusted(&t, &y, &x).expect("well-posed");
    let i = PropensityScoreEstimator::ipw(&t, &y, &x).expect("well-posed");
    for (label, v) in [
        ("dim ate", d.ate()),
        ("dim se", d.std_error()),
        ("ols ate", o.ate()),
        ("ols se", o.std_error()),
        ("ipw ate", i.ate()),
    ] {
        println!("{label:>8}  0x{:016x}  {v:.17e}", v.to_bits());
    }
}
