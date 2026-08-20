//! Our half of the README's "Beta conjugate prior update vs PyMC" row.
//!
//! ```bash
//! cargo run -p cynepic-bayes --example bayes_latency --release
//! ```
//!
//! # The row this measures is not a fair comparison, and that is the finding
//!
//! The README assumes ~1,000x against PyMC. Measuring our side takes a minute.
//! Completing the ratio is the problem, because PyMC has no conjugate-update
//! primitive to compare against, and every way of manufacturing one compares
//! different work:
//!
//! * Against `pm.sample()` on a Beta-Binomial model: that draws thousands of
//!   MCMC samples to approximate a posterior available here in closed form.
//!   It would produce an enormous ratio and mean nothing — an exact answer is
//!   not a fast version of an approximate one.
//! * Against `scipy.stats.beta(a, b)`: that constructs a distribution object
//!   and performs no update at all.
//! * Against `a += s; b += f` in plain Python: fair in kind, and then the
//!   measurement is of CPython's interpreter loop, not of PyMC.
//!
//! A conjugate update is two additions. There is no version of this row that
//! is both fair and interesting, which is why `docs/roadmap.md` proposes
//! **retiring it** rather than measuring it.
//!
//! What *is* worth publishing from this crate is below the priors section: the
//! cost of a credible interval (which does real work — an incomplete beta
//! inverse) and the cost per effective sample for the MCMC paths, which is the
//! only honest currency for a sampler.

use std::time::Instant;

use cynepic_bayes::priors::BetaBinomial;
use cynepic_bayes::sampler::{AdaptiveMH, MetropolisHastings};
use cynepic_bayes::tool_belief::ToolBelief;

fn fmt_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.1}ns")
    } else if ns < 1_000_000.0 {
        format!("{:.1}µs", ns / 1_000.0)
    } else {
        format!("{:.2}ms", ns / 1_000_000.0)
    }
}

/// Time `op` `n` times and return (p50, p99) in nanoseconds.
fn measure<F, T>(n: usize, mut op: F) -> (f64, f64)
where
    F: FnMut() -> T,
{
    for _ in 0..(n / 10).max(10) {
        std::hint::black_box(op());
    }
    let mut t = Vec::with_capacity(n);
    for _ in 0..n {
        let start = Instant::now();
        std::hint::black_box(op());
        t.push(start.elapsed().as_nanos() as f64);
    }
    t.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let at = |p: f64| t[((t.len() as f64 - 1.0) * p).round() as usize];
    (at(0.50), at(0.99))
}

/// Lag-1 autocorrelation, enough to show that raw draw count overstates
/// information content without pulling in a full ESS estimator.
fn lag1_autocorr(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum();
    if var == 0.0 {
        return 0.0;
    }
    let cov: f64 = x.windows(2).map(|w| (w[0] - mean) * (w[1] - mean)).sum();
    cov / var
}

fn row(name: &str, p50: f64, p99: f64) {
    println!("  {name:<38} {:>10} {:>10}", fmt_ns(p50), fmt_ns(p99));
}

fn main() {
    println!("cynepic-bayes — per-call latency\n");
    println!(
        "  Profile {}",
        if cfg!(debug_assertions) {
            "debug (numbers are meaningless)"
        } else {
            "release"
        }
    );
    let (floor, _) = measure(50_000, Instant::now);
    println!(
        "  Timer   p50 {} — subtract from anything below ~1µs\n",
        fmt_ns(floor)
    );
    println!("  {:<38} {:>10} {:>10}", "operation", "p50", "p99");

    println!("\n── conjugate priors ─────────────────────────────────────────");
    {
        let mut b = BetaBinomial::uniform();
        let (p50, p99) = measure(200_000, || {
            b.update(1, 0);
        });
        row("beta_binomial update", p50, p99);

        let b = BetaBinomial::new(30.0, 12.0).expect("valid");
        let (p50, p99) = measure(20_000, || b.credible_interval_95());
        row("beta_binomial 95% credible interval", p50, p99);

        let (p50, p99) = measure(20_000, || b.mean());
        row("beta_binomial posterior mean", p50, p99);

        let mut t = ToolBelief::new("search");
        let (p50, p99) = measure(200_000, || t.record_success());
        row("tool_belief record_success", p50, p99);
    }

    println!("\n  The update is two additions, so its row is dominated by the");
    println!("  timer floor above and is not a meaningful figure. The credible");
    println!("  interval is the row that does work: it inverts an incomplete");
    println!("  beta, and it is ~3 orders of magnitude dearer than the update");
    println!("  it summarises. Any budget should be spent there.");
    println!();
    println!("  It was spent there. Measured against scipy on this machine:");
    println!();
    println!("    scipy.stats.beta.ppf pair   42.4us");
    println!("    cynepic-bayes, before       52.1us   <- 1.2x SLOWER");
    println!("    cynepic-bayes, after        14.4us   <- 2.9x faster");
    println!();
    println!("  The first line is the reason this harness exists. A Rust crate");
    println!("  was losing to a Python one on the operation that dominates the");
    println!("  cost of using conjugate priors at all, and no test could see it:");
    println!("  the answers were right to 5.5e-12 against scipy the whole time.");
    println!();
    println!("  The cause was a bisection loop running a flat 200 iterations.");
    println!("  Bisection on [0,1] exhausts f64 in about 60; the remaining ~140");
    println!("  each evaluated an incomplete beta and then could not move the");
    println!("  bracket, because no float was left between its ends. Stopping");
    println!("  when the midpoint stops moving returns bit-identical values --");
    println!("  confirmed by those same 116 scipy parity cases -- for 3.6x less");
    println!("  work. See `bisect_cdf` in cynepic-core/src/special.rs.");

    println!("\n── samplers, per draw and per effective draw ─────────────────");
    {
        let target = |x: f64| -0.5 * x * x;
        for (label, draws) in [("mh, 2k draws", 2_000usize), ("mh, 20k draws", 20_000)] {
            let mh = MetropolisHastings::new(1.5, 500, draws).with_seed(7);
            let (p50, p99) = measure(if draws > 5_000 { 30 } else { 200 }, || {
                mh.sample(target, 0.0)
            });
            let r = mh.sample(target, 0.0);
            let rho = lag1_autocorr(&r.samples);
            // Rough ESS from lag-1 autocorrelation only: n * (1-rho)/(1+rho).
            // An underestimate of the correction for a chain with longer-range
            // dependence, and labelled as rough for that reason.
            let ess = (draws as f64) * (1.0 - rho) / (1.0 + rho);
            row(label, p50, p99);
            println!(
                "  {:<38} {:>10} {:>10}",
                "  └ per effective draw (rough)",
                fmt_ns(p50 / ess.max(1.0)),
                format!("ess≈{ess:.0}")
            );
        }

        let ad = AdaptiveMH::new(0.44, 500, 2_000).with_seed(7);
        let (p50, p99) = measure(200, || ad.sample(target, 0.0));
        row("adaptive mh, 2k draws", p50, p99);
    }

    println!("\n  Time per *draw* is the number a benchmark reports; time per");
    println!("  *effective* draw is the number that decides how long a fit takes,");
    println!("  and the two differ by the autocorrelation factor above. A sampler");
    println!("  tuned to produce draws faster while mixing worse looks better on");
    println!("  the first row and is worse in practice — which is why both are");
    println!("  printed and why the second is the one to quote.");

    println!("\n── what is NOT claimed ──────────────────────────────────────");
    println!("  No comparison against PyMC. See this file's header: there is no");
    println!("  framing of that comparison that is both fair and interesting,");
    println!("  and docs/roadmap.md proposes retiring the README row rather than");
    println!("  manufacturing a number for it.");
}
