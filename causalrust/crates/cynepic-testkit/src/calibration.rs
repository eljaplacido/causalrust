//! Calibration for Bayesian inference — the analogue of coverage.
//!
//! # The claim a credible interval makes
//!
//! A 95% credible interval makes an operational promise that is directly
//! checkable: draw a parameter from the prior, generate data from it, form the
//! interval, and 95% of those intervals must contain the parameter that
//! generated the data. This is *not* a frequentist reinterpretation — it is the
//! Bayesian statement itself, averaged over the prior, and it holds exactly for
//! any correct posterior.
//!
//! So it can be measured, and an interval that fails it is wrong on its own
//! terms. That makes it the same kind of artifact as confidence-interval
//! coverage in [`crate::validate`]: a number that can come back bad.
//!
//! # Simulation-based calibration
//!
//! For samplers, where the posterior is approximate rather than closed-form,
//! coverage of a single interval is a coarse instrument. **Simulation-based
//! calibration** is the sharp one:
//!
//! ```text
//!   1. draw theta* from the prior
//!   2. draw data y from p(y | theta*)
//!   3. draw L samples from the posterior p(theta | y)
//!   4. record the RANK of theta* among those L samples
//! ```
//!
//! If the sampler targets the right posterior, that rank is uniform on
//! `0..=L`. Any departure is a diagnosis, and the shape of the departure names
//! the fault:
//!
//! | Rank histogram | Meaning |
//! |---|---|
//! | Uniform | Correct |
//! | ∪-shaped (mass at both ends) | Posterior too narrow — overconfident |
//! | ∩-shaped (mass in the middle) | Posterior too wide — underconfident |
//! | Sloped | Posterior biased in the direction of the slope |
//!
//! A ∪-shaped histogram is the dangerous one, and it is invisible to any test
//! that only checks the posterior mean.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Result of a credible-interval coverage run.
#[derive(Debug, Clone)]
pub struct CalibrationReport {
    /// What was measured.
    pub model: String,
    /// Replications performed.
    pub replications: usize,
    /// Nominal credible level, e.g. `0.95`.
    pub nominal: f64,
    /// Fraction of intervals that contained the generating parameter.
    pub coverage: f64,
    /// Mean interval width. Read with coverage: an interval can only be
    /// trusted if it is both correct and informative.
    pub mean_width: f64,
    /// Mean signed error of the posterior point estimate.
    pub bias: f64,
    /// Replications where the model could not produce an interval.
    pub n_failed: usize,
}

impl CalibrationReport {
    /// Whether coverage is within `tolerance` of nominal.
    pub fn coverage_ok(&self, tolerance: f64) -> bool {
        self.coverage.is_finite() && (self.coverage - self.nominal).abs() <= tolerance
    }

    /// Monte Carlo standard error of the coverage estimate.
    ///
    /// Reported so a near-miss can be told from a real one: with 1,000
    /// replications at 95% nominal this is about 0.7 percentage points, so a
    /// measured 93.5% is roughly two standard errors low and worth
    /// investigating, while 94.5% is noise.
    pub fn coverage_mcse(&self) -> f64 {
        if self.replications == 0 {
            return f64::NAN;
        }
        #[allow(clippy::cast_precision_loss)]
        let n = self.replications as f64;
        (self.coverage * (1.0 - self.coverage) / n).sqrt()
    }

    /// One-line summary.
    pub fn summary(&self) -> String {
        format!(
            "{:<28} n={:<5} coverage={:.1}% (±{:.1}, nominal {:.0}%) bias={:+.4} width={:.4}{}",
            self.model,
            self.replications,
            self.coverage * 100.0,
            self.coverage_mcse() * 100.0,
            self.nominal * 100.0,
            self.bias,
            self.mean_width,
            if self.n_failed > 0 {
                format!(" failed={}", self.n_failed)
            } else {
                String::new()
            }
        )
    }
}

/// Measure credible-interval coverage under repeated draws from the prior.
///
/// `trial` receives a seeded RNG and returns `(truth, point_estimate, lower,
/// upper)`: the parameter it drew from the prior, the posterior point estimate,
/// and the interval bounds. Returning `None` counts as a failure rather than a
/// miss.
///
/// The RNG is seeded per replication from `base_seed`, so a failing replication
/// is reproducible from the index alone.
pub fn credible_coverage<F>(
    model: &str,
    replications: usize,
    nominal: f64,
    base_seed: u64,
    mut trial: F,
) -> CalibrationReport
where
    F: FnMut(&mut ChaCha8Rng) -> Option<(f64, f64, f64, f64)>,
{
    let mut covered = 0usize;
    let mut total_width = 0.0;
    let mut total_error = 0.0;
    let mut n_ok = 0usize;
    let mut n_failed = 0usize;

    for k in 0..replications {
        let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(k as u64));
        match trial(&mut rng) {
            Some((truth, estimate, lo, hi))
                if truth.is_finite() && lo.is_finite() && hi.is_finite() =>
            {
                if truth >= lo && truth <= hi {
                    covered += 1;
                }
                total_width += hi - lo;
                total_error += estimate - truth;
                n_ok += 1;
            }
            _ => n_failed += 1,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let denom = n_ok as f64;
    CalibrationReport {
        model: model.to_string(),
        replications: n_ok,
        nominal,
        coverage: if n_ok == 0 {
            f64::NAN
        } else {
            #[allow(clippy::cast_precision_loss)]
            let c = covered as f64;
            c / denom
        },
        mean_width: if n_ok == 0 {
            f64::NAN
        } else {
            total_width / denom
        },
        bias: if n_ok == 0 {
            f64::NAN
        } else {
            total_error / denom
        },
        n_failed,
    }
}

/// Result of a simulation-based calibration run.
#[derive(Debug, Clone)]
pub struct SbcReport {
    /// What was measured.
    pub model: String,
    /// Counts of ranks falling in each bin.
    pub histogram: Vec<usize>,
    /// Replications contributing to the histogram.
    pub replications: usize,
    /// Chi-squared statistic against a uniform histogram.
    pub chi_squared: f64,
    /// Degrees of freedom, `bins - 1`.
    pub dof: usize,
    /// Upper-tail probability of `chi_squared`. Small values reject uniformity.
    pub p_value: f64,
}

impl SbcReport {
    /// Whether the rank histogram is consistent with uniform at `alpha`.
    pub fn is_calibrated(&self, alpha: f64) -> bool {
        self.p_value.is_finite() && self.p_value > alpha
    }

    /// Which way the sampler is miscalibrated, in words.
    ///
    /// Reads the histogram's shape rather than only its p-value, because
    /// "not uniform" is a much less useful message than "your posterior is too
    /// narrow".
    pub fn diagnosis(&self) -> &'static str {
        if self.histogram.len() < 3 || self.replications == 0 {
            return "too few bins or replications to diagnose";
        }
        let bins = self.histogram.len();
        #[allow(clippy::cast_precision_loss)]
        let expected = self.replications as f64 / bins as f64;

        let edge_bins = (bins / 4).max(1);
        #[allow(clippy::cast_precision_loss)]
        let low: f64 = self.histogram[..edge_bins].iter().sum::<usize>() as f64;
        #[allow(clippy::cast_precision_loss)]
        let high: f64 = self.histogram[bins - edge_bins..].iter().sum::<usize>() as f64;
        #[allow(clippy::cast_precision_loss)]
        let middle: f64 = self.histogram[edge_bins..bins - edge_bins]
            .iter()
            .sum::<usize>() as f64;

        #[allow(clippy::cast_precision_loss)]
        let edge_expected = expected * edge_bins as f64;
        #[allow(clippy::cast_precision_loss)]
        let middle_expected = expected * (bins - 2 * edge_bins) as f64;

        let edges_heavy = (low + high) > 1.25 * 2.0 * edge_expected;
        let middle_heavy = middle > 1.25 * middle_expected;
        let sloped = (high - low).abs() > 0.5 * edge_expected;

        if edges_heavy {
            "U-shaped: the posterior is TOO NARROW — intervals are overconfident"
        } else if middle_heavy {
            "peaked: the posterior is TOO WIDE — intervals are underconfident"
        } else if sloped {
            "sloped: the posterior is biased"
        } else {
            "uniform: calibrated"
        }
    }

    /// One-line summary.
    pub fn summary(&self) -> String {
        format!(
            "{:<28} n={:<5} chi2={:.1} (dof {}) p={:.3}  {}",
            self.model,
            self.replications,
            self.chi_squared,
            self.dof,
            self.p_value,
            self.diagnosis()
        )
    }
}

/// Run simulation-based calibration.
///
/// `trial` receives a seeded RNG and returns `(theta_star, posterior_draws)`:
/// the parameter drawn from the prior and samples from the posterior given data
/// generated by it. The rank of `theta_star` among those draws is accumulated
/// into a histogram which must be uniform.
///
/// `bins` should divide the draw count evenly for the histogram to be exactly
/// uniform under the null; a mismatch adds a small discretisation artefact.
pub fn simulation_based_calibration<F>(
    model: &str,
    replications: usize,
    bins: usize,
    base_seed: u64,
    mut trial: F,
) -> SbcReport
where
    F: FnMut(&mut ChaCha8Rng) -> Option<(f64, Vec<f64>)>,
{
    let bins = bins.max(2);
    let mut histogram = vec![0usize; bins];
    let mut n = 0usize;

    for k in 0..replications {
        let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(k as u64));
        let Some((theta, draws)) = trial(&mut rng) else {
            continue;
        };
        if draws.is_empty() || !theta.is_finite() {
            continue;
        }

        let rank = draws.iter().filter(|&&d| d < theta).count();
        // Map rank in 0..=draws.len() onto a bin. Using the draw count plus one
        // as the denominator keeps both extremes reachable.
        #[allow(clippy::cast_precision_loss)]
        let frac = rank as f64 / (draws.len() + 1) as f64;
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let bin = ((frac * bins as f64) as usize).min(bins - 1);
        histogram[bin] += 1;
        n += 1;
    }

    #[allow(clippy::cast_precision_loss)]
    let expected = n as f64 / bins as f64;
    let chi_squared = if expected > 0.0 {
        histogram
            .iter()
            .map(|&o| {
                #[allow(clippy::cast_precision_loss)]
                let o = o as f64;
                (o - expected).powi(2) / expected
            })
            .sum()
    } else {
        f64::NAN
    };
    let dof = bins - 1;

    SbcReport {
        model: model.to_string(),
        histogram,
        replications: n,
        chi_squared,
        dof,
        p_value: chi_squared_upper_tail(chi_squared, dof),
    }
}

/// Draw one sample from a Beta(a, b) distribution.
///
/// Johnk's algorithm for small parameters, Cheng's BB/BC otherwise. Provided
/// here so calibration trials can draw a Bernoulli success probability from a
/// Beta prior without pulling a distributions crate into this module's callers.
pub fn sample_beta(rng: &mut ChaCha8Rng, a: f64, b: f64) -> f64 {
    // Beta(a,b) = X / (X + Y) with X ~ Gamma(a,1), Y ~ Gamma(b,1). Using two
    // gamma draws is slower than a specialised Beta sampler but correct for all
    // parameter ranges, which matters more here than speed.
    let x = sample_gamma(rng, a);
    let y = sample_gamma(rng, b);
    if x + y <= 0.0 { 0.5 } else { x / (x + y) }
}

/// Draw one sample from a Gamma(shape, 1) distribution.
///
/// Marsaglia–Tsang, with the standard boost for `shape < 1`.
pub fn sample_gamma(rng: &mut ChaCha8Rng, shape: f64) -> f64 {
    if shape <= 0.0 {
        return 0.0;
    }
    if shape < 1.0 {
        // Boost: Gamma(a) = Gamma(a+1) * U^(1/a).
        let g = sample_gamma(rng, shape + 1.0);
        let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        return g * u.powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let z = sample_standard_normal(rng);
        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// Draw one standard normal sample by the Box–Muller transform.
pub fn sample_standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Upper-tail probability `P(X > x)` for `X ~ chi-squared(dof)`.
///
/// Wilson–Hilferty cube-root transform: `(X/k)^(1/3)` is close to normal with
/// mean `1 - 2/(9k)` and variance `2/(9k)`. Accurate to about three decimal
/// places for `dof >= 3`, which is well inside what a calibration verdict
/// needs, and avoids a dependency on a special-functions crate.
fn chi_squared_upper_tail(x: f64, dof: usize) -> f64 {
    if !x.is_finite() || dof == 0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let k = dof as f64;
    let t = (x / k).cbrt();
    let mean = 1.0 - 2.0 / (9.0 * k);
    let sd = (2.0 / (9.0 * k)).sqrt();
    let z = (t - mean) / sd;
    1.0 - standard_normal_cdf(z)
}

/// Standard normal CDF via Abramowitz–Stegun 26.2.17.
///
/// Absolute error below 7.5e-8, which is far beyond what a p-value threshold
/// needs.
fn standard_normal_cdf(z: f64) -> f64 {
    const P: f64 = 0.231_641_9;
    const B: [f64; 5] = [
        0.319_381_530,
        -0.356_563_782,
        1.781_477_937,
        -1.821_255_978,
        1.330_274_429,
    ];

    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let z_abs = z.abs();
    let t = 1.0 / (1.0 + P * z_abs);
    let poly = B[0] * t + B[1] * t.powi(2) + B[2] * t.powi(3) + B[3] * t.powi(4) + B[4] * t.powi(5);
    let pdf = (-0.5 * z_abs * z_abs).exp() / (std::f64::consts::TAU).sqrt();
    let tail = pdf * poly;
    if sign > 0.0 { 1.0 - tail } else { tail }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_cdf_matches_known_values() {
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-9);
        assert!((standard_normal_cdf(1.959_963_985) - 0.975).abs() < 1e-6);
        assert!((standard_normal_cdf(-1.959_963_985) - 0.025).abs() < 1e-6);
    }

    #[test]
    fn chi_squared_tail_matches_tabulated_critical_values() {
        // chi2(9) upper 5% critical value is 16.919; the tail there is 0.05.
        assert!((chi_squared_upper_tail(16.919, 9) - 0.05).abs() < 0.005);
        // chi2(19) upper 5% is 30.144.
        assert!((chi_squared_upper_tail(30.144, 19) - 0.05).abs() < 0.005);
        // A statistic at its own dof sits near the middle of the distribution.
        let p = chi_squared_upper_tail(9.0, 9);
        assert!((0.3..0.6).contains(&p), "p was {p}");
    }

    #[test]
    fn beta_sampler_has_the_right_mean() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let (a, b) = (2.0, 5.0);
        let n = 20_000;
        let mean: f64 = (0..n).map(|_| sample_beta(&mut rng, a, b)).sum::<f64>() / f64::from(n);
        // E[Beta(2,5)] = 2/7.
        assert!((mean - 2.0 / 7.0).abs() < 0.01, "mean was {mean}");
    }

    #[test]
    fn gamma_sampler_handles_shape_below_one() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let n = 20_000;
        let mean: f64 = (0..n).map(|_| sample_gamma(&mut rng, 0.5)).sum::<f64>() / f64::from(n);
        // E[Gamma(0.5, 1)] = 0.5.
        assert!((mean - 0.5).abs() < 0.02, "mean was {mean}");
    }

    #[test]
    fn standard_normal_sampler_is_standard() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let n = 50_000;
        let xs: Vec<f64> = (0..n).map(|_| sample_standard_normal(&mut rng)).collect();
        #[allow(clippy::cast_precision_loss)]
        let m = n as f64;
        let mean = xs.iter().sum::<f64>() / m;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m - 1.0);
        assert!(mean.abs() < 0.02, "mean was {mean}");
        assert!((var - 1.0).abs() < 0.03, "variance was {var}");
    }

    /// The harness must reject a sampler that is deliberately overconfident.
    ///
    /// Without this, a green SBC report would only prove the harness is
    /// incapable of complaining.
    #[test]
    fn sbc_detects_a_posterior_that_is_too_narrow() {
        let report = simulation_based_calibration("too-narrow", 400, 10, 7, |rng| {
            let theta = sample_standard_normal(rng);
            // Posterior draws clustered far too tightly around the truth: the
            // rank of theta lands at an extreme almost every time.
            let draws: Vec<f64> = (0..99)
                .map(|_| theta + 0.01 * sample_standard_normal(rng))
                .collect();
            // Shift so theta is systematically outside the cluster.
            let shifted: Vec<f64> = draws.iter().map(|d| d + 0.5).collect();
            Some((theta, shifted))
        });
        assert!(
            !report.is_calibrated(0.01),
            "harness failed to reject an obviously wrong posterior: {}",
            report.summary()
        );
    }

    /// And must accept a correct one.
    #[test]
    fn sbc_accepts_an_exact_posterior() {
        // Conjugate normal with known variance: prior N(0,1), one observation
        // with variance 1, so the posterior is N(y/2, 1/2) exactly.
        let report = simulation_based_calibration("exact-normal", 2_000, 20, 11, |rng| {
            let theta = sample_standard_normal(rng);
            let y = theta + sample_standard_normal(rng);
            let post_mean = y / 2.0;
            let post_sd = (0.5_f64).sqrt();
            let draws: Vec<f64> = (0..99)
                .map(|_| post_mean + post_sd * sample_standard_normal(rng))
                .collect();
            Some((theta, draws))
        });
        assert!(
            report.is_calibrated(0.01),
            "exact posterior was rejected: {}",
            report.summary()
        );
        assert_eq!(report.diagnosis(), "uniform: calibrated");
    }

    /// The coverage harness must notice an interval that is too narrow.
    #[test]
    fn credible_coverage_detects_a_narrow_interval() {
        let report = credible_coverage("narrow", 1_000, 0.95, 13, |rng| {
            let theta = sample_standard_normal(rng);
            let y = theta + sample_standard_normal(rng);
            let post_mean = y / 2.0;
            let post_sd = (0.5_f64).sqrt();
            // Half the width it should be.
            let half = 0.5 * 1.96 * post_sd;
            Some((theta, post_mean, post_mean - half, post_mean + half))
        });
        assert!(
            !report.coverage_ok(0.03),
            "harness accepted a half-width interval: {}",
            report.summary()
        );
        assert!(report.coverage < 0.85, "{}", report.summary());
    }

    /// And must accept an exact one.
    #[test]
    fn credible_coverage_accepts_an_exact_interval() {
        let report = credible_coverage("exact", 2_000, 0.95, 17, |rng| {
            let theta = sample_standard_normal(rng);
            let y = theta + sample_standard_normal(rng);
            let post_mean = y / 2.0;
            let post_sd = (0.5_f64).sqrt();
            let half = 1.959_963_985 * post_sd;
            Some((theta, post_mean, post_mean - half, post_mean + half))
        });
        assert!(
            report.coverage_ok(0.02),
            "exact interval was rejected: {}",
            report.summary()
        );
    }

    #[test]
    fn coverage_mcse_shrinks_with_replications() {
        // A non-degenerate interval: coverage must be strictly between 0 and 1,
        // or the binomial standard error is zero at every sample size and the
        // comparison is vacuous.
        let trial = |rng: &mut ChaCha8Rng| {
            let theta = sample_standard_normal(rng);
            let y = theta + sample_standard_normal(rng);
            let post_mean = y / 2.0;
            let half = 1.959_963_985 * (0.5_f64).sqrt();
            Some((theta, post_mean, post_mean - half, post_mean + half))
        };
        let small = credible_coverage("s", 200, 0.95, 1, trial);
        let large = credible_coverage("l", 20_000, 0.95, 1, trial);

        assert!(
            small.coverage > 0.0 && small.coverage < 1.0,
            "{}",
            small.summary()
        );
        assert!(
            large.coverage_mcse() < small.coverage_mcse(),
            "MCSE must fall with replications: {} vs {}",
            small.coverage_mcse(),
            large.coverage_mcse()
        );
    }
}

/// Regularised incomplete beta function `I_x(a, b)` — the Beta CDF.
///
/// Lentz's continued fraction, with the standard symmetry reflection to keep
/// convergence fast on both sides. Provided as a **reference implementation**:
/// its purpose is to let a test distinguish "this approximation is wrong" from
/// "any interval behaves this way here", which is a distinction that cannot be
/// drawn without an exact answer to compare against.
pub fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp();

    // The continued fraction converges quickly only for x < (a+1)/(a+b+2);
    // reflect otherwise.
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - beta_cdf(1.0 - x, b, a);
    }

    // Modified Lentz.
    const TINY: f64 = 1e-30;
    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 0.0;

    for i in 0..=300 {
        let m = i / 2;
        #[allow(clippy::cast_precision_loss)]
        let m_f = m as f64;
        #[allow(clippy::cast_precision_loss)]
        let i_f = i as f64;

        let numerator = if i == 0 {
            1.0
        } else if i % 2 == 0 {
            (m_f * (b - m_f) * x) / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f))
        } else {
            let _ = i_f;
            -((a + m_f) * (a + b + m_f) * x) / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0))
        };

        d = 1.0 + numerator * d;
        if d.abs() < TINY {
            d = TINY;
        }
        d = 1.0 / d;

        c = 1.0 + numerator / c;
        if c.abs() < TINY {
            c = TINY;
        }

        let cd = c * d;
        f *= cd;

        if (1.0 - cd).abs() < 1e-12 {
            break;
        }
    }

    front * (f - 1.0) / a
}

/// Quantile of a Beta(a, b) distribution, by bisection on [`beta_cdf`].
///
/// Bisection rather than Newton because it cannot diverge, and 200 iterations
/// of bisection on `[0, 1]` reaches machine precision. This is reference code
/// used to judge an approximation, so robustness matters more than speed.
pub fn beta_quantile(p: f64, a: f64, b: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }

    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if beta_cdf(mid, a, b) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Exact equal-tailed credible interval for a Beta posterior.
///
/// The reference against which a normal approximation is judged.
pub fn beta_credible_interval(a: f64, b: f64, level: f64) -> (f64, f64) {
    let tail = (1.0 - level) / 2.0;
    (beta_quantile(tail, a, b), beta_quantile(1.0 - tail, a, b))
}

/// Log-gamma via the Lanczos approximation, g = 7, n = 9.
///
/// Relative error below 1e-13 for positive arguments, which is well beyond what
/// a quantile bisection needs.
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection: ln G(x) = ln(pi / sin(pi x)) - ln G(1-x).
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, &g) in G.iter().enumerate().skip(1) {
        #[allow(clippy::cast_precision_loss)]
        let i_f = i as f64;
        a += g / (x + i_f);
    }
    0.5 * (std::f64::consts::TAU).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

#[cfg(test)]
mod beta_reference_tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_known_values() {
        // G(1) = 1, G(2) = 1, G(5) = 24, G(0.5) = sqrt(pi).
        assert!(ln_gamma(1.0).abs() < 1e-10);
        assert!(ln_gamma(2.0).abs() < 1e-10);
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-10);
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-10);
    }

    #[test]
    fn beta_cdf_matches_closed_forms() {
        // Beta(1,1) is uniform: CDF(x) = x.
        for x in [0.1, 0.25, 0.5, 0.9] {
            assert!((beta_cdf(x, 1.0, 1.0) - x).abs() < 1e-9, "at {x}");
        }
        // Beta(3,1): CDF(x) = x^3.
        for x in [0.2_f64, 0.5, 0.8] {
            assert!((beta_cdf(x, 3.0, 1.0) - x.powi(3)).abs() < 1e-9, "at {x}");
        }
        // Beta(1,3): CDF(x) = 1 - (1-x)^3.
        for x in [0.2_f64, 0.5, 0.8] {
            let expected = 1.0 - (1.0 - x).powi(3);
            assert!((beta_cdf(x, 1.0, 3.0) - expected).abs() < 1e-9, "at {x}");
        }
    }

    #[test]
    fn beta_cdf_is_symmetric_for_equal_parameters() {
        for a in [0.5, 1.0, 3.0, 10.0] {
            assert!(
                (beta_cdf(0.5, a, a) - 0.5).abs() < 1e-9,
                "Beta({a},{a}) median should be 0.5"
            );
        }
    }

    #[test]
    fn beta_quantile_inverts_the_cdf() {
        for (a, b) in [(1.0, 1.0), (3.0, 1.0), (2.0, 5.0), (10.0, 10.0), (0.5, 0.5)] {
            for p in [0.025, 0.1, 0.5, 0.9, 0.975] {
                let x = beta_quantile(p, a, b);
                let back = beta_cdf(x, a, b);
                assert!(
                    (back - p).abs() < 1e-8,
                    "Beta({a},{b}) quantile({p}) = {x}, cdf back = {back}"
                );
            }
        }
    }

    #[test]
    fn exact_interval_matches_a_hand_computed_case() {
        // Beta(3,1) has CDF x^3, so the equal-tailed 95% interval is
        // (0.025^(1/3), 0.975^(1/3)).
        let (lo, hi) = beta_credible_interval(3.0, 1.0, 0.95);
        assert!((lo - 0.025_f64.cbrt()).abs() < 1e-8, "lo was {lo}");
        assert!((hi - 0.975_f64.cbrt()).abs() < 1e-8, "hi was {hi}");
    }

    #[test]
    fn exact_interval_is_inside_the_unit_range() {
        // The property a clamped normal approximation cannot guarantee without
        // discarding tail mass.
        for (a, b) in [(1.0, 20.0), (20.0, 1.0), (1.0, 1.0), (0.5, 0.5)] {
            let (lo, hi) = beta_credible_interval(a, b, 0.95);
            assert!((0.0..=1.0).contains(&lo), "Beta({a},{b}) lo = {lo}");
            assert!((0.0..=1.0).contains(&hi), "Beta({a},{b}) hi = {hi}");
            assert!(lo < hi);
        }
    }
}
