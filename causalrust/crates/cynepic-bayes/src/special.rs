//! Special functions needed for exact posterior quantiles.
//!
//! # Why this exists rather than a dependency
//!
//! `statrs` is already in the workspace and provides all of this. It is not used
//! here because `cynepic-bayes` is one of the crates gated on `wasm32-wasip1` in
//! CI, and `statrs` pulls `nalgebra`, which is both large and a recurring source
//! of platform trouble (it is the reason this workspace cannot be built on
//! Windows MSVC without Spectre-mitigated libraries). Three hundred lines of
//! well-tested numerics is a better trade than a transitive linear-algebra
//! stack for two functions.
//!
//! Everything here is checked against closed forms — `Beta(1,1)` is uniform,
//! `Beta(3,1)` has CDF `x^3`, `Beta(1,3)` has CDF `1-(1-x)^3` — and
//! cross-checked against `cynepic-testkit`'s independently written reference in
//! the calibration suite.

/// Log-gamma via the Lanczos approximation (g = 7, n = 9).
///
/// Relative error below 1e-13 for positive arguments.
pub fn ln_gamma(x: f64) -> f64 {
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
        // Reflection formula, since the series above is only valid for x >= 0.5.
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
    0.5 * std::f64::consts::TAU.ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularised incomplete beta function `I_x(a, b)` — the Beta CDF.
///
/// Modified Lentz continued fraction. Reflected when `x` is past the point
/// where convergence slows, which keeps the iteration count bounded across the
/// whole parameter range rather than only the easy part of it.
pub fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - beta_cdf(1.0 - x, b, a);
    }

    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp();

    const TINY: f64 = 1e-30;
    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 0.0;

    for i in 0..=300 {
        let m = i / 2;
        #[allow(clippy::cast_precision_loss)]
        let m_f = m as f64;

        let numerator = if i == 0 {
            1.0
        } else if i % 2 == 0 {
            (m_f * (b - m_f) * x) / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f))
        } else {
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

/// Quantile of `Beta(a, b)`: the `p`-th percentile.
///
/// Bisection on [`beta_cdf`]. Chosen over Newton because it cannot diverge on
/// the extreme parameter values a reliability tracker actually sees — a tool
/// with 200 successes and no failures gives `Beta(201, 1)`, where the density
/// is concentrated in the last thousandth of the range and a Newton step
/// overshoots out of `[0, 1]`.
///
/// 200 bisections narrow `[0, 1]` past `f64` resolution, so accuracy is limited
/// by [`beta_cdf`] rather than by the search.
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

/// Quantile of `Gamma(shape, rate)`.
///
/// Bisection on the regularised lower incomplete gamma, with a bracket that
/// expands until it contains the target rather than being assumed.
pub fn gamma_quantile(p: f64, shape: f64, rate: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) || shape <= 0.0 || rate <= 0.0 {
        return f64::NAN;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Gamma is unbounded above, so find a bracket before bisecting.
    let mut hi = shape / rate;
    if hi <= 0.0 {
        hi = 1.0;
    }
    let mut guard = 0;
    while gamma_cdf(hi, shape, rate) < p && guard < 200 {
        hi *= 2.0;
        guard += 1;
    }

    let mut lo = 0.0_f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if gamma_cdf(mid, shape, rate) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Regularised lower incomplete gamma `P(shape, rate*x)` — the Gamma CDF.
///
/// Series expansion below the transition point, continued fraction above,
/// which is the standard split: each converges quickly only on its own side.
pub fn gamma_cdf(x: f64, shape: f64, rate: f64) -> f64 {
    if shape <= 0.0 || rate <= 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    let z = x * rate;

    if z < shape + 1.0 {
        // Series.
        let mut term = 1.0 / shape;
        let mut sum = term;
        for n in 1..1_000 {
            #[allow(clippy::cast_precision_loss)]
            let n_f = f64::from(n);
            term *= z / (shape + n_f);
            sum += term;
            if term.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        (sum.ln() - z + shape * z.ln() - ln_gamma(shape)).exp()
    } else {
        // Continued fraction for the upper tail, then complement.
        const TINY: f64 = 1e-300;
        let mut b = z + 1.0 - shape;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1_000 {
            #[allow(clippy::cast_precision_loss)]
            let i_f = f64::from(i);
            let an = -i_f * (i_f - shape);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        let upper = (-z + shape * z.ln() - ln_gamma(shape)).exp() * h;
        1.0 - upper
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_factorials() {
        assert!(ln_gamma(1.0).abs() < 1e-10);
        assert!(ln_gamma(2.0).abs() < 1e-10);
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-10);
        assert!((ln_gamma(10.0) - 362_880.0_f64.ln()).abs() < 1e-9);
        // G(1/2) = sqrt(pi), which exercises the reflection branch.
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-10);
    }

    #[test]
    fn beta_cdf_matches_closed_forms() {
        for x in [0.1_f64, 0.25, 0.5, 0.9] {
            // Beta(1,1) is uniform.
            assert!((beta_cdf(x, 1.0, 1.0) - x).abs() < 1e-9, "uniform at {x}");
            // Beta(3,1) has CDF x^3.
            assert!(
                (beta_cdf(x, 3.0, 1.0) - x.powi(3)).abs() < 1e-9,
                "Beta(3,1) at {x}"
            );
            // Beta(1,3) has CDF 1-(1-x)^3.
            assert!(
                (beta_cdf(x, 1.0, 3.0) - (1.0 - (1.0 - x).powi(3))).abs() < 1e-9,
                "Beta(1,3) at {x}"
            );
        }
    }

    #[test]
    fn beta_cdf_is_symmetric_for_equal_parameters() {
        for a in [0.5, 1.0, 3.0, 10.0, 100.0] {
            assert!((beta_cdf(0.5, a, a) - 0.5).abs() < 1e-9, "Beta({a},{a})");
        }
    }

    #[test]
    fn beta_quantile_inverts_the_cdf() {
        for (a, b) in [
            (1.0, 1.0),
            (3.0, 1.0),
            (2.0, 5.0),
            (10.0, 10.0),
            (0.5, 0.5),
            (201.0, 1.0),
        ] {
            for p in [0.025, 0.1, 0.5, 0.9, 0.975] {
                let x = beta_quantile(p, a, b);
                assert!(
                    (beta_cdf(x, a, b) - p).abs() < 1e-8,
                    "Beta({a},{b}) quantile({p}) = {x}"
                );
            }
        }
    }

    #[test]
    fn beta_quantile_handles_the_perfect_record_case() {
        // 200 successes, no failures: Beta(201, 1). A tool-reliability tracker
        // reaches this within a day, and it is where a Newton solver leaves the
        // unit interval.
        let (lo, hi) = (
            beta_quantile(0.025, 201.0, 1.0),
            beta_quantile(0.975, 201.0, 1.0),
        );
        assert!((0.0..1.0).contains(&lo), "lo = {lo}");
        assert!(lo < hi && hi <= 1.0, "[{lo}, {hi}]");
        // Closed form for Beta(n,1): CDF = x^n, so the 2.5% quantile is
        // 0.025^(1/201).
        assert!((lo - 0.025_f64.powf(1.0 / 201.0)).abs() < 1e-9);
    }

    #[test]
    fn gamma_cdf_matches_the_exponential_special_case() {
        // Gamma(1, rate) is Exponential(rate): CDF = 1 - exp(-rate x).
        for rate in [0.5_f64, 1.0, 3.0] {
            for x in [0.1_f64, 1.0, 5.0] {
                let expected: f64 = 1.0 - (-rate * x).exp();
                assert!(
                    (gamma_cdf(x, 1.0, rate) - expected).abs() < 1e-9,
                    "Gamma(1,{rate}) at {x}"
                );
            }
        }
    }

    #[test]
    fn gamma_cdf_matches_chi_squared_two_dof() {
        // chi2(2) = Gamma(shape 1, rate 1/2): CDF = 1 - exp(-x/2).
        for x in [0.5_f64, 2.0, 6.0] {
            let expected: f64 = 1.0 - (-x / 2.0).exp();
            assert!((gamma_cdf(x, 1.0, 0.5) - expected).abs() < 1e-9, "at {x}");
        }
    }

    #[test]
    fn gamma_quantile_inverts_the_cdf() {
        for (shape, rate) in [(1.0, 1.0), (2.0, 3.0), (0.5, 1.0), (20.0, 2.0)] {
            for p in [0.025, 0.5, 0.975] {
                let x = gamma_quantile(p, shape, rate);
                assert!(
                    (gamma_cdf(x, shape, rate) - p).abs() < 1e-8,
                    "Gamma({shape},{rate}) quantile({p}) = {x}"
                );
            }
        }
    }

    #[test]
    fn quantiles_are_monotone_in_p() {
        let mut previous = -1.0;
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let q = beta_quantile(p, 2.0, 5.0);
            assert!(q > previous, "not monotone at p={p}");
            previous = q;
        }
    }

    #[test]
    fn invalid_parameters_yield_nan_not_a_number_that_looks_real() {
        assert!(beta_quantile(0.5, -1.0, 1.0).is_nan());
        assert!(beta_quantile(0.5, 1.0, 0.0).is_nan());
        assert!(beta_cdf(0.5, 0.0, 1.0).is_nan());
        assert!(gamma_quantile(0.5, -1.0, 1.0).is_nan());
    }
}
