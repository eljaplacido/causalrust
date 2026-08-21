//! Metamorphic relations — properties any correct estimator must satisfy.
//!
//! Ordinary tests need a known answer. Metamorphic tests need only a known
//! *relationship* between two runs, which makes them usable where no ground
//! truth exists — including on real data.
//!
//! Each relation below states a transformation of the input and the change it
//! must produce in the output. A violation is a bug, with no reference
//! implementation required to prove it.

use ndarray::{Array1, Array2};

/// Multiply every outcome by `c`. A correct effect estimate scales by exactly `c`.
///
/// Catches: unit-handling errors, hard-coded thresholds, and any place a
/// magnitude is compared against an absolute constant rather than a relative one
/// — which is the shape of finding C7's `< 0.15` verdict rule.
pub fn scale_outcome(outcome: &Array1<f64>, c: f64) -> Array1<f64> {
    outcome.mapv(|y| y * c)
}

/// Add a constant to every outcome. The effect estimate must not change at all.
///
/// Catches: a missing intercept, or an estimator sensitive to the outcome's
/// location rather than its differences.
pub fn shift_outcome(outcome: &Array1<f64>, k: f64) -> Array1<f64> {
    outcome.mapv(|y| y + k)
}

/// Reverse the unit ordering. Every estimate must be invariant.
///
/// Catches: order-dependent accumulation, and RNG state leaking into a result
/// that is supposed to depend only on the data.
pub fn reverse_units(
    treatment: &Array1<f64>,
    outcome: &Array1<f64>,
    covariates: &Array2<f64>,
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let n = treatment.len();
    let idx: Vec<usize> = (0..n).rev().collect();
    permute(treatment, outcome, covariates, &idx)
}

/// Duplicate the dataset. The point estimate must be unchanged; the standard
/// error must shrink by a factor of √2.
///
/// Catches: standard errors that ignore the sample size, or a variance formula
/// inconsistent with its own point estimate — precisely finding C5. This one is
/// unusually good at exposing that class of bug, because the point estimate and
/// the SE respond to duplication in *different* ways, so an inconsistent pair
/// cannot satisfy both halves at once.
pub fn duplicate(
    treatment: &Array1<f64>,
    outcome: &Array1<f64>,
    covariates: &Array2<f64>,
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let n = treatment.len();
    let idx: Vec<usize> = (0..n).chain(0..n).collect();
    permute(treatment, outcome, covariates, &idx)
}

/// Append a covariate of pure noise, independent of everything.
///
/// The estimate must move by less than Monte Carlo error. A large move means the
/// estimator is fitting noise, and the "random common cause" refutation should
/// have caught it.
pub fn append_noise_covariate(covariates: &Array2<f64>, noise: &Array1<f64>) -> Array2<f64> {
    let n = covariates.nrows();
    let p = covariates.ncols();
    let mut out = Array2::zeros((n, p + 1));
    for i in 0..n {
        for j in 0..p {
            out[[i, j]] = covariates[[i, j]];
        }
        out[[i, p]] = noise[i];
    }
    out
}

/// Duplicate an existing covariate column exactly.
///
/// The design matrix becomes rank deficient. A correct estimator returns a named
/// error identifying the aliased columns; the current one returns
/// `ate: 0.0, std_error: 0.0` — finding C1.
pub fn duplicate_covariate(covariates: &Array2<f64>, col: usize) -> Array2<f64> {
    let n = covariates.nrows();
    let p = covariates.ncols();
    let mut out = Array2::zeros((n, p + 1));
    for i in 0..n {
        for j in 0..p {
            out[[i, j]] = covariates[[i, j]];
        }
        out[[i, p]] = covariates[[i, col]];
    }
    out
}

/// Swap treatment and control labels. The effect must flip sign, same magnitude.
///
/// Catches: asymmetric handling of the two arms — a surprisingly common bug in
/// weighting code, where the treated and control branches are written separately
/// and drift apart.
pub fn flip_treatment(treatment: &Array1<f64>) -> Array1<f64> {
    treatment.mapv(|t| if t > 0.5 { 0.0 } else { 1.0 })
}

fn permute(
    treatment: &Array1<f64>,
    outcome: &Array1<f64>,
    covariates: &Array2<f64>,
    idx: &[usize],
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let m = idx.len();
    let p = covariates.ncols();
    let t = Array1::from_iter(idx.iter().map(|&i| treatment[i]));
    let y = Array1::from_iter(idx.iter().map(|&i| outcome[i]));
    let mut x = Array2::zeros((m, p));
    for (r, &i) in idx.iter().enumerate() {
        for j in 0..p {
            x[[r, j]] = covariates[[i, j]];
        }
    }
    (t, y, x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dgp::Dgp;

    #[test]
    fn scaling_outcome_scales_values() {
        let d = Dgp::new().with_n(20).sample(1);
        let scaled = scale_outcome(&d.outcome, 3.0);
        for i in 0..d.n() {
            assert!((scaled[i] - d.outcome[i] * 3.0).abs() < 1e-12);
        }
    }

    #[test]
    fn duplicate_doubles_the_sample() {
        let d = Dgp::new().with_n(30).sample(2);
        let (t, y, x) = duplicate(&d.treatment, &d.outcome, &d.covariates);
        assert_eq!(t.len(), 60);
        assert_eq!(y.len(), 60);
        assert_eq!(x.nrows(), 60);
        // Second half mirrors the first.
        assert!((t[0] - t[30]).abs() < 1e-12);
        assert!((y[5] - y[35]).abs() < 1e-12);
    }

    #[test]
    fn reversing_preserves_the_multiset() {
        let d = Dgp::new().with_n(25).sample(3);
        let (t, y, _) = reverse_units(&d.treatment, &d.outcome, &d.covariates);
        assert!((t.sum() - d.treatment.sum()).abs() < 1e-12);
        assert!((y.sum() - d.outcome.sum()).abs() < 1e-12);
    }

    #[test]
    fn flipping_treatment_swaps_the_arms() {
        let d = Dgp::new().with_n(40).sample(4);
        let flipped = flip_treatment(&d.treatment);
        let before = d.treatment.iter().filter(|&&t| t > 0.5).count();
        let after = flipped.iter().filter(|&&t| t > 0.5).count();
        assert_eq!(before + after, d.n());
    }

    #[test]
    fn duplicating_a_covariate_makes_the_design_rank_deficient() {
        let d = Dgp::new().with_p(3).with_n(20).sample(5);
        let x = duplicate_covariate(&d.covariates, 1);
        assert_eq!(x.ncols(), 4);
        for i in 0..d.n() {
            assert!((x[[i, 3]] - x[[i, 1]]).abs() < 1e-12);
        }
    }
}
