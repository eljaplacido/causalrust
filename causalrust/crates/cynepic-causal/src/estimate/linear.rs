//! Least-squares treatment-effect estimation.
//!
//! # What changed and why
//!
//! The previous solver was Gaussian elimination that returned a zero vector
//! when it met a small pivot, and a companion inverter that returned a zero
//! matrix on the same condition. Neither signalled. On a rank-deficient design
//! the caller received a plausible ATE with a standard error of exactly 0.0 —
//! infinite confidence, which passes every downstream significance test
//! (finding C1).
//!
//! This module uses **Householder QR with column pivoting**, which does not
//! merely tolerate rank deficiency but detects and reports it: the pivoting
//! order moves dependent columns to the end, so they can be named. Callers get
//! [`EstimationError::RankDeficient`] listing the aliased columns instead of a
//! number.
//!
//! # Standard errors
//!
//! Three estimators, chosen explicitly rather than by default:
//!
//! - [`StdErrorKind::Classical`] assumes homoskedasticity. Fast, and wrong
//!   whenever error variance depends on the covariates.
//! - [`StdErrorKind::Hc1`] is White's estimator with the `n/(n-k)` correction.
//! - [`StdErrorKind::Hc3`] divides each squared residual by `(1 - h_ii)^2`.
//!   Preferred at small `n`, where high-leverage points shrink their own
//!   residuals and make HC1 under-cover.
//!
//! The default is HC1. Classical is available because it is right under its
//! assumption and cheaper, but it must be asked for.

use ndarray::{Array1, Array2};

use crate::error::EstimationError;
use crate::estimand::{ATEResult, Diagnostics, Estimand, StdErrorKind};

/// Relative tolerance for declaring a QR pivot numerically zero.
///
/// Scaled by the largest pivot, so the test is on the condition number rather
/// than on absolute magnitude — otherwise the same design in different units
/// would get different answers.
const RANK_TOL: f64 = 1e-10;

/// Ordinary-least-squares treatment-effect estimation.
#[derive(Debug, Clone, Copy, Default)]
pub struct LinearATEEstimator;

impl LinearATEEstimator {
    /// Difference in means: `E[Y|T=1] - E[Y|T=0]`.
    ///
    /// Unbiased for the ATE only under random assignment. With observational
    /// data this is the confounded baseline that adjustment is meant to improve
    /// on — useful as a comparison, not as an answer.
    ///
    /// # Errors
    ///
    /// - [`EstimationError::LengthMismatch`] if the inputs disagree on `n`.
    /// - [`EstimationError::NoObservations`] if `n == 0`.
    /// - [`EstimationError::EmptyArm`] if either arm is empty. Previously the
    ///   missing arm's mean was taken as `0.0`, so a single-arm dataset returned
    ///   a confident effect equal to the other arm's mean (finding C10).
    pub fn difference_in_means(
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
    ) -> Result<ATEResult, EstimationError> {
        check_lengths("treatment", treatment.len(), "outcome", outcome.len())?;
        let n = treatment.len();
        if n == 0 {
            return Err(EstimationError::NoObservations);
        }

        let mut sum_t = 0.0;
        let mut sum_c = 0.0;
        let mut n_t = 0usize;
        let mut n_c = 0usize;
        for i in 0..n {
            if treatment[i] > 0.5 {
                sum_t += outcome[i];
                n_t += 1;
            } else {
                sum_c += outcome[i];
                n_c += 1;
            }
        }

        if n_t == 0 || n_c == 0 {
            return Err(EstimationError::EmptyArm {
                arm: if n_t == 0 { "treated" } else { "control" },
                n_treated: n_t,
                n_control: n_c,
            });
        }

        let mean_t = sum_t / n_t as f64;
        let mean_c = sum_c / n_c as f64;

        let mut ss_t = 0.0;
        let mut ss_c = 0.0;
        for i in 0..n {
            if treatment[i] > 0.5 {
                ss_t += (outcome[i] - mean_t).powi(2);
            } else {
                ss_c += (outcome[i] - mean_c).powi(2);
            }
        }

        // Welch, not pooled: the two arms have no reason to share a variance,
        // and assuming they do understates the standard error exactly when the
        // treatment affects dispersion as well as location.
        let std_error = if n_t > 1 && n_c > 1 {
            let v_t = ss_t / (n_t - 1) as f64;
            let v_c = ss_c / (n_c - 1) as f64;
            (v_t / n_t as f64 + v_c / n_c as f64).sqrt()
        } else {
            // One unit in an arm carries no information about variance. NaN is
            // the honest answer; `confidence_interval` returns None for it.
            f64::NAN
        };

        ATEResult::new(
            mean_t - mean_c,
            std_error,
            n,
            Estimand::Ate,
            StdErrorKind::Classical,
            Diagnostics {
                arm_sizes: Some((n_t, n_c)),
                ..Diagnostics::default()
            },
        )
    }

    /// OLS with covariate adjustment, using HC1 robust standard errors.
    ///
    /// Fits `Y = b0 + b1*T + X*b2` and reports `b1`.
    ///
    /// # Errors
    ///
    /// See [`Self::ols_adjusted_with`], which this delegates to.
    pub fn ols_adjusted(
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
        covariates: &Array2<f64>,
    ) -> Result<ATEResult, EstimationError> {
        Self::ols_adjusted_with(treatment, outcome, covariates, StdErrorKind::Hc1)
    }

    /// OLS with covariate adjustment and an explicit standard-error estimator.
    ///
    /// # Errors
    ///
    /// - [`EstimationError::LengthMismatch`], [`EstimationError::NoObservations`].
    /// - [`EstimationError::InsufficientData`] if `n <= p + 2`.
    /// - [`EstimationError::ConstantTreatment`] if `T` never varies.
    /// - [`EstimationError::RankDeficient`] if any design column is a linear
    ///   combination of the others, naming the aliased columns (finding C1).
    pub fn ols_adjusted_with(
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
        covariates: &Array2<f64>,
        se_kind: StdErrorKind,
    ) -> Result<ATEResult, EstimationError> {
        check_lengths("treatment", treatment.len(), "outcome", outcome.len())?;
        check_lengths(
            "treatment",
            treatment.len(),
            "covariates",
            covariates.nrows(),
        )?;

        let n = treatment.len();
        if n == 0 {
            return Err(EstimationError::NoObservations);
        }

        let p = covariates.ncols();
        let cols = p + 2; // intercept, treatment, covariates
        if n <= cols {
            return Err(EstimationError::InsufficientData { n, p: cols });
        }

        // A constant treatment column is collinear with the intercept, so QR
        // would report it as rank deficiency. Naming the real problem is more
        // use than naming its symptom.
        let first = treatment[0];
        if treatment.iter().all(|&t| (t - first).abs() < f64::EPSILON) {
            return Err(EstimationError::ConstantTreatment { value: first });
        }

        // Design matrix X = [1 | T | covariates], row-major.
        let mut x = vec![0.0; n * cols];
        for i in 0..n {
            x[i * cols] = 1.0;
            x[i * cols + 1] = treatment[i];
            for j in 0..p {
                x[i * cols + 2 + j] = covariates[[i, j]];
            }
        }
        let y: Vec<f64> = outcome.iter().copied().collect();

        let names = column_names(p);
        let qr = QrPivoted::factor(&x, n, cols, &names)?;
        let beta = qr.solve(&y);

        // Residuals and leverages, both needed for the robust estimators.
        let mut residual = vec![0.0; n];
        for i in 0..n {
            let mut fitted = 0.0;
            for (c, b) in beta.iter().enumerate() {
                fitted += x[i * cols + c] * b;
            }
            residual[i] = y[i] - fitted;
        }

        let ate = beta[1];
        let std_error = match se_kind {
            StdErrorKind::Classical => qr.classical_se(&residual, 1, n, cols),
            StdErrorKind::Hc1 | StdErrorKind::Hc3 => {
                qr.robust_se(&x, &residual, 1, n, cols, se_kind)
            }
            StdErrorKind::InfluenceFunction | StdErrorKind::Bootstrap => {
                // OLS has no weights, so the sandwich reduces to HC1, and
                // there is no nuisance model for a bootstrap to capture. Both
                // requests are honoured with the estimator that answers them.
                qr.robust_se(&x, &residual, 1, n, cols, StdErrorKind::Hc1)
            }
        };

        let (n_t, n_c) = treatment.iter().fold((0usize, 0usize), |(t, c), &v| {
            if v > 0.5 { (t + 1, c) } else { (t, c + 1) }
        });

        ATEResult::new(
            ate,
            std_error,
            n,
            Estimand::Ate,
            se_kind,
            Diagnostics {
                rank: Some((qr.rank, cols)),
                arm_sizes: Some((n_t, n_c)),
                ..Diagnostics::default()
            },
        )
    }
}

/// Names for the design columns, used to make a rank-deficiency error
/// actionable.
fn column_names(p: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(p + 2);
    names.push("intercept".to_string());
    names.push("treatment".to_string());
    for j in 0..p {
        names.push(format!("covariate[{j}]"));
    }
    names
}

/// Length check shared by every estimator.
pub(crate) fn check_lengths(
    name_a: &'static str,
    len_a: usize,
    name_b: &'static str,
    len_b: usize,
) -> Result<(), EstimationError> {
    if len_a == len_b {
        Ok(())
    } else {
        Err(EstimationError::LengthMismatch {
            name_a,
            len_a,
            name_b,
            len_b,
        })
    }
}

/// Householder QR with column pivoting.
///
/// Column pivoting is what distinguishes this from a plain QR: it selects the
/// column with the largest remaining norm at each step, so a rank-deficient
/// design leaves its dependent columns at the end with negligible pivots. That
/// makes rank *detectable*, and makes the offending columns *nameable*.
#[derive(Debug)]
pub(crate) struct QrPivoted {
    /// Packed factorisation, `rows x cols`, row-major. Upper triangle holds R.
    qr: Vec<f64>,
    /// Householder scalars.
    tau: Vec<f64>,
    /// Column permutation applied by pivoting.
    perm: Vec<usize>,
    rows: usize,
    cols: usize,
    /// Numerical rank found.
    pub rank: usize,
}

impl QrPivoted {
    /// Factor `a` (`rows x cols`, row-major), failing if it is rank deficient.
    ///
    /// # Errors
    ///
    /// [`EstimationError::RankDeficient`], naming the columns whose pivots
    /// collapsed — which are exactly the ones that carry no information the
    /// earlier columns did not already have.
    pub(crate) fn factor(
        a: &[f64],
        rows: usize,
        cols: usize,
        names: &[String],
    ) -> Result<Self, EstimationError> {
        let mut qr = a.to_vec();
        let mut tau = vec![0.0; cols];
        let mut perm: Vec<usize> = (0..cols).collect();

        // Squared column norms, updated downdate-style as elimination proceeds.
        let mut col_norm = vec![0.0; cols];
        for c in 0..cols {
            col_norm[c] = (0..rows).map(|r| qr[r * cols + c].powi(2)).sum();
        }

        let mut max_pivot: f64 = 0.0;
        let mut rank = 0usize;
        let steps = rows.min(cols);

        for k in 0..steps {
            // Pivot: bring forward the column with the largest remaining norm.
            let (best, &best_norm) = col_norm[k..]
                .iter()
                .enumerate()
                .map(|(i, v)| (i + k, v))
                .fold(
                    (k, &col_norm[k]),
                    |acc, cur| {
                        if *cur.1 > *acc.1 { cur } else { acc }
                    },
                );
            if best != k {
                for r in 0..rows {
                    qr.swap(r * cols + k, r * cols + best);
                }
                perm.swap(k, best);
                col_norm.swap(k, best);
            }
            let _ = best_norm;

            // Householder reflector zeroing below the diagonal in column k.
            let mut norm = 0.0;
            for r in k..rows {
                norm += qr[r * cols + k].powi(2);
            }
            norm = norm.sqrt();

            if norm.abs() < f64::MIN_POSITIVE {
                tau[k] = 0.0;
                continue;
            }

            let alpha = if qr[k * cols + k] > 0.0 { -norm } else { norm };
            let diag = qr[k * cols + k] - alpha;
            if diag.abs() < f64::MIN_POSITIVE {
                tau[k] = 0.0;
                qr[k * cols + k] = alpha;
                continue;
            }

            for r in (k + 1)..rows {
                qr[r * cols + k] /= diag;
            }
            tau[k] = -diag / alpha;
            qr[k * cols + k] = alpha;

            // Apply the reflector to the trailing columns.
            for c in (k + 1)..cols {
                let mut dot = qr[k * cols + c];
                for r in (k + 1)..rows {
                    dot += qr[r * cols + k] * qr[r * cols + c];
                }
                let scale = tau[k] * dot;
                qr[k * cols + c] -= scale;
                for r in (k + 1)..rows {
                    qr[r * cols + c] -= scale * qr[r * cols + k];
                }
            }

            // Downdate the remaining column norms.
            for c in (k + 1)..cols {
                col_norm[c] = (k + 1..rows).map(|r| qr[r * cols + c].powi(2)).sum();
            }

            let pivot = alpha.abs();
            max_pivot = max_pivot.max(pivot);
            if pivot > max_pivot * RANK_TOL {
                rank += 1;
            }
        }

        if rank < cols {
            // Pivoting has already sorted the informationless columns to the
            // end, so the tail of `perm` names them.
            let aliased = perm[rank..]
                .iter()
                .map(|&c| {
                    names
                        .get(c)
                        .cloned()
                        .unwrap_or_else(|| format!("column[{c}]"))
                })
                .collect();
            return Err(EstimationError::RankDeficient {
                rank,
                expected: cols,
                aliased,
            });
        }

        Ok(Self {
            qr,
            tau,
            perm,
            rows,
            cols,
            rank,
        })
    }

    /// Solve the least-squares problem for right-hand side `b`.
    pub(crate) fn solve(&self, b: &[f64]) -> Vec<f64> {
        // Apply Q^T to b.
        let mut qtb = b.to_vec();
        for k in 0..self.cols.min(self.rows) {
            if self.tau[k] == 0.0 {
                continue;
            }
            let dot = qtb[k]
                + qtb
                    .iter()
                    .enumerate()
                    .skip(k + 1)
                    .map(|(r, v)| self.qr[r * self.cols + k] * v)
                    .sum::<f64>();
            let scale = self.tau[k] * dot;
            qtb[k] -= scale;
            for (r, v) in qtb.iter_mut().enumerate().skip(k + 1) {
                *v -= scale * self.qr[r * self.cols + k];
            }
        }

        // Back-substitute through R.
        let mut z = vec![0.0; self.cols];
        for i in (0..self.cols).rev() {
            let sum = qtb[i]
                - z.iter()
                    .enumerate()
                    .skip(i + 1)
                    .map(|(j, v)| self.qr[i * self.cols + j] * v)
                    .sum::<f64>();
            z[i] = sum / self.qr[i * self.cols + i];
        }

        // Undo the pivoting.
        let mut beta = vec![0.0; self.cols];
        for (i, &pi) in self.perm.iter().enumerate() {
            beta[pi] = z[i];
        }
        beta
    }

    /// Leverage `h_ii` for each row of the design that was factored.
    ///
    /// `h_ii = s_i' (S'S)^-1 s_i`, the diagonal of the hat matrix — how much of
    /// its own fitted value a row supplies. Computed as `||R^-T P' s_i||^2` by
    /// forward substitution, so `S'S` is never formed and its condition number
    /// is never squared.
    ///
    /// # What it is for here
    ///
    /// A residual from a `k`-coefficient fit is shrunk toward zero by a factor
    /// of `1 - h_ii`. The usual `n / (n - k)` correction spreads that shrinkage
    /// evenly across rows, which is right only when every row has the same
    /// leverage. Where a handful of rows dominate the fit — heavy propensity
    /// weights, exactly the case finding C14 is about — those rows are shrunk
    /// far more than average and an even correction under-corrects.
    ///
    /// This is the same reasoning behind HC2/HC3 robust standard errors, which
    /// this crate already uses for OLS.
    pub(crate) fn leverages(&self, a: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.rows];
        let k = self.rank.min(self.cols);
        if k == 0 {
            return out;
        }
        for (i, slot) in out.iter_mut().enumerate() {
            // Permute the row to match R's column order.
            let mut z = vec![0.0; k];
            for j in 0..k {
                let v = a[i * self.cols + self.perm[j]];
                // Forward substitution through R', which is lower triangular.
                let sum: f64 = (0..j).map(|m| self.qr[m * self.cols + j] * z[m]).sum();
                let diag = self.qr[j * self.cols + j];
                z[j] = if diag.abs() > f64::EPSILON {
                    (v - sum) / diag
                } else {
                    0.0
                };
            }
            // Leverage is bounded by 1 in exact arithmetic; clamp so rounding
            // cannot produce a negative `1 - h` downstream.
            *slot = z.iter().map(|x| x * x).sum::<f64>().clamp(0.0, 1.0);
        }
        out
    }

    /// `(X'X)^-1` for the permuted system, computed from R alone.
    ///
    /// `(X'X)^-1 = R^-1 R^-T`, which avoids ever forming `X'X` — the step that
    /// squares the condition number and is the usual reason a normal-equations
    /// solver loses precision that QR keeps.
    fn xtx_inverse(&self) -> Vec<f64> {
        let c = self.cols;
        // R^-1 by back substitution, column by column.
        let mut r_inv = vec![0.0; c * c];
        for col in 0..c {
            let mut e = vec![0.0; c];
            e[col] = 1.0;
            for i in (0..=col).rev() {
                let mut sum = e[i];
                for j in (i + 1)..=col {
                    sum -= self.qr[i * c + j] * r_inv[j * c + col];
                }
                r_inv[i * c + col] = sum / self.qr[i * c + i];
            }
        }

        // R^-1 * R^-T, then un-permute back to input column order.
        let mut permuted = vec![0.0; c * c];
        for i in 0..c {
            for j in 0..c {
                let mut s = 0.0;
                for k in i.max(j)..c {
                    s += r_inv[i * c + k] * r_inv[j * c + k];
                }
                permuted[i * c + j] = s;
            }
        }

        let mut out = vec![0.0; c * c];
        for i in 0..c {
            for j in 0..c {
                out[self.perm[i] * c + self.perm[j]] = permuted[i * c + j];
            }
        }
        out
    }

    /// Classical standard error for coefficient `idx`, assuming homoskedasticity.
    fn classical_se(&self, residual: &[f64], idx: usize, n: usize, cols: usize) -> f64 {
        let dof = n as f64 - cols as f64;
        if dof <= 0.0 {
            return f64::NAN;
        }
        let rss: f64 = residual.iter().map(|r| r * r).sum();
        let sigma2 = rss / dof;
        let xtx_inv = self.xtx_inverse();
        (sigma2 * xtx_inv[idx * cols + idx]).max(0.0).sqrt()
    }

    /// Heteroskedasticity-consistent standard error (HC1 or HC3).
    ///
    /// Sandwich form `(X'X)^-1 X' diag(w_i e_i^2) X (X'X)^-1`, where the weight
    /// is the finite-sample correction `n/(n-k)` for HC1, or `1/(1-h_ii)^2` for
    /// HC3. HC3 is the better choice at small `n` because a high-leverage point
    /// shrinks its own residual, so HC1 systematically understates the variance
    /// exactly where it matters most.
    fn robust_se(
        &self,
        x: &[f64],
        residual: &[f64],
        idx: usize,
        n: usize,
        cols: usize,
        kind: StdErrorKind,
    ) -> f64 {
        let xtx_inv = self.xtx_inverse();

        // Leverages h_ii = x_i' (X'X)^-1 x_i, needed only for HC3.
        let leverage: Vec<f64> = if matches!(kind, StdErrorKind::Hc3) {
            (0..n)
                .map(|i| {
                    let mut h = 0.0;
                    for a in 0..cols {
                        let mut inner = 0.0;
                        for b in 0..cols {
                            inner += xtx_inv[a * cols + b] * x[i * cols + b];
                        }
                        h += x[i * cols + a] * inner;
                    }
                    // Numerical slop can push h just past 1; clamp so the HC3
                    // weight stays finite rather than exploding.
                    h.clamp(0.0, 1.0 - 1e-10)
                })
                .collect()
        } else {
            Vec::new()
        };

        let hc1_scale = n as f64 / (n as f64 - cols as f64);

        // meat = X' diag(w e^2) X
        let mut meat = vec![0.0; cols * cols];
        for i in 0..n {
            let w = match kind {
                StdErrorKind::Hc3 => 1.0 / (1.0 - leverage[i]).powi(2),
                _ => hc1_scale,
            };
            let we2 = w * residual[i] * residual[i];
            for a in 0..cols {
                let xa = x[i * cols + a];
                if xa == 0.0 {
                    continue;
                }
                for b in 0..cols {
                    meat[a * cols + b] += we2 * xa * x[i * cols + b];
                }
            }
        }

        // Only row `idx` of the sandwich is needed for one coefficient's
        // variance, so form `(X'X)^-1[idx,:] * meat * (X'X)^-1[:,idx]` directly
        // rather than the full k x k product.
        let mut tmp = vec![0.0; cols];
        for a in 0..cols {
            let mut s = 0.0;
            for b in 0..cols {
                s += xtx_inv[idx * cols + b] * meat[b * cols + a];
            }
            tmp[a] = s;
        }
        let mut var = 0.0;
        for a in 0..cols {
            var += tmp[a] * xtx_inv[a * cols + idx];
        }

        var.max(0.0).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn difference_in_means_recovers_a_known_effect() {
        let treatment = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let outcome = array![10.0, 12.0, 14.0, 4.0, 6.0, 8.0];
        let r = LinearATEEstimator::difference_in_means(&treatment, &outcome).expect("valid");
        assert!((r.ate() - 6.0).abs() < 1e-9);
        assert_eq!(r.estimand(), Estimand::Ate);
        assert_eq!(r.diagnostics().arm_sizes, Some((3, 3)));
    }

    #[test]
    fn single_arm_is_an_error_not_a_number() {
        // C10. This previously substituted 0.0 for the missing arm's mean and
        // returned the treated mean as if it were an effect.
        let treatment = array![1.0, 1.0, 1.0];
        let outcome = array![10.0, 11.0, 12.0];
        let err = LinearATEEstimator::difference_in_means(&treatment, &outcome).unwrap_err();
        assert!(matches!(
            err,
            EstimationError::EmptyArm {
                arm: "control",
                n_treated: 3,
                n_control: 0
            }
        ));
    }

    #[test]
    fn mismatched_lengths_are_an_error_not_a_panic() {
        // C10. This was `assert_eq!`, so a service embedding the crate died on
        // malformed input.
        let treatment = array![1.0, 0.0, 1.0];
        let outcome = array![1.0, 2.0];
        let err = LinearATEEstimator::difference_in_means(&treatment, &outcome).unwrap_err();
        assert!(matches!(err, EstimationError::LengthMismatch { .. }));
    }

    #[test]
    fn empty_input_is_an_error() {
        let treatment: Array1<f64> = Array1::from_vec(vec![]);
        let outcome: Array1<f64> = Array1::from_vec(vec![]);
        let err = LinearATEEstimator::difference_in_means(&treatment, &outcome).unwrap_err();
        assert_eq!(err, EstimationError::NoObservations);
    }

    #[test]
    fn ols_recovers_a_known_effect_with_confounding() {
        // Y = 3 + 5T + 2X, T correlated with X.
        let n = 200;
        let mut treatment = Array1::zeros(n);
        let mut outcome = Array1::zeros(n);
        let mut covariates = Array2::zeros((n, 1));
        for i in 0..n {
            let x = (i as f64 / n as f64) * 4.0 - 2.0;
            let t = f64::from(u8::from(x + ((i % 7) as f64 - 3.0) * 0.3 > 0.0));
            covariates[[i, 0]] = x;
            treatment[i] = t;
            outcome[i] = 3.0 + 5.0 * t + 2.0 * x;
        }
        let r = LinearATEEstimator::ols_adjusted(&treatment, &outcome, &covariates).expect("valid");
        assert!((r.ate() - 5.0).abs() < 1e-6, "ate was {}", r.ate());
        assert_eq!(r.diagnostics().rank, Some((3, 3)));
    }

    #[test]
    fn collinear_design_is_rejected_and_names_the_column() {
        // C1. This previously returned a plausible ATE with se = 0.0.
        let n = 100;
        let mut treatment = Array1::zeros(n);
        let mut outcome = Array1::zeros(n);
        let mut covariates = Array2::zeros((n, 3));
        for i in 0..n {
            let a = (i % 11) as f64;
            let b = (i % 7) as f64;
            covariates[[i, 0]] = a;
            covariates[[i, 1]] = b;
            covariates[[i, 2]] = a + b; // exactly aliased
            treatment[i] = f64::from(u8::from(i % 2 == 0));
            outcome[i] = 1.0 + 2.0 * treatment[i] + 0.5 * a;
        }
        let err = LinearATEEstimator::ols_adjusted(&treatment, &outcome, &covariates).unwrap_err();
        match err {
            EstimationError::RankDeficient {
                rank,
                expected,
                aliased,
            } => {
                assert_eq!(expected, 5);
                assert_eq!(rank, 4);
                assert_eq!(aliased.len(), 1);
                assert!(
                    aliased[0].starts_with("covariate["),
                    "should name a covariate, got {aliased:?}"
                );
            }
            other => panic!("expected RankDeficient, got {other:?}"),
        }
    }

    #[test]
    fn constant_treatment_is_named_directly() {
        let n = 50;
        let treatment = Array1::ones(n);
        let outcome = Array1::from_shape_fn(n, |i| i as f64);
        let covariates = Array2::from_shape_fn((n, 1), |(i, _)| (i % 5) as f64);
        let err = LinearATEEstimator::ols_adjusted(&treatment, &outcome, &covariates).unwrap_err();
        assert!(matches!(
            err,
            EstimationError::ConstantTreatment { value } if (value - 1.0).abs() < 1e-12
        ));
    }

    #[test]
    fn too_few_observations_is_an_error() {
        let treatment = array![1.0, 0.0, 1.0];
        let outcome = array![1.0, 2.0, 3.0];
        let covariates = Array2::from_shape_fn((3, 4), |(i, j)| (i + j) as f64);
        let err = LinearATEEstimator::ols_adjusted(&treatment, &outcome, &covariates).unwrap_err();
        assert!(matches!(
            err,
            EstimationError::InsufficientData { n: 3, p: 6 }
        ));
    }

    #[test]
    fn standard_error_is_never_exactly_zero_on_a_well_posed_fit() {
        let n = 300;
        let mut treatment = Array1::zeros(n);
        let mut outcome = Array1::zeros(n);
        let mut covariates = Array2::zeros((n, 2));
        for i in 0..n {
            let x = ((i * 37) % 101) as f64 / 50.0 - 1.0;
            let z = ((i * 53) % 97) as f64 / 48.0 - 1.0;
            covariates[[i, 0]] = x;
            covariates[[i, 1]] = z;
            treatment[i] = f64::from(u8::from(i % 3 == 0));
            outcome[i] = 1.0 + 2.0 * treatment[i] + x - 0.5 * z + ((i % 13) as f64 - 6.0) * 0.1;
        }
        let r = LinearATEEstimator::ols_adjusted(&treatment, &outcome, &covariates).expect("valid");
        assert!(r.std_error() > 0.0, "se was {}", r.std_error());
        assert!(r.std_error().is_finite());
    }

    #[test]
    fn hc3_is_wider_than_hc1_at_small_n() {
        // HC3 inflates by 1/(1-h)^2 rather than n/(n-k), so it must be the more
        // conservative of the two when leverage is non-trivial.
        let n = 40;
        let mut treatment = Array1::zeros(n);
        let mut outcome = Array1::zeros(n);
        let mut covariates = Array2::zeros((n, 2));
        for i in 0..n {
            let x = ((i * 17) % 23) as f64 / 11.0 - 1.0;
            covariates[[i, 0]] = x;
            covariates[[i, 1]] = x * x;
            treatment[i] = f64::from(u8::from(i % 2 == 0));
            // Error variance grows with |x| — the case HC1/HC3 exist for.
            outcome[i] = 1.0 + 2.0 * treatment[i] + x + x.abs() * ((i % 5) as f64 - 2.0);
        }
        let hc1 = LinearATEEstimator::ols_adjusted_with(
            &treatment,
            &outcome,
            &covariates,
            StdErrorKind::Hc1,
        )
        .expect("valid");
        let hc3 = LinearATEEstimator::ols_adjusted_with(
            &treatment,
            &outcome,
            &covariates,
            StdErrorKind::Hc3,
        )
        .expect("valid");
        assert!((hc1.ate() - hc3.ate()).abs() < 1e-12, "same point estimate");
        assert!(
            hc3.std_error() > hc1.std_error(),
            "HC3 {} should exceed HC1 {}",
            hc3.std_error(),
            hc1.std_error()
        );
    }

    #[test]
    fn qr_solves_a_known_system_exactly() {
        // 3 equations, 2 unknowns, exact fit: y = 1 + 2x.
        let x = vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0];
        let y = vec![1.0, 3.0, 5.0];
        let names = vec!["a".to_string(), "b".to_string()];
        let qr = QrPivoted::factor(&x, 3, 2, &names).expect("full rank");
        let beta = qr.solve(&y);
        assert!((beta[0] - 1.0).abs() < 1e-10, "intercept {}", beta[0]);
        assert!((beta[1] - 2.0).abs() < 1e-10, "slope {}", beta[1]);
    }

    #[test]
    fn qr_reports_rank_deficiency_rather_than_returning_zeros() {
        // Third column is the sum of the first two.
        let mut x = Vec::new();
        for i in 0..6 {
            let a = f64::from(i);
            let b = f64::from(i * i % 5);
            x.extend_from_slice(&[a, b, a + b]);
        }
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let err = QrPivoted::factor(&x, 6, 3, &names).unwrap_err();
        assert!(matches!(
            err,
            EstimationError::RankDeficient {
                rank: 2,
                expected: 3,
                ..
            }
        ));
    }
}
