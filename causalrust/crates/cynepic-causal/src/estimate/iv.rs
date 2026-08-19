//! Instrumental variables, via two-stage least squares.
//!
//! # 2SLS estimates LATE, not ATE
//!
//! This is the crate's most consequential labelling decision. Under
//! heterogeneous effects, an instrument identifies the effect **among
//! compliers** — the units whose treatment status the instrument actually
//! moves — and not the population average. Compliers are not observable
//! individually; the population is defined by the instrument itself, so a
//! different instrument on the same data estimates a different quantity.
//!
//! Reporting that number as "the" treatment effect is a category error. In a
//! clinical setting it is the difference between "this works" and "this works
//! for the people who comply with it", which is a different clinical claim.
//!
//! Every result from this module therefore carries [`Estimand::Late`]. The
//! previous implementation returned the same `ATEResult` type as OLS with no
//! way to tell them apart.
//!
//! # Weak instruments are refused
//!
//! Below a first-stage F of about 10 — the Stock–Yogo rule of thumb — 2SLS is
//! *more* biased than the OLS it was supposed to fix, and its standard errors
//! badly understate the true uncertainty. Returning an estimate there would be
//! worse than returning nothing, so this module returns
//! [`EstimationError::WeakInstrument`] with the F statistic attached.

use ndarray::{Array1, Array2};

use super::linear::{QrPivoted, check_lengths};
use crate::error::EstimationError;
use crate::estimand::{ATEResult, Diagnostics, Estimand, StdErrorKind};

/// Conventional first-stage F below which an instrument is considered weak.
pub const WEAK_INSTRUMENT_F: f64 = 10.0;

/// Two-stage least squares.
#[derive(Debug, Clone, Copy, Default)]
pub struct IVEstimator;

impl IVEstimator {
    /// Estimate the LATE by 2SLS.
    ///
    /// Stage 1 regresses treatment on the instruments; stage 2 regresses the
    /// outcome on fitted treatment. The reported standard error uses the
    /// **second-stage residuals computed from actual treatment**, not from
    /// fitted treatment — the classic 2SLS variance correction, without which
    /// the standard error is understated because the fitted regressor has less
    /// variance than the real one.
    ///
    /// # Errors
    ///
    /// - [`EstimationError::LengthMismatch`], [`EstimationError::NoObservations`],
    ///   [`EstimationError::InsufficientData`].
    /// - [`EstimationError::RankDeficient`] if the instruments are collinear.
    /// - [`EstimationError::WeakInstrument`] if the first-stage F is below
    ///   [`WEAK_INSTRUMENT_F`].
    pub fn two_stage_ls(
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
        instruments: &Array2<f64>,
    ) -> Result<ATEResult, EstimationError> {
        check_lengths("treatment", treatment.len(), "outcome", outcome.len())?;
        check_lengths(
            "treatment",
            treatment.len(),
            "instruments",
            instruments.nrows(),
        )?;

        let n = treatment.len();
        if n == 0 {
            return Err(EstimationError::NoObservations);
        }
        let k = instruments.ncols();
        if k == 0 {
            return Err(EstimationError::InsufficientData { n, p: 1 });
        }
        let stage1_cols = k + 1;
        if n <= stage1_cols + 1 {
            return Err(EstimationError::InsufficientData {
                n,
                p: stage1_cols + 1,
            });
        }

        // --- Stage 1: treatment on [1 | Z] -----------------------------------
        let mut z = vec![0.0; n * stage1_cols];
        for i in 0..n {
            z[i * stage1_cols] = 1.0;
            for j in 0..k {
                z[i * stage1_cols + 1 + j] = instruments[[i, j]];
            }
        }
        let mut z_names = Vec::with_capacity(stage1_cols);
        z_names.push("intercept".to_string());
        for j in 0..k {
            z_names.push(format!("instrument[{j}]"));
        }

        let t: Vec<f64> = treatment.iter().copied().collect();
        let qr1 = QrPivoted::factor(&z, n, stage1_cols, &z_names)?;
        let gamma = qr1.solve(&t);

        let mut t_hat = vec![0.0; n];
        for i in 0..n {
            let mut v = 0.0;
            for (c, g) in gamma.iter().enumerate() {
                v += z[i * stage1_cols + c] * g;
            }
            t_hat[i] = v;
        }

        // First-stage F for the joint significance of the instruments: the
        // standard weak-instrument diagnostic, and the reason this stage is not
        // just a means to an end.
        #[allow(clippy::cast_precision_loss)]
        let n_f = n as f64;
        let t_mean = t.iter().sum::<f64>() / n_f;
        let tss: f64 = t.iter().map(|v| (v - t_mean).powi(2)).sum();
        let rss: f64 = (0..n).map(|i| (t[i] - t_hat[i]).powi(2)).sum();
        #[allow(clippy::cast_precision_loss)]
        let k_f = k as f64;
        let first_stage_f = if rss > 0.0 && tss > rss {
            ((tss - rss) / k_f) / (rss / (n_f - k_f - 1.0))
        } else if rss <= 0.0 {
            // A perfect first stage is not weak; it is degenerate in the other
            // direction (the instrument determines treatment exactly).
            f64::INFINITY
        } else {
            0.0
        };

        if first_stage_f < WEAK_INSTRUMENT_F {
            return Err(EstimationError::WeakInstrument {
                first_stage_f,
                threshold: WEAK_INSTRUMENT_F,
            });
        }

        // --- Stage 2: outcome on [1 | T_hat] ---------------------------------
        let stage2_cols = 2;
        let mut x2 = vec![0.0; n * stage2_cols];
        for i in 0..n {
            x2[i * stage2_cols] = 1.0;
            x2[i * stage2_cols + 1] = t_hat[i];
        }
        let y: Vec<f64> = outcome.iter().copied().collect();
        let names2 = vec!["intercept".to_string(), "fitted_treatment".to_string()];
        let qr2 = QrPivoted::factor(&x2, n, stage2_cols, &names2)?;
        let beta = qr2.solve(&y);
        let late = beta[1];

        // 2SLS variance correction.
        //
        // Residuals must be formed against ACTUAL treatment, not fitted
        // treatment. Using the stage-2 residuals directly understates the
        // variance, because T_hat has had its endogenous variation removed and
        // so fits better than the real regressor ever could. This is the single
        // most common way a hand-rolled 2SLS produces intervals that are too
        // narrow.
        let mut rss2 = 0.0;
        for i in 0..n {
            let fitted = beta[0] + beta[1] * treatment[i];
            rss2 += (outcome[i] - fitted).powi(2);
        }
        let dof = n_f - 2.0;
        let sigma2 = if dof > 0.0 { rss2 / dof } else { f64::NAN };

        // Var(beta) = sigma^2 * (T_hat' T_hat)^-1, taken from the stage-2 QR.
        let t_hat_mean = t_hat.iter().sum::<f64>() / n_f;
        let s_tt: f64 = t_hat.iter().map(|v| (v - t_hat_mean).powi(2)).sum();
        let std_error = if s_tt > 0.0 && sigma2.is_finite() {
            (sigma2 / s_tt).max(0.0).sqrt()
        } else {
            f64::NAN
        };

        let (n_t, n_c) = treatment.iter().fold((0usize, 0usize), |(a, b), &v| {
            if v > 0.5 { (a + 1, b) } else { (a, b + 1) }
        });

        ATEResult::new(
            late,
            std_error,
            n,
            // The whole point: this is a LATE, and the type says so.
            Estimand::Late,
            StdErrorKind::Classical,
            Diagnostics {
                rank: Some((qr2.rank, stage2_cols)),
                first_stage_f: Some(first_stage_f),
                arm_sizes: Some((n_t, n_c)),
                ..Diagnostics::default()
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Encouragement design: Z shifts treatment, U confounds T and Y.
    fn iv_data(n: usize, strength: f64, effect: f64) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut treatment = Array1::zeros(n);
        let mut outcome = Array1::zeros(n);
        let mut instruments = Array2::zeros((n, 1));

        for i in 0..n {
            let z = f64::from(u8::from(rng.random::<f64>() < 0.5));
            let u: f64 = rng.random::<f64>() * 2.0 - 1.0;
            // Compliers follow the instrument; the rest follow the confounder.
            let complier = rng.random::<f64>() < strength;
            let t = if complier {
                z
            } else {
                f64::from(u8::from(u > 0.0))
            };
            instruments[[i, 0]] = z;
            treatment[i] = t;
            outcome[i] = 1.0 + effect * t + 3.0 * u + (rng.random::<f64>() - 0.5) * 0.2;
        }
        (treatment, outcome, instruments)
    }

    #[test]
    fn two_stage_ls_is_labelled_late_not_ate() {
        // The headline contract of this module.
        let (t, y, z) = iv_data(4_000, 0.8, 2.0);
        let r = IVEstimator::two_stage_ls(&t, &y, &z).expect("strong instrument");
        assert_eq!(r.estimand(), Estimand::Late);
        assert_eq!(r.estimand().label(), "LATE");
        assert!(r.estimand().population().contains("complier"));
    }

    #[test]
    fn recovers_the_effect_with_a_strong_instrument() {
        let (t, y, z) = iv_data(4_000, 0.85, 2.0);
        let r = IVEstimator::two_stage_ls(&t, &y, &z).expect("strong instrument");
        assert!(
            (r.ate() - 2.0).abs() < 0.5,
            "late was {} (F = {:?})",
            r.ate(),
            r.diagnostics().first_stage_f
        );
    }

    #[test]
    fn weak_instrument_is_refused_with_its_f_statistic() {
        // Below Stock-Yogo, 2SLS is more biased than the OLS it replaces.
        let (t, y, z) = iv_data(500, 0.02, 2.0);
        let err = IVEstimator::two_stage_ls(&t, &y, &z).unwrap_err();
        match err {
            EstimationError::WeakInstrument {
                first_stage_f,
                threshold,
            } => {
                assert!(first_stage_f < threshold);
                assert!((threshold - WEAK_INSTRUMENT_F).abs() < f64::EPSILON);
            }
            other => panic!("expected WeakInstrument, got {other:?}"),
        }
    }

    #[test]
    fn first_stage_f_is_reported_on_success() {
        let (t, y, z) = iv_data(4_000, 0.85, 2.0);
        let r = IVEstimator::two_stage_ls(&t, &y, &z).expect("strong instrument");
        let f = r.diagnostics().first_stage_f.expect("recorded");
        assert!(f >= WEAK_INSTRUMENT_F, "F was {f}");
    }

    #[test]
    fn standard_error_uses_actual_treatment_residuals() {
        // The correction matters: residuals against fitted treatment would give
        // a visibly smaller SE. Check the reported one is large enough to be
        // the corrected version by confirming the interval covers the truth.
        let (t, y, z) = iv_data(4_000, 0.85, 2.0);
        let r = IVEstimator::two_stage_ls(&t, &y, &z).expect("strong instrument");
        let (lo, hi) = r.confidence_interval(0.95).expect("finite se");
        assert!(lo < 2.0 && hi > 2.0, "95% CI [{lo}, {hi}] missed the truth");
    }

    #[test]
    fn mismatched_lengths_are_an_error() {
        let t = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let y = Array1::from_vec(vec![1.0, 2.0]);
        let z = Array2::zeros((3, 1));
        assert!(matches!(
            IVEstimator::two_stage_ls(&t, &y, &z),
            Err(EstimationError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn no_instruments_is_an_error() {
        let t = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let z = Array2::zeros((4, 0));
        assert!(IVEstimator::two_stage_ls(&t, &y, &z).is_err());
    }
}
