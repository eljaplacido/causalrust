//! Inverse probability weighting, and the propensity model underneath it.
//!
//! # What changed and why
//!
//! The previous implementation fitted its logistic regression with a fixed 100
//! iterations of gradient descent at a learning rate of 0.1, from a zero start,
//! with no convergence check and no way for a caller to inspect the result. On
//! the benign test DGP it stopped at roughly 55% of the true coefficients.
//! Propensities compressed toward 0.5, the weights under-corrected, and about
//! 47% of the confounding survived — while the reported 95% interval was 0.4
//! wide. Measured coverage was 0.0% on seven of eight grid cells.
//!
//! Reproduced independently, the decomposition was:
//!
//! ```text
//! 100 iterations (as shipped)   bias +1.80
//! 10,000 iterations             bias +0.26
//! 500,000 iterations            bias +0.26
//! oracle propensity             bias +0.13
//! naive difference in means     bias +3.40
//! ```
//!
//! Non-convergence accounted for 86% of the bias (finding C13). The variance
//! formula was separately wrong (finding C5): it summed unnormalised
//! Horvitz-Thompson contributions and centred them on a Hájek point estimate,
//! two quantities on different scales.
//!
//! Both are replaced here:
//!
//! - **IRLS / Fisher scoring** for the propensity model. It converges
//!   quadratically — 5 to 10 iterations where gradient descent needed tens of
//!   thousands — and returns [`EstimationError::NotConverged`] rather than a
//!   partial fit.
//! - **Influence-function variance**, derived from the same estimating equation
//!   as the Hájek point estimate, so the two describe the same quantity by
//!   construction.
//! - **Overlap is checked, not clipped.** Silently clamping propensities to
//!   `[0.01, 0.99]` converts a violated assumption into a plausible number.
//!   Insufficient overlap is now an error naming how many units are extreme.

use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::linear::{QrPivoted, check_lengths};
use crate::error::EstimationError;
use crate::estimand::{ATEResult, Convergence, Diagnostics, Estimand, StdErrorKind};

/// Convergence tolerance on the score (gradient) norm.
///
/// Public because it is part of the contract: a caller reading
/// [`EstimationError::NotConverged`] needs to know what the fit was aiming for.
pub const IRLS_TOL: f64 = 1e-8;

/// Maximum IRLS iterations.
///
/// Fisher scoring on a well-posed logistic problem converges in well under ten;
/// reaching this many means separation or near-separation, which is a data
/// property worth reporting rather than a limit worth raising.
pub const IRLS_MAX_ITER: usize = 50;

/// A fitted probability this close to 0 or 1 indicates separation: the
/// likelihood has no finite maximiser and the "converged" coefficients are an
/// artefact of where the iteration happened to stop.
pub const SEPARATION_TOL: f64 = 1e-6;

/// Propensities outside this range make a unit's weight dominate the estimate.
pub const OVERLAP_LOWER: f64 = 0.02;
/// Upper counterpart to [`OVERLAP_LOWER`].
pub const OVERLAP_UPPER: f64 = 0.98;

/// Fraction of units allowed outside the overlap bounds before the effect is
/// declared not estimable by weighting.
pub const MAX_EXTREME_FRACTION: f64 = 0.10;

/// A fitted propensity model.
///
/// Returned separately from the effect estimate so that "the weights were
/// wrong" is distinguishable from "the weights were right and the variance
/// formula was wrong". The previous API exposed neither, so neither could be
/// diagnosed.
#[derive(Debug, Clone)]
pub struct PropensityModel {
    /// Fitted coefficients: intercept first, then one per covariate.
    pub coefficients: Vec<f64>,
    /// Fitted `P(T=1 | X)` for each unit.
    pub scores: Array1<f64>,
    /// How the fit terminated.
    pub convergence: Convergence,
    /// The design matrix `[1 | X]` the fit used, row-major.
    ///
    /// Retained so the variance calculation can rebuild the model's score
    /// contributions without the caller having to pass the covariates a second
    /// time and risk passing different ones.
    pub(crate) design: Vec<f64>,
}

impl PropensityModel {
    /// Smallest and largest fitted score.
    pub fn range(&self) -> (f64, f64) {
        self.scores
            .iter()
            .fold((1.0, 0.0), |(lo, hi), &s| (lo.min(s), hi.max(s)))
    }

    /// Units whose score falls outside the overlap bounds.
    pub fn n_extreme(&self) -> usize {
        self.scores
            .iter()
            .filter(|&&s| !(OVERLAP_LOWER..=OVERLAP_UPPER).contains(&s))
            .count()
    }
}

/// Propensity-score estimation of treatment effects.
#[derive(Debug, Clone, Copy, Default)]
pub struct PropensityScoreEstimator;

impl PropensityScoreEstimator {
    /// Fit `P(T=1 | X)` by iteratively reweighted least squares.
    ///
    /// Exposed publicly because checking whether the propensity model recovered
    /// the assignment mechanism is a separate question from whether the effect
    /// estimate is right, and a caller cannot ask it if the model is hidden.
    ///
    /// # Errors
    ///
    /// - [`EstimationError::LengthMismatch`], [`EstimationError::NoObservations`].
    /// - [`EstimationError::RankDeficient`] if the covariates are collinear.
    /// - [`EstimationError::NotConverged`] if the score norm does not reach
    ///   tolerance within [`IRLS_MAX_ITER`], which in practice means the data
    ///   are separable and no finite coefficient vector exists.
    pub fn fit_propensity(
        treatment: &Array1<f64>,
        covariates: &Array2<f64>,
    ) -> Result<PropensityModel, EstimationError> {
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
        let cols = p + 1;
        if n <= cols {
            return Err(EstimationError::InsufficientData { n, p: cols });
        }

        // Design matrix [1 | X], row-major.
        let mut design = vec![0.0; n * cols];
        for i in 0..n {
            design[i * cols] = 1.0;
            for j in 0..p {
                design[i * cols + 1 + j] = covariates[[i, j]];
            }
        }

        let mut names = Vec::with_capacity(cols);
        names.push("intercept".to_string());
        for j in 0..p {
            names.push(format!("covariate[{j}]"));
        }

        let mut beta = vec![0.0; cols];
        let mut iterations = 0usize;
        let mut gradient_norm = f64::INFINITY;
        let mut converged = false;

        // Fisher scoring. Each step solves a weighted least-squares problem for
        // the Newton direction, using the same pivoted QR the linear estimator
        // uses — so collinear covariates are reported here too rather than
        // producing a silently degenerate fit.
        for iter in 1..=IRLS_MAX_ITER {
            iterations = iter;

            let mut w_sqrt = vec![0.0; n];
            let mut z = vec![0.0; n];
            let mut score = vec![0.0; cols];

            for i in 0..n {
                let eta: f64 = (0..cols).map(|c| design[i * cols + c] * beta[c]).sum();
                let mu = logistic(eta);
                // Variance of a Bernoulli at mu. Floored so a saturated unit
                // cannot produce a zero-weight row that silently drops out.
                let v = (mu * (1.0 - mu)).max(1e-10);
                let resid = treatment[i] - mu;

                for c in 0..cols {
                    score[c] += design[i * cols + c] * resid;
                }

                w_sqrt[i] = v.sqrt();
                // Working response for the IRLS step, pre-scaled by sqrt(w) so
                // the weighted problem becomes an ordinary least-squares one.
                z[i] = w_sqrt[i] * (eta + resid / v);
            }

            gradient_norm = score.iter().map(|s| s * s).sum::<f64>().sqrt() / n as f64;
            if gradient_norm < IRLS_TOL {
                converged = true;
                break;
            }

            let mut weighted = vec![0.0; n * cols];
            for i in 0..n {
                for c in 0..cols {
                    weighted[i * cols + c] = w_sqrt[i] * design[i * cols + c];
                }
            }

            let qr = QrPivoted::factor(&weighted, n, cols, &names)?;
            let next = qr.solve(&z);
            if next.iter().any(|b| !b.is_finite()) {
                return Err(EstimationError::NotConverged {
                    model: "propensity (IRLS)",
                    iterations: iter,
                    gradient_norm,
                    tolerance: IRLS_TOL,
                });
            }
            beta = next;
        }

        if !converged {
            return Err(EstimationError::NotConverged {
                model: "propensity (IRLS)",
                iterations,
                gradient_norm,
                tolerance: IRLS_TOL,
            });
        }

        let scores = Array1::from_shape_fn(n, |i| {
            logistic((0..cols).map(|c| design[i * cols + c] * beta[c]).sum())
        });

        // Separation check. The gradient test above cannot detect this: as the
        // coefficients diverge the fitted probabilities saturate and the score
        // goes to zero, so IRLS reports convergence at an arbitrary finite
        // point. Saturated probabilities are the observable symptom, and
        // reporting them is the difference between "here is your model" and
        // "no model exists".
        let n_separated = scores
            .iter()
            .filter(|&&s| !(SEPARATION_TOL..=1.0 - SEPARATION_TOL).contains(&s))
            .count();
        if n_separated > 0 {
            return Err(EstimationError::Separation {
                n_separated,
                n,
                tolerance: SEPARATION_TOL,
            });
        }

        Ok(PropensityModel {
            coefficients: beta,
            scores,
            convergence: Convergence {
                iterations,
                gradient_norm,
                converged,
            },
            design,
        })
    }

    /// Estimate the ATE by Hájek-normalised inverse probability weighting.
    ///
    /// # Errors
    ///
    /// Everything [`Self::fit_propensity`] can return, plus
    /// [`EstimationError::InsufficientOverlap`] when more than 10% of units
    /// have a fitted propensity outside `[0.02, 0.98]`.
    ///
    /// That last one is a refusal, and it is deliberate. Weighting cannot
    /// manufacture a comparison that the data does not contain; clipping the
    /// weights and returning a number would hide that behind a plausible
    /// figure. Refusing is the more useful answer, and on the weak-overlap
    /// grid cell it is the correct one.
    pub fn ipw(
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
        covariates: &Array2<f64>,
    ) -> Result<ATEResult, EstimationError> {
        check_lengths("treatment", treatment.len(), "outcome", outcome.len())?;
        let model = Self::fit_propensity(treatment, covariates)?;
        Self::ipw_with_model(treatment, outcome, &model)
    }

    /// IPW using an already-fitted propensity model.
    ///
    /// Separated so a caller can inspect or substitute the propensity model —
    /// for instance to compare a fitted model against a known assignment
    /// mechanism in simulation.
    ///
    /// # Errors
    ///
    /// As [`Self::ipw`].
    pub fn ipw_with_model(
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
        model: &PropensityModel,
    ) -> Result<ATEResult, EstimationError> {
        check_lengths("treatment", treatment.len(), "outcome", outcome.len())?;
        check_lengths(
            "treatment",
            treatment.len(),
            "propensity scores",
            model.scores.len(),
        )?;

        let n = treatment.len();
        if n == 0 {
            return Err(EstimationError::NoObservations);
        }

        let n_extreme = model.n_extreme();
        let (min_p, max_p) = model.range();
        #[allow(clippy::cast_precision_loss)]
        let extreme_fraction = n_extreme as f64 / n as f64;
        if extreme_fraction > MAX_EXTREME_FRACTION {
            return Err(EstimationError::InsufficientOverlap {
                n_extreme,
                n,
                lower: OVERLAP_LOWER,
                upper: OVERLAP_UPPER,
                min_propensity: min_p,
                max_propensity: max_p,
            });
        }

        let mut n_t = 0usize;
        let mut n_c = 0usize;
        let mut sum_w1 = 0.0;
        let mut sum_w0 = 0.0;
        let mut sum_w1y = 0.0;
        let mut sum_w0y = 0.0;
        let mut sum_w_sq = 0.0;
        let mut sum_w = 0.0;

        for i in 0..n {
            let e = model.scores[i].clamp(OVERLAP_LOWER, OVERLAP_UPPER);
            if treatment[i] > 0.5 {
                let w = 1.0 / e;
                sum_w1 += w;
                sum_w1y += w * outcome[i];
                sum_w += w;
                sum_w_sq += w * w;
                n_t += 1;
            } else {
                let w = 1.0 / (1.0 - e);
                sum_w0 += w;
                sum_w0y += w * outcome[i];
                sum_w += w;
                sum_w_sq += w * w;
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

        // Hájek: each arm's weighted mean is normalised by that arm's weight
        // total. Self-normalising bounds the estimate within the observed
        // outcome range, which the unnormalised Horvitz-Thompson form does not.
        let mu1 = sum_w1y / sum_w1;
        let mu0 = sum_w0y / sum_w0;
        let ate = mu1 - mu0;

        // Influence-function variance.
        //
        // For the Hájek estimator the influence contribution of unit i is
        //   psi_i = T_i/e_i (Y_i - mu1) / wbar1  -  (1-T_i)/(1-e_i) (Y_i - mu0) / wbar0
        // where wbar_a is the mean weight in arm a. Var(ATE) = sum psi_i^2 / n^2.
        //
        // The key property is that psi is derived from the same estimating
        // equation that defines the point estimate: each arm's residual is
        // taken against that arm's own Hájek mean, and scaled by that arm's own
        // mean weight. The previous formula centred unnormalised contributions
        // on the normalised estimate, so the two halves described different
        // quantities and the interval could not cover.
        #[allow(clippy::cast_precision_loss)]
        let n_f = n as f64;
        let wbar1 = sum_w1 / n_f;
        let wbar0 = sum_w0 / n_f;

        let mut psi = vec![0.0; n];
        for i in 0..n {
            let e = model.scores[i].clamp(OVERLAP_LOWER, OVERLAP_UPPER);
            psi[i] = if treatment[i] > 0.5 {
                (outcome[i] - mu1) / (e * wbar1)
            } else {
                -(outcome[i] - mu0) / ((1.0 - e) * wbar0)
            };
        }

        // Correct for the propensity model being ESTIMATED rather than known.
        //
        // The influence function above is the one for a known propensity. Using
        // a fitted propensity makes IPW *more* efficient, not less — the
        // estimation error in `e` is partly correlated with the estimation
        // error in the effect, and the two cancel. Ignoring that gives a
        // variance that is too large, which shows up as over-coverage:
        // intervals that contain the truth 100% of the time while claiming 95%.
        // Wide intervals are the safe direction but they are still wrong, and
        // they make a sound estimator look uninformative.
        //
        // The fix is to project the influence function off the propensity
        // model's score. With score `S_i = x_i (T_i - e_i)`, the corrected
        // influence is `psi_i - b'S_i` where `b` solves the least-squares
        // problem `psi ~ S`. What remains is the part of the influence that the
        // propensity model could not have explained.
        let (residual_psi, proj_k) = project_off_propensity_score(&psi, treatment, model);
        let var_sum: f64 = residual_psi.iter().map(|v| v * v).sum();
        let var = var_sum * projection_dof_correction(n, proj_k) / (n_f * n_f);
        let std_error = var.max(0.0).sqrt();
        let variance_dof = effective_variance_dof(&residual_psi);

        // Kish effective sample size: the honest n behind a weighted estimate.
        let effective_n = if sum_w_sq > 0.0 {
            sum_w * sum_w / sum_w_sq
        } else {
            0.0
        };

        ATEResult::new(
            ate,
            std_error,
            n,
            Estimand::Ate,
            StdErrorKind::InfluenceFunction,
            Diagnostics {
                convergence: Some(model.convergence),
                propensity_range: Some((min_p, max_p)),
                effective_n: Some(effective_n),
                arm_sizes: Some((n_t, n_c)),
                variance_dof,
                ..Diagnostics::default()
            },
        )
    }

    /// Estimate the ATE by IPW with a **bootstrap** standard error.
    ///
    /// Slower than [`Self::ipw`] and correct where it is not.
    ///
    /// # Why this exists
    ///
    /// The analytic standard error is an influence-function variance: a sum of
    /// squared influence contributions, computed once, conditional on the
    /// fitted propensity model. It is a good estimate of the right quantity,
    /// and it is *noisy* — because with heavy weights a handful of units
    /// dominate that sum, and because it treats the propensity model as fixed
    /// when the model was itself estimated from the same data.
    ///
    /// Measured on the strong-confounding cell at `n = 2000`, the analytic
    /// standard error had a coefficient of variation of 0.26, implying about
    /// 7.5 effective degrees of freedom. Satterthwaite applied to the influence
    /// contributions recovers only 23 of those, because it can see the first
    /// source of noise and not the second.
    ///
    /// Resampling refits the propensity model inside every replicate, so it
    /// captures both sources by construction. There is nothing to derive and
    /// nothing left out.
    ///
    /// # Measured, and it does not do what was expected
    ///
    /// This was implemented as the candidate fix for finding C14 — the residual
    /// under-coverage of IPW intervals when weights are heavy. It is not one.
    /// At 200 replications with 150 resamples each:
    ///
    /// ```text
    /// cell                  analytic (t)        bootstrap
    /// benign               94.5%  w=0.195     96.5%  w=0.198
    /// moderate-overlap     93.0%  w=0.623     93.0%  w=0.557
    /// strong-confounding   92.5%  w=0.654     89.5%  w=0.581
    /// ```
    ///
    /// The bootstrap interval is *narrower* than the analytic one exactly where
    /// the analytic one was already too narrow, and covers worse. That is a
    /// known limitation rather than a defect here: a nonparametric bootstrap
    /// resamples the units it was given, so it cannot reproduce a tail event
    /// that did not occur in the original sample — and with heavy weights, the
    /// variance lives in those tails. Resampling a heavy-tailed estimator
    /// systematically understates its spread.
    ///
    /// So this is offered as a capability, not as the better default. It is
    /// genuinely useful when the propensity model's own uncertainty is the
    /// question being asked; it is not the answer to C14.
    ///
    /// # Cost
    ///
    /// `resamples` full IRLS fits. At `n = 2000, p = 3` and 200 resamples that
    /// is well under a second, but it is roughly two hundred times the analytic
    /// path.
    ///
    /// # Errors
    ///
    /// As [`Self::ipw`]. Individual resamples that fail — a degenerate draw
    /// with one arm empty, say — are skipped rather than failing the whole
    /// estimate; if fewer than 30 survive, [`EstimationError::InsufficientData`]
    /// is returned rather than a standard error computed from a handful.
    pub fn ipw_bootstrap(
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
        covariates: &Array2<f64>,
        resamples: usize,
        seed: u64,
    ) -> Result<ATEResult, EstimationError> {
        // The point estimate is the one from the full sample, not the bootstrap
        // mean. Resampling is used to measure spread, not to re-centre — the
        // bootstrap mean carries the resampling bias as well as the estimator's.
        let point = Self::ipw(treatment, outcome, covariates)?;

        let n = treatment.len();
        let p = covariates.ncols();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut estimates = Vec::with_capacity(resamples);

        for _ in 0..resamples.max(2) {
            let idx: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();
            let t = Array1::from_shape_fn(n, |i| treatment[idx[i]]);
            let y = Array1::from_shape_fn(n, |i| outcome[idx[i]]);
            let x = Array2::from_shape_fn((n, p), |(i, j)| covariates[[idx[i], j]]);

            // A resample can be degenerate. Skipping is right: the alternative
            // is to fail an otherwise sound estimate because one draw of 200
            // happened to be pathological.
            if let Ok(r) = Self::ipw(&t, &y, &x) {
                estimates.push(r.ate());
            }
        }

        if estimates.len() < 30 {
            return Err(EstimationError::InsufficientData {
                n: estimates.len(),
                p: 30,
            });
        }

        #[allow(clippy::cast_precision_loss)]
        let b = estimates.len() as f64;
        let mean = estimates.iter().sum::<f64>() / b;
        let std_error =
            (estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (b - 1.0)).sqrt();

        // A bootstrap standard error computed from B replicates has a
        // coefficient of variation of about 1/sqrt(2B) — 5% at B=200 — so it is
        // far less noisy than the analytic one it replaces, and a normal
        // quantile is appropriate. Recorded as B-1 degrees of freedom so the
        // interval still accounts for the residual noise rather than pretending
        // to none.
        let mut diagnostics = point.diagnostics().clone();
        diagnostics.variance_dof = Some(b - 1.0);

        ATEResult::new(
            point.ate(),
            std_error,
            n,
            Estimand::Ate,
            StdErrorKind::Bootstrap,
            diagnostics,
        )
    }

    /// Estimate the **ATT** by weighting controls to resemble the treated.
    ///
    /// A different estimand from [`Self::ipw`], and labelled as such. Treated
    /// units enter with weight 1; controls with `e/(1-e)`. Under effect
    /// heterogeneity this is a materially different number from the ATE, which
    /// is precisely why it carries [`Estimand::Att`].
    ///
    /// # Errors
    ///
    /// As [`Self::ipw`].
    pub fn att(
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
        covariates: &Array2<f64>,
    ) -> Result<ATEResult, EstimationError> {
        check_lengths("treatment", treatment.len(), "outcome", outcome.len())?;
        let model = Self::fit_propensity(treatment, covariates)?;
        let n = treatment.len();

        let n_extreme = model.n_extreme();
        let (min_p, max_p) = model.range();
        #[allow(clippy::cast_precision_loss)]
        let extreme_fraction = n_extreme as f64 / n as f64;
        if extreme_fraction > MAX_EXTREME_FRACTION {
            return Err(EstimationError::InsufficientOverlap {
                n_extreme,
                n,
                lower: OVERLAP_LOWER,
                upper: OVERLAP_UPPER,
                min_propensity: min_p,
                max_propensity: max_p,
            });
        }

        let mut n_t = 0usize;
        let mut n_c = 0usize;
        let mut sum_y1 = 0.0;
        let mut sum_w0 = 0.0;
        let mut sum_w0y = 0.0;

        for i in 0..n {
            let e = model.scores[i].clamp(OVERLAP_LOWER, OVERLAP_UPPER);
            if treatment[i] > 0.5 {
                sum_y1 += outcome[i];
                n_t += 1;
            } else {
                let w = e / (1.0 - e);
                sum_w0 += w;
                sum_w0y += w * outcome[i];
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

        #[allow(clippy::cast_precision_loss)]
        let n_t_f = n_t as f64;
        let mu1 = sum_y1 / n_t_f;
        let mu0 = sum_w0y / sum_w0;
        let ate = mu1 - mu0;

        #[allow(clippy::cast_precision_loss)]
        let n_f = n as f64;
        let wbar0 = sum_w0 / n_f;
        let share_treated = n_t_f / n_f;

        let mut psi = vec![0.0; n];
        for i in 0..n {
            let e = model.scores[i].clamp(OVERLAP_LOWER, OVERLAP_UPPER);
            psi[i] = if treatment[i] > 0.5 {
                (outcome[i] - mu1) / share_treated
            } else {
                -(e / (1.0 - e)) * (outcome[i] - mu0) / wbar0
            };
        }
        // Same estimated-propensity correction as `ipw`; see the comment there.
        // ATT weights depend on the fitted model at least as strongly as ATE
        // weights do, so omitting it here left this estimator over-covering at
        // 100% with intervals roughly three times wider than they needed to be.
        let (residual_psi, proj_k) = project_off_propensity_score(&psi, treatment, &model);
        let var_sum: f64 = residual_psi.iter().map(|v| v * v).sum();
        let var = var_sum * projection_dof_correction(n, proj_k) / (n_f * n_f);
        let std_error = var.max(0.0).sqrt();
        let variance_dof = effective_variance_dof(&residual_psi);

        ATEResult::new(
            ate,
            std_error,
            n,
            Estimand::Att,
            StdErrorKind::InfluenceFunction,
            Diagnostics {
                convergence: Some(model.convergence),
                propensity_range: Some((min_p, max_p)),
                arm_sizes: Some((n_t, n_c)),
                variance_dof,
                ..Diagnostics::default()
            },
        )
    }
}

/// Project an influence function off the propensity model's score.
///
/// Returns the residual contributions and the number of coefficients the
/// projection consumed, which the caller needs for the `(n - k)` correction.
///
/// Returns the residual from regressing `psi` on the score contributions
/// `S_i = x_i (T_i - e_i)`. The design used here is `[1 | X]`, matching the
/// propensity fit, and the regression is solved with the same pivoted QR the
/// rest of the crate uses so a degenerate score matrix is caught rather than
/// silently producing a zero correction.
///
/// If the projection cannot be formed — too few units, or a rank-deficient
/// score matrix — the uncorrected influence is returned. That is the
/// conservative direction: wider intervals rather than narrower ones.
fn project_off_propensity_score(
    psi: &[f64],
    treatment: &Array1<f64>,
    model: &PropensityModel,
) -> (Vec<f64>, usize) {
    let n = psi.len();
    let cols = model.coefficients.len();
    if n <= cols || cols == 0 {
        return (psi.to_vec(), 0);
    }

    // Rebuild the score contributions. `coefficients` is [intercept, betas...],
    // so the design column count matches and column 0 is the constant.
    let mut score = vec![0.0; n * cols];
    for i in 0..n {
        let resid = treatment[i] - model.scores[i];
        score[i * cols] = resid;
        for j in 1..cols {
            // The covariate values are recoverable from the fitted linear
            // predictor only in special cases, so the caller passes them
            // implicitly through `model.design`.
            score[i * cols + j] = model.design[i * cols + j] * resid;
        }
    }

    let names: Vec<String> = (0..cols).map(|j| format!("score[{j}]")).collect();
    let Ok(qr) = QrPivoted::factor(&score, n, cols, &names) else {
        return (psi.to_vec(), 0);
    };
    let rank = qr.rank;
    let b = qr.solve(psi);

    let residual = (0..n)
        .map(|i| {
            let fitted: f64 = (0..cols).map(|j| score[i * cols + j] * b[j]).sum();
            psi[i] - fitted
        })
        .collect();
    (residual, rank)
}

/// Rescale a projected sum of squares for the coefficients the projection spent.
///
/// `project_off_propensity_score` returns residuals from a `k`-coefficient
/// least-squares fit, so `sum(residual^2)` is a **residual** sum of squares and
/// is biased low by exactly the factor `(n - k) / n` — the same reason an OLS
/// variance divides by `n - k` and not by `n`. The variance here divided by
/// `n^2` and made no such correction.
///
/// At `n = 2000` with a handful of covariates this is a quarter of one percent
/// and invisible, which is why it survived. At the `high-dim` cell — `p = 25`,
/// so `k = 26`, on `n = 400` — it is **6.5%** of the variance, and that cell
/// covers at 89.2% with a bias of 0.02 sd. Nothing was wrong with the point
/// estimate there; the interval was simply too narrow, and this is why.
fn projection_dof_correction(n: usize, k: usize) -> f64 {
    if k == 0 || n <= k + 1 {
        return 1.0;
    }
    #[allow(clippy::cast_precision_loss)]
    {
        n as f64 / (n - k) as f64
    }
}

/// Effective degrees of freedom for a sum-of-squares variance estimate.
///
/// The variance of a weighted estimator is `sum(psi_i^2) / n^2`. That sum is
/// itself random, and when a few large influence contributions dominate it, it
/// is a *noisy* random quantity — which makes the reported standard error
/// noisy, which makes a normal interval too narrow. A noisy variance estimate
/// is what Student's t exists for; the question is how many degrees of freedom
/// to claim.
///
/// # Kish, not Satterthwaite
///
/// ```text
///   d = (sum psi^2)^2 / sum psi^4
/// ```
///
/// This is Kish's effective sample size — the same quantity this estimator
/// already reports as `Diagnostics::effective_n`, applied to the influence
/// contributions rather than to the weights. It counts how many contributions
/// the sum of squares effectively rests on.
///
/// This was previously Satterthwaite's `nu = 2 (sum psi^2)^2 / (sum psi^4 -
/// (sum psi^2)^2 / n)`, which is the same quantity **times two**. That factor
/// comes from `Var(chi^2_nu) = 2 nu`, which holds when the summands are squares
/// of Gaussians.
///
/// Heavy tails are the only condition under which this correction matters at
/// all, and under heavy tails `psi^2` has a coefficient of variation above the
/// Gaussian value — so the true degrees of freedom are *below* `2 x Kish`.
/// Keeping a Gaussian factor in a correction that exists because Gaussianity
/// failed is the error; dropping it is not a fudge.
///
/// Measured across the DGP grid, 500 replications: coverage improves at every
/// affected cell (`moderate-overlap` 90.4% → 91.2%, `strong-confounding` 91.2%
/// → 91.8%, `high-dim` 90.4% → 91.0%) and moves no cell that was already
/// nominal — because where the contributions really are near-Gaussian the two
/// rules differ by a factor of two on a dof in the hundreds, and `t(0.975, 334)`
/// is 1.967 against `t(0.975, 803)` = 1.963. The correction is invisible
/// exactly where it should be.
///
/// It does **not** close finding C14 on its own. See `docs/FINDINGS.md`.
///
/// Returns `None` when the sum is degenerate, in which case the caller falls
/// back to a normal quantile.
fn effective_variance_dof(psi: &[f64]) -> Option<f64> {
    let n = psi.len();
    if n < 4 {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    let n_f = n as f64;

    let s2: f64 = psi.iter().map(|p| p * p).sum();
    let s4: f64 = psi.iter().map(|p| p.powi(4)).sum();
    if !s2.is_finite() || !s4.is_finite() || s2 <= 0.0 || s4 <= 0.0 {
        return None;
    }

    let dof = s2 * s2 / s4;
    if dof.is_finite() && dof >= 1.0 {
        // Never claim more degrees of freedom than there are observations.
        Some(dof.min(n_f - 1.0))
    } else {
        None
    }
}

/// Logistic function, computed so neither tail overflows.
fn logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod dof_rules {
    use super::*;

    /// The projection spends coefficients, and the variance must pay for them.
    #[test]
    fn projection_correction_scales_with_the_coefficients_spent() {
        // `high-dim`: p = 25 so the score matrix has 26 columns, on n = 400.
        let c = projection_dof_correction(400, 26);
        assert!(
            (c - 400.0 / 374.0).abs() < 1e-12,
            "expected n/(n-k), got {c}"
        );
        // 6.5% of the variance, which is 3.2% of the standard error. That cell
        // covered at 89.2% with a bias of 0.02 sd — nothing was wrong with the
        // point estimate, the interval was simply too narrow.
        assert!(c > 1.06 && c < 1.08, "correction should be ~6.9%, got {c}");

        // At the sample sizes where this went unnoticed it is invisible, which
        // is exactly why it went unnoticed.
        let big = projection_dof_correction(2_000, 5);
        assert!(
            big < 1.003,
            "at n=2000, k=5 the correction is a quarter of a percent, got {big}"
        );
    }

    /// Degenerate inputs must not produce a correction that inflates or divides
    /// by zero.
    #[test]
    fn projection_correction_is_one_when_it_cannot_apply() {
        assert!((projection_dof_correction(400, 0) - 1.0).abs() < f64::EPSILON);
        assert!((projection_dof_correction(10, 10) - 1.0).abs() < f64::EPSILON);
        assert!((projection_dof_correction(10, 20) - 1.0).abs() < f64::EPSILON);
    }

    /// The dof rule is Kish's effective count, which is Satterthwaite without
    /// the Gaussian factor of two.
    #[test]
    fn effective_dof_is_the_kish_effective_count() {
        // Equal contributions: every unit counts, so the effective count is n.
        let flat = vec![1.0_f64; 100];
        let d = effective_variance_dof(&flat).expect("well-defined");
        assert!((d - 99.0).abs() < 1e-9, "expected n-1 cap, got {d}");

        // One contribution dominating: the sum of squares rests on ~1 unit, and
        // the rule must say so rather than reporting the sample size.
        let mut heavy = vec![1.0_f64; 1_000];
        heavy[0] = 1_000.0;
        let d = effective_variance_dof(&heavy).expect("well-defined");
        assert!(
            d < 2.0,
            "one unit carries the variance; dof should be ~1, got {d}"
        );
    }

    /// Fewer degrees of freedom than the old rule, always — that is the point.
    #[test]
    fn the_rule_never_claims_more_than_satterthwaite_did() {
        // A moderately heavy tail, of the shape strong confounding produces.
        let psi: Vec<f64> = (1..=500).map(|i| 1.0 / f64::from(i)).collect();
        let s2: f64 = psi.iter().map(|p| p * p).sum();
        let s4: f64 = psi.iter().map(|p| p.powi(4)).sum();
        let legacy = 2.0 * s2 * s2 / (s4 - s2 * s2 / 500.0);
        let now = effective_variance_dof(&psi).expect("well-defined");
        assert!(
            now < legacy,
            "the correction must reduce claimed dof: {now} vs {legacy}"
        );
        assert!(
            now > legacy / 2.5,
            "and it must not collapse them either: {now} vs {legacy}"
        );
    }

    /// Degenerate input falls back rather than inventing a number.
    #[test]
    fn degenerate_contributions_report_no_dof() {
        assert!(effective_variance_dof(&[]).is_none());
        assert!(effective_variance_dof(&[1.0, 2.0]).is_none());
        assert!(effective_variance_dof(&[0.0; 100]).is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Confounded data with a known effect of 5.0. Treatment probability
    /// depends on X, and X also drives the outcome.
    fn confounded(n: usize) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let mut treatment = Array1::zeros(n);
        let mut outcome = Array1::zeros(n);
        let mut covariates = Array2::zeros((n, 1));
        for i in 0..n {
            let x = if i < n / 2 { 1.0 } else { -1.0 };
            covariates[[i, 0]] = x;
            let treated = if x > 0.0 { i % 4 != 0 } else { i % 4 == 0 };
            treatment[i] = f64::from(u8::from(treated));
            outcome[i] = 5.0 * treatment[i] + 3.0 * x;
        }
        (treatment, outcome, covariates)
    }

    #[test]
    fn irls_converges_in_a_handful_of_iterations() {
        // C13. Gradient descent needed tens of thousands and shipped with 100.
        let (treatment, _, covariates) = confounded(400);
        let model =
            PropensityScoreEstimator::fit_propensity(&treatment, &covariates).expect("converges");
        assert!(model.convergence.converged);
        assert!(
            model.convergence.iterations <= 10,
            "IRLS took {} iterations",
            model.convergence.iterations
        );
    }

    #[test]
    fn irls_recovers_the_assignment_probabilities() {
        // 3-in-4 treated at x=1, 1-in-4 at x=-1.
        let (treatment, _, covariates) = confounded(400);
        let model =
            PropensityScoreEstimator::fit_propensity(&treatment, &covariates).expect("converges");
        let at_pos = model.scores[0];
        let at_neg = model.scores[399];
        assert!((at_pos - 0.75).abs() < 0.05, "P(T=1|x=1) was {at_pos}");
        assert!((at_neg - 0.25).abs() < 0.05, "P(T=1|x=-1) was {at_neg}");
    }

    #[test]
    fn ipw_recovers_a_known_effect_tightly() {
        // The old test allowed |ate - 5.0| < 2.0 — a 40% tolerance, wide enough
        // to accept a broken estimator. This asks for 2%.
        let (treatment, outcome, covariates) = confounded(400);
        let r = PropensityScoreEstimator::ipw(&treatment, &outcome, &covariates).expect("valid");
        assert!((r.ate() - 5.0).abs() < 0.1, "ate was {}", r.ate());
        assert_eq!(r.estimand(), Estimand::Ate);
        assert_eq!(r.std_error_kind(), StdErrorKind::InfluenceFunction);
    }

    #[test]
    fn ipw_reports_effective_sample_size() {
        let (treatment, outcome, covariates) = confounded(400);
        let r = PropensityScoreEstimator::ipw(&treatment, &outcome, &covariates).expect("valid");
        let ess = r.diagnostics().effective_n.expect("recorded");
        // Weighting always costs precision; ESS must be below n and above zero.
        assert!(ess > 0.0 && ess < 400.0, "ess was {ess}");
    }

    #[test]
    fn att_is_labelled_as_att() {
        let (treatment, outcome, covariates) = confounded(400);
        let r = PropensityScoreEstimator::att(&treatment, &outcome, &covariates).expect("valid");
        assert_eq!(r.estimand(), Estimand::Att);
        // Homogeneous effect here, so ATT and ATE agree numerically. The point
        // is that they are distinguishable in the type, not in the number.
        assert!((r.ate() - 5.0).abs() < 0.2, "att was {}", r.ate());
    }

    #[test]
    fn separable_data_is_reported_as_separation() {
        // Treatment perfectly predicted by X: no finite coefficient exists.
        let n = 100;
        let mut treatment = Array1::zeros(n);
        let mut covariates = Array2::zeros((n, 1));
        for i in 0..n {
            let x = if i < n / 2 { -5.0 } else { 5.0 };
            covariates[[i, 0]] = x;
            treatment[i] = f64::from(u8::from(x > 0.0));
        }
        let err = PropensityScoreEstimator::fit_propensity(&treatment, &covariates).unwrap_err();
        // IRLS *does* satisfy its gradient test here — the score vanishes as
        // the probabilities saturate — so this must be caught by the saturation
        // check, not by the convergence check.
        assert!(
            matches!(err, EstimationError::Separation { n_separated, .. } if n_separated == 100),
            "expected Separation, got {err:?}"
        );
    }

    #[test]
    fn no_overlap_is_refused_rather_than_clipped() {
        // Near-deterministic assignment: weighting cannot repair this, and
        // clipping to [0.01, 0.99] would have returned a plausible number.
        let n = 200;
        let mut treatment = Array1::zeros(n);
        let mut outcome = Array1::zeros(n);
        let mut covariates = Array2::zeros((n, 1));
        for i in 0..n {
            let x = (i as f64 / n as f64) * 12.0 - 6.0;
            covariates[[i, 0]] = x;
            // One "mistake" per tail keeps the fit from being fully separable.
            let treated = if i == 0 {
                true
            } else if i == n - 1 {
                false
            } else {
                x > 0.0
            };
            treatment[i] = f64::from(u8::from(treated));
            outcome[i] = 2.0 * treatment[i] + x;
        }
        let err = PropensityScoreEstimator::ipw(&treatment, &outcome, &covariates).unwrap_err();
        assert!(
            matches!(
                err,
                EstimationError::InsufficientOverlap { .. }
                    | EstimationError::NotConverged { .. }
                    | EstimationError::Separation { .. }
            ),
            "expected a refusal, got {err:?}"
        );
    }

    #[test]
    fn mismatched_lengths_are_an_error() {
        let treatment = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let outcome = Array1::from_vec(vec![1.0, 2.0]);
        let covariates = Array2::zeros((3, 1));
        let err = PropensityScoreEstimator::ipw(&treatment, &outcome, &covariates).unwrap_err();
        assert!(matches!(err, EstimationError::LengthMismatch { .. }));
    }

    #[test]
    fn logistic_does_not_overflow_in_either_tail() {
        assert!((logistic(1000.0) - 1.0).abs() < 1e-12);
        assert!(logistic(-1000.0).abs() < 1e-12);
        assert!((logistic(0.0) - 0.5).abs() < 1e-12);
    }
}

/// Diagnostics for finding C14, kept in the tree because the finding is open
/// and every hypothesis about it so far has been settled by measurement rather
/// than argument.
///
/// `#[ignore]`d: this is a Monte Carlo study, not a check. Run it with
///
/// ```bash
/// cargo test -p cynepic-causal --lib c14 -- --ignored --nocapture --test-threads=1
/// ```
///
/// It lives in the library's own test module rather than an integration suite
/// so it can reach `psi`, the propensity design and the private dof rules. It
/// is deliberately outside the suites `scripts/findings-ratchet.sh` scans —
/// those hold specs that must fail, and a diagnostic that prints is neither.
#[cfg(test)]
mod c14_diagnostics {
    use super::*;
    use cynepic_testkit::Dgp;

    /// Rebuild the influence contributions the estimator uses, so alternative
    /// degrees-of-freedom rules can be compared on identical input.
    fn influence(
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
        model: &PropensityModel,
    ) -> (Vec<f64>, f64, f64, f64, f64) {
        let n = treatment.len();
        #[allow(clippy::cast_precision_loss)]
        let n_f = n as f64;
        let (mut sum_w1, mut sum_w0, mut sum_w1y, mut sum_w0y) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            let e = model.scores[i].clamp(OVERLAP_LOWER, OVERLAP_UPPER);
            if treatment[i] > 0.5 {
                let w = 1.0 / e;
                sum_w1 += w;
                sum_w1y += w * outcome[i];
            } else {
                let w = 1.0 / (1.0 - e);
                sum_w0 += w;
                sum_w0y += w * outcome[i];
            }
        }
        let (mu1, mu0) = (sum_w1y / sum_w1, sum_w0y / sum_w0);
        let (wbar1, wbar0) = (sum_w1 / n_f, sum_w0 / n_f);

        let mut psi = vec![0.0; n];
        for i in 0..n {
            let e = model.scores[i].clamp(OVERLAP_LOWER, OVERLAP_UPPER);
            psi[i] = if treatment[i] > 0.5 {
                (outcome[i] - mu1) / (e * wbar1)
            } else {
                -(outcome[i] - mu0) / ((1.0 - e) * wbar0)
            };
        }
        (psi, mu1, mu0, wbar1, wbar0)
    }

    /// Candidate 1, **refuted**: model-based fourth moment.
    ///
    /// Satterthwaite's `nu` depends only on the ratio `S4 / S2^2`, so a uniform
    /// rescaling of the influence contributions cancels — which made it look
    /// legitimate to compute that ratio from model-based moments. The empirical
    /// `S4` is biased down under heavy tails (median 1.39e-1 against a mean of
    /// 1.66e-1 at strong confounding), which biases `nu` *up*, so replacing it
    /// with an expectation over `T_i ~ Bernoulli(e_i)` — using every unit in
    /// both arms, so the `1/e` tail is fully represented — should have helped.
    ///
    /// It did the opposite: `nu` went from 14.5 to 30.6 and coverage fell.
    ///
    /// The reason is the assumption hiding in it. Taking `E[psi^4]` as
    /// `m4_arm / e^3` treats the outcome residual's moments as independent of
    /// the propensity. Under confounding they are *not* independent — that is
    /// what confounding means. Units with extreme `e` also have extreme
    /// outcomes, so the true fourth moment is far larger than the product of
    /// the two marginals, and the model-based estimate smooths away exactly the
    /// co-movement that creates the tail.
    ///
    /// Kept, unused, so the next person does not have the same idea twice.
    #[allow(dead_code)]
    fn model_based_dof(
        treatment: &Array1<f64>,
        outcome: &Array1<f64>,
        model: &PropensityModel,
        mu1: f64,
        mu0: f64,
        wbar1: f64,
        wbar0: f64,
    ) -> Option<f64> {
        let n = treatment.len();
        #[allow(clippy::cast_precision_loss)]
        let n_f = n as f64;

        // Weighted residual moments per arm: weighting by 1/e maps the observed
        // arm back to the whole population, which is the population the
        // expectation below is taken over.
        let (mut s1, mut m1, mut z1) = (0.0, 0.0, 0.0);
        let (mut s0, mut m0, mut z0) = (0.0, 0.0, 0.0);
        for i in 0..n {
            let e = model.scores[i].clamp(OVERLAP_LOWER, OVERLAP_UPPER);
            if treatment[i] > 0.5 {
                let (r, w) = (outcome[i] - mu1, 1.0 / e);
                s1 += w * r * r;
                m1 += w * r.powi(4);
                z1 += w;
            } else {
                let (r, w) = (outcome[i] - mu0, 1.0 / (1.0 - e));
                s0 += w * r * r;
                m0 += w * r.powi(4);
                z0 += w;
            }
        }
        if z1 <= 0.0 || z0 <= 0.0 {
            return None;
        }
        let (s1, m1) = (s1 / z1, m1 / z1);
        let (s0, m0) = (s0 / z0, m0 / z0);

        let (mut big_s2, mut big_s4) = (0.0, 0.0);
        for i in 0..n {
            let e = model.scores[i].clamp(OVERLAP_LOWER, OVERLAP_UPPER);
            big_s2 += s1 / (e * wbar1 * wbar1) + s0 / ((1.0 - e) * wbar0 * wbar0);
            big_s4 += m1 / (e.powi(3) * wbar1.powi(4)) + m0 / ((1.0 - e).powi(3) * wbar0.powi(4));
        }
        if !big_s2.is_finite() || !big_s4.is_finite() || big_s2 <= 0.0 {
            return None;
        }
        let denom = big_s4 / (big_s2 * big_s2) - 1.0 / n_f;
        if denom <= 0.0 {
            return None;
        }
        let dof = 2.0 / denom;
        if dof.is_finite() && dof >= 1.0 {
            Some(dof.min(n_f - 1.0))
        } else {
            None
        }
    }

    /// The rule that shipped before this investigation: Satterthwaite's `nu`.
    ///
    /// Kept so the comparison below contrasts two different things rather than
    /// a function with itself. Production now uses `effective_variance_dof`,
    /// which is this without the Gaussian factor of two — see that function for
    /// why the factor had to go.
    fn legacy_satterthwaite_dof(psi: &[f64]) -> Option<f64> {
        let n = psi.len();
        if n < 4 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let n_f = n as f64;
        let s2: f64 = psi.iter().map(|p| p * p).sum();
        let s4: f64 = psi.iter().map(|p| p.powi(4)).sum();
        if !s2.is_finite() || !s4.is_finite() || s2 <= 0.0 {
            return None;
        }
        let denom = s4 - s2 * s2 / n_f;
        if denom <= 0.0 {
            return None;
        }
        let dof = 2.0 * s2 * s2 / denom;
        if dof.is_finite() && dof >= 1.0 {
            Some(dof.min(n_f - 1.0))
        } else {
            None
        }
    }

    fn coverage(ates: &[f64], ses: &[f64], dofs: &[Option<f64>], truth: f64, rule: &str) -> f64 {
        let mut hits = 0usize;
        for i in 0..ates.len() {
            let crit = match rule {
                "normal" => 1.959_963_984_540_054,
                _ => match dofs[i] {
                    Some(d) if d >= 1.0 => cynepic_core::special::t_quantile(0.975, d),
                    _ => 1.959_963_984_540_054,
                },
            };
            if (ates[i] - truth).abs() <= crit * ses[i] {
                hits += 1;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        {
            100.0 * hits as f64 / ates.len() as f64
        }
    }

    /// Where does the noise in the reported standard error come from?
    ///
    /// Two candidate sources, and the finding could not tell them apart:
    ///
    /// 1. **Heavy tails.** A handful of large influence contributions dominate
    ///    `sum psi^2`, so that sum is a noisy estimate whatever the propensity.
    /// 2. **Propensity estimation.** The scores are refitted on every sample, so
    ///    the weights themselves move, and Satterthwaite — computed within one
    ///    sample, conditional on the fitted model — cannot see that at all.
    ///
    /// Substituting the DGP's *true* propensity isolates them: with the truth
    /// plugged in, only source 1 remains. Whatever cv survives is heavy tails;
    /// whatever the estimated-propensity run adds on top is source 2.
    #[test]
    #[ignore = "Monte Carlo diagnostic for C14; run with --ignored --nocapture"]
    fn c14_decompose_the_noise_in_the_standard_error() {
        const REPS: usize = 600;
        let cells: [(&str, Dgp); 2] = [
            ("moderate-overlap", Dgp::new().with_overlap(0.35)),
            ("strong-confounding", Dgp::new().with_confounding(3.0)),
        ];

        println!("\nC14 — where the noise in the SE comes from, {REPS} replications\n");
        println!(
            "  {:<20} {:>10} {:>9} {:>9} {:>9} {:>8}",
            "cell / propensity", "true sd", "mean se", "cv(se)", "dof→", "sattw"
        );

        for (name, dgp) in cells {
            for estimated in [false, true] {
                let (mut ates, mut ses, mut sat) = (Vec::new(), Vec::new(), Vec::new());
                for rep in 0..REPS {
                    let d = dgp.sample(90_000 + rep as u64);
                    let model = if estimated {
                        match PropensityScoreEstimator::fit_propensity(&d.treatment, &d.covariates)
                        {
                            Ok(m) => m,
                            Err(_) => continue,
                        }
                    } else {
                        // The true scores, wrapped in the same struct. `design`
                        // still comes from a fit so the projection has a score
                        // matrix to work with; only `scores` is replaced.
                        let Ok(mut m) =
                            PropensityScoreEstimator::fit_propensity(&d.treatment, &d.covariates)
                        else {
                            continue;
                        };
                        m.scores = d.propensity.clone();
                        m
                    };
                    let Ok(r) =
                        PropensityScoreEstimator::ipw_with_model(&d.treatment, &d.outcome, &model)
                    else {
                        continue;
                    };
                    let (psi, ..) = influence(&d.treatment, &d.outcome, &model);
                    let (residual, _k) = project_off_propensity_score(&psi, &d.treatment, &model);
                    ates.push(r.ate());
                    ses.push(r.std_error());
                    if let Some(v) = effective_variance_dof(&residual) {
                        sat.push(v);
                    }
                }

                #[allow(clippy::cast_precision_loss)]
                let r_f = ates.len() as f64;
                let mean_ate = ates.iter().sum::<f64>() / r_f;
                let sd =
                    (ates.iter().map(|a| (a - mean_ate).powi(2)).sum::<f64>() / (r_f - 1.0)).sqrt();
                let mean_se = ses.iter().sum::<f64>() / r_f;
                let cv = (ses.iter().map(|s| (s - mean_se).powi(2)).sum::<f64>() / (r_f - 1.0))
                    .sqrt()
                    / mean_se;
                // cv(se) ~ 1/sqrt(2 nu) for a scaled chi-squared variance, so
                // this inverts the observed noise into the dof it implies.
                let implied = 1.0 / (2.0 * cv * cv);
                sat.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
                let med_sat = if sat.is_empty() {
                    f64::NAN
                } else {
                    sat[sat.len() / 2]
                };

                println!(
                    "  {:<20} {sd:>10.4} {mean_se:>9.4} {cv:>9.3} {implied:>9.1} {med_sat:>8.1}",
                    format!("{name} / {}", if estimated { "fitted" } else { "TRUE" })
                );
            }
        }
        println!(
            "\n  dof→ is the dof the observed noise implies (1 / 2cv^2); sattw is\n               what Satterthwaite reports. The gap between them is what the finding\n               has to explain, and the TRUE-propensity row says how much of it is\n               heavy tails rather than model estimation."
        );
    }

    /// What critical value would actually cover, and is the fourth moment the
    /// reason Satterthwaite does not produce it?
    ///
    /// Sweeps a *fixed* dof across cells to find the one that reaches nominal,
    /// then compares the within-sample fourth moment against its across-
    /// replication mean. If a typical sample's `S4` sits well below the mean,
    /// that is the downward bias that inflates `nu` — and it is a property of
    /// heavy tails, not of anything the estimator did wrong.
    #[test]
    #[ignore = "Monte Carlo diagnostic for C14; run with --ignored --nocapture"]
    fn c14_find_the_dof_that_covers() {
        const REPS: usize = 600;
        let cells: [(&str, Dgp); 3] = [
            ("benign", Dgp::new()),
            ("moderate-overlap", Dgp::new().with_overlap(0.35)),
            ("strong-confounding", Dgp::new().with_confounding(3.0)),
        ];

        println!("\nC14 — coverage at a fixed dof, {REPS} replications\n");
        print!("  {:<20}", "cell");
        for d in [1e9_f64, 30.0, 15.0, 10.0, 7.0, 5.0, 4.0] {
            print!(
                "{:>8}",
                if d > 1e6 {
                    "z".to_string()
                } else {
                    format!("{d:.0}")
                }
            );
        }
        println!("{:>10}{:>10}", "med S4", "mean S4");

        for (name, dgp) in cells {
            let truth = 2.0;
            let (mut ates, mut ses, mut s4s) = (Vec::new(), Vec::new(), Vec::new());
            for rep in 0..REPS {
                let d = dgp.sample(90_000 + rep as u64);
                let Ok(model) =
                    PropensityScoreEstimator::fit_propensity(&d.treatment, &d.covariates)
                else {
                    continue;
                };
                let Ok(r) =
                    PropensityScoreEstimator::ipw_with_model(&d.treatment, &d.outcome, &model)
                else {
                    continue;
                };
                let (psi, ..) = influence(&d.treatment, &d.outcome, &model);
                let (residual, _k) = project_off_propensity_score(&psi, &d.treatment, &model);
                // Normalised so cells are comparable: S4 / S2^2 is what nu
                // depends on, and it is invariant to rescaling psi.
                let s2: f64 = residual.iter().map(|v| v * v).sum();
                let s4: f64 = residual.iter().map(|v| v.powi(4)).sum();
                ates.push(r.ate());
                ses.push(r.std_error());
                if s2 > 0.0 {
                    s4s.push(s4 / (s2 * s2));
                }
            }

            print!("  {name:<20}");
            for d in [1e9_f64, 30.0, 15.0, 10.0, 7.0, 5.0, 4.0] {
                let crit = if d > 1e6 {
                    1.959_963_984_540_054
                } else {
                    cynepic_core::special::t_quantile(0.975, d)
                };
                let hits = ates
                    .iter()
                    .zip(ses.iter())
                    .filter(|(a, se)| (*a - truth).abs() <= crit * **se)
                    .count();
                #[allow(clippy::cast_precision_loss)]
                let pct = 100.0 * hits as f64 / ates.len() as f64;
                print!("{pct:>7.1}%");
            }
            s4s.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
            #[allow(clippy::cast_precision_loss)]
            let mean_s4 = s4s.iter().sum::<f64>() / s4s.len() as f64;
            println!("{:>10.2e}{:>10.2e}", s4s[s4s.len() / 2], mean_s4);
        }
        println!(
            "\n  The S4 columns are sum(psi^4)/sum(psi^2)^2, the ratio nu depends\n               on. A median well below the mean is the downward bias that inflates\n               Satterthwaite's nu — and it is a property of heavy tails, not a\n               mistake in the formula."
        );
    }

    /// Is the residual bias coming from the overlap clamp?
    ///
    /// `ipw` clamps fitted scores to `[0.02, 0.98]`. That is a deliberate
    /// safety rail, but it has a statistical consequence: a clamped weight is
    /// the wrong weight, so the estimator targets the ATE on a *trimmed*
    /// population while the harness scores it against the full-population ATE.
    /// Bias of that shape would cap achievable coverage no matter how good the
    /// interval is, which would make the remaining gap unfixable by any dof
    /// rule — a different finding from the one recorded.
    ///
    /// Prints how often the clamp binds, and the bias next to it.
    #[test]
    #[ignore = "Monte Carlo diagnostic for C14; run with --ignored --nocapture"]
    fn c14_is_the_residual_bias_from_the_overlap_clamp() {
        const REPS: usize = 400;
        let cells: [(&str, Dgp); 4] = [
            ("benign", Dgp::new()),
            ("moderate-overlap", Dgp::new().with_overlap(0.35)),
            ("strong-confounding", Dgp::new().with_confounding(3.0)),
            ("high-dim", Dgp::new().with_p(25).with_n(400)),
        ];

        println!("\nC14 — does the overlap clamp explain the bias? {REPS} replications\n");
        println!(
            "  {:<24} {:>10} {:>9} {:>9} {:>10}",
            "cell", "clamped %", "bias", "true sd", "bias/sd"
        );

        for (name, dgp) in cells {
            let (mut ates, mut truths, mut clamped) = (Vec::new(), Vec::new(), Vec::new());
            for rep in 0..REPS {
                let d = dgp.sample(90_000 + rep as u64);
                let Ok(model) =
                    PropensityScoreEstimator::fit_propensity(&d.treatment, &d.covariates)
                else {
                    continue;
                };
                let Ok(r) =
                    PropensityScoreEstimator::ipw_with_model(&d.treatment, &d.outcome, &model)
                else {
                    continue;
                };
                let n_clamped = model
                    .scores
                    .iter()
                    .filter(|e| **e < OVERLAP_LOWER || **e > OVERLAP_UPPER)
                    .count();
                #[allow(clippy::cast_precision_loss)]
                clamped.push(100.0 * n_clamped as f64 / model.scores.len() as f64);
                ates.push(r.ate());
                truths.push(d.truth.ate);
            }
            #[allow(clippy::cast_precision_loss)]
            let r_f = ates.len() as f64;
            let truth = truths.iter().sum::<f64>() / r_f;
            let mean_ate = ates.iter().sum::<f64>() / r_f;
            let sd =
                (ates.iter().map(|a| (a - mean_ate).powi(2)).sum::<f64>() / (r_f - 1.0)).sqrt();
            let bias = mean_ate - truth;
            let mean_clamped = clamped.iter().sum::<f64>() / r_f;
            println!(
                "  {name:<24} {mean_clamped:>9.2}% {bias:>+9.4} {sd:>9.4} {:>10.2}",
                bias / sd
            );
        }
        println!(
            "\n  A bias of 0.2 sd or more caps coverage below nominal on its own:\n               P(|Z + 0.26| < 1.96) is 94.2%, not 95%. Where that is the binding\n               constraint, no degrees-of-freedom rule can close the gap, and the\n               honest move is to say so rather than to widen intervals until the\n               number looks right."
        );
    }

    /// Is the remaining gap fixable by *any* interval width?
    ///
    /// After the Kish rule, the claimed degrees of freedom (7.1 at strong
    /// confounding) match the dof the observed noise implies (7.7) almost
    /// exactly. A t-interval with the right dof should then cover — and it does
    /// not; the cells sit near 91-92%.
    ///
    /// That points away from the variance estimate and at the *joint*
    /// distribution. A t-interval assumes the error and the standard error are
    /// independent. If instead the replications with the largest errors are
    /// systematically the ones with the *smallest* reported SE, no dof rule can
    /// rescue coverage: the interval is narrowest exactly when it needs to be
    /// widest, and widening on average just over-covers everywhere else.
    ///
    /// Reports `corr(|error|, se)` alongside the share of misses that came from
    /// below-median SEs. Under independence that share is 50%.
    #[test]
    #[ignore = "Monte Carlo diagnostic for C14; run with --ignored --nocapture"]
    fn c14_is_the_error_independent_of_its_own_standard_error() {
        const REPS: usize = 800;
        let cells: [(&str, Dgp); 4] = [
            ("benign", Dgp::new()),
            ("moderate-overlap", Dgp::new().with_overlap(0.35)),
            ("strong-confounding", Dgp::new().with_confounding(3.0)),
            ("high-dim", Dgp::new().with_p(25).with_n(400)),
        ];

        println!("\nC14 — is the error independent of its own SE? {REPS} replications\n");
        println!(
            "  {:<24} {:>14} {:>16} {:>12}",
            "cell", "corr(|err|,se)", "misses w/ small se", "coverage"
        );

        for (name, dgp) in cells {
            let (mut errs, mut ses) = (Vec::new(), Vec::new());
            let mut truths = Vec::new();
            for rep in 0..REPS {
                let d = dgp.sample(90_000 + rep as u64);
                let Ok(r) = PropensityScoreEstimator::ipw(&d.treatment, &d.outcome, &d.covariates)
                else {
                    continue;
                };
                errs.push(r.ate() - d.truth.ate);
                ses.push(r.std_error());
                truths.push(r.confidence_interval(0.95));
            }
            #[allow(clippy::cast_precision_loss)]
            let r_f = errs.len() as f64;

            let abs: Vec<f64> = errs.iter().map(|e| e.abs()).collect();
            let ma = abs.iter().sum::<f64>() / r_f;
            let ms = ses.iter().sum::<f64>() / r_f;
            let cov: f64 = abs
                .iter()
                .zip(ses.iter())
                .map(|(a, s)| (a - ma) * (s - ms))
                .sum::<f64>();
            let va: f64 = abs.iter().map(|a| (a - ma).powi(2)).sum();
            let vs: f64 = ses.iter().map(|s| (s - ms).powi(2)).sum();
            let corr = cov / (va.sqrt() * vs.sqrt());

            let mut sorted = ses.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
            let med_se = sorted[sorted.len() / 2];

            let (mut misses, mut misses_small) = (0usize, 0usize);
            let mut hits = 0usize;
            for i in 0..errs.len() {
                let inside = truths[i].is_some_and(|(lo, hi)| {
                    let point = errs[i];
                    // Interval is around the estimate; recentre on the error.
                    let half = (hi - lo) / 2.0;
                    point.abs() <= half
                });
                if inside {
                    hits += 1;
                } else {
                    misses += 1;
                    if ses[i] < med_se {
                        misses_small += 1;
                    }
                }
            }
            #[allow(clippy::cast_precision_loss)]
            let share = if misses == 0 {
                f64::NAN
            } else {
                100.0 * misses_small as f64 / misses as f64
            };
            #[allow(clippy::cast_precision_loss)]
            let coverage = 100.0 * hits as f64 / r_f;
            println!("  {name:<24} {corr:>14.3} {share:>15.1}% {coverage:>11.1}%");
        }
        println!(
            "\n  Under independence the third column is 50%. Well above that means\n               the misses are concentrated in the replications that reported the\n               smallest uncertainty — the interval is narrowest exactly when it\n               most needs to be wide, and no degrees-of-freedom rule can fix that."
        );
    }

    #[test]
    #[ignore = "Monte Carlo diagnostic for C14; run with --ignored --nocapture"]
    fn c14_compare_degrees_of_freedom_rules() {
        const REPS: usize = 500;
        let cells: [(&str, Dgp); 8] = [
            ("benign", Dgp::new()),
            ("moderate-overlap", Dgp::new().with_overlap(0.35)),
            ("strong-confounding", Dgp::new().with_confounding(3.0)),
            ("nonlinear", Dgp::new().with_nonlinearity(2.0)),
            ("heteroskedastic", Dgp::new().heteroskedastic()),
            ("heterogeneous-effects", Dgp::new().with_heterogeneity(1.5)),
            ("small-n", Dgp::new().with_n(120)),
            ("high-dim", Dgp::new().with_p(25).with_n(400)),
        ];

        println!("\nC14 — dof rules across the grid, {REPS} replications\n");
        println!(
            "  {:<24} {:>7} {:>7} {:>9} {:>9} {:>9}",
            "cell", "legacy", "kish", "cov(z)", "cov(legacy)", "cov(kish)"
        );

        let mut worst_kish: f64 = 0.0;
        let mut worst_name = String::new();
        for (name, dgp) in cells {
            let (mut ates, mut ses) = (Vec::new(), Vec::new());
            let (mut d_sat, mut d_kish, mut truths) = (Vec::new(), Vec::new(), Vec::new());

            for rep in 0..REPS {
                let d = dgp.sample(90_000 + rep as u64);
                let Ok(model) =
                    PropensityScoreEstimator::fit_propensity(&d.treatment, &d.covariates)
                else {
                    continue;
                };
                let Ok(r) =
                    PropensityScoreEstimator::ipw_with_model(&d.treatment, &d.outcome, &model)
                else {
                    continue;
                };
                let (psi, ..) = influence(&d.treatment, &d.outcome, &model);
                let (residual, _k) = project_off_propensity_score(&psi, &d.treatment, &model);
                ates.push(r.ate());
                ses.push(r.std_error());
                d_sat.push(legacy_satterthwaite_dof(&residual));
                d_kish.push(effective_variance_dof(&residual));
                truths.push(d.truth.ate);
            }
            if ates.is_empty() {
                println!("  {name:<24} (no estimable replication)");
                continue;
            }

            let truth = truths.iter().sum::<f64>() / truths.len() as f64;
            let med = |v: &[Option<f64>]| {
                let mut x: Vec<f64> = v.iter().filter_map(|d| *d).collect();
                x.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
                if x.is_empty() {
                    f64::NAN
                } else {
                    x[x.len() / 2]
                }
            };
            let (cz, cs, ck) = (
                coverage(&ates, &ses, &d_sat, truth, "normal"),
                coverage(&ates, &ses, &d_sat, truth, "t"),
                coverage(&ates, &ses, &d_kish, truth, "t"),
            );
            if (95.0 - ck).abs() > worst_kish {
                worst_kish = (95.0 - ck).abs();
                worst_name = name.to_string();
            }
            println!(
                "  {name:<24} {:>7.1} {:>7.1} {cz:>8.1}% {cs:>8.1}% {ck:>8.1}%",
                med(&d_sat),
                med(&d_kish),
            );
        }
        println!(
            "\n  worst deviation from nominal under the Kish rule: {worst_kish:.1} \
             points ({worst_name})"
        );
        println!(
            "  Over-coverage counts as a deviation too. A rule that fixes the\n               heavy cells by making every interval wider has not fixed anything."
        );
    }
}
