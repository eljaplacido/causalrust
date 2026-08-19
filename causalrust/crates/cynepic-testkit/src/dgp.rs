//! Data-generating processes with known ground truth.
//!
//! Each [`Dgp`] describes a world: how confounders are drawn, how treatment is
//! assigned given those confounders, and how the outcome responds. Because we
//! write the rules, we know the true causal effect exactly — which is what makes
//! validation possible.
//!
//! The knobs on [`Dgp`] are not arbitrary. Each one corresponds to a way real
//! estimators fail:
//!
//! | Knob | Breaks |
//! |---|---|
//! | `confounding` | Naive difference-in-means; the stronger it is, the larger the bias |
//! | `overlap` | IPW — weak overlap means extreme weights and a collapsing effective sample |
//! | `nonlinearity` | Linear OLS adjustment; the outcome model is misspecified |
//! | `heteroskedastic` | Classical standard errors, hence coverage |
//! | `heterogeneity` | The ATE/ATT/LATE distinction — they diverge as this grows |
//! | `collinearity` | The solver itself (finding C1) |
//! | `instrument_strength` | 2SLS — below the Stock–Yogo threshold it is unusable |

use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// A simulated dataset together with the truth that generated it.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Binary treatment indicator, one entry per unit.
    pub treatment: Array1<f64>,
    /// Observed outcome, one entry per unit.
    pub outcome: Array1<f64>,
    /// Observed covariates, `n × p`.
    pub covariates: Array2<f64>,
    /// Instrument, when the DGP defines one (for IV/2SLS validation).
    pub instrument: Option<Array1<f64>>,
    /// The **true** propensity `P(T=1 | X)` used to assign each unit.
    ///
    /// Not available to an estimator in the real world, which is the point:
    /// having it here lets a test check whether the propensity *model* recovered
    /// the truth, separately from whether the effect estimate came out right.
    /// That separation is what distinguishes "the weights were wrong" from
    /// "the weights were right and the variance formula was wrong" (finding C5).
    pub propensity: Array1<f64>,
    /// The effect values that actually generated this data.
    pub truth: GroundTruth,
    /// Seed that produced this dataset. Print it in any failure message.
    pub seed: u64,
}

impl Dataset {
    /// Number of units.
    pub fn n(&self) -> usize {
        self.treatment.len()
    }

    /// Number of observed covariates.
    pub fn p(&self) -> usize {
        self.covariates.ncols()
    }

    /// Units assigned to treatment.
    pub fn n_treated(&self) -> usize {
        self.treatment.iter().filter(|&&t| t > 0.5).count()
    }

    /// Fraction of units whose true propensity lies within `eps` of 0 or 1.
    ///
    /// This is the honest measure of a positivity (overlap) violation. The
    /// marginal treated share can sit at a comfortable 50% while individual
    /// propensities are pinned at the extremes — and it is the individual
    /// extremes that make inverse-probability weights explode.
    pub fn extreme_propensity_fraction(&self, eps: f64) -> f64 {
        let k = self
            .propensity
            .iter()
            .filter(|&&e| e < eps || e > 1.0 - eps)
            .count();
        k as f64 / self.n() as f64
    }

    /// Effective sample size of the true inverse-probability weights,
    /// `(Σw)² / Σw²`.
    ///
    /// # This is not an overlap diagnostic
    ///
    /// ESS measures how concentrated the *realised* weights are. It is the right
    /// thing to report alongside an IPW estimate, because it says how much
    /// information the weighted sample actually carries.
    ///
    /// It is the wrong thing to use for detecting a positivity violation, and
    /// the reason is counter-intuitive enough to be worth stating: as assignment
    /// becomes near-deterministic, almost every unit receives the arm it was
    /// nearly certain to receive, so almost every weight is close to 1 and ESS
    /// looks *excellent*. Meanwhile the estimand is not identified at all,
    /// because whole regions of covariate space contain no comparison units.
    ///
    /// Use [`Dataset::extreme_propensity_fraction`] to detect overlap failure.
    /// Use this to report precision once overlap is established.
    pub fn true_weight_ess(&self) -> f64 {
        let w: Vec<f64> = self
            .treatment
            .iter()
            .zip(self.propensity.iter())
            .map(|(&t, &e)| if t > 0.5 { 1.0 / e } else { 1.0 / (1.0 - e) })
            .collect();
        let sum: f64 = w.iter().sum();
        let sum_sq: f64 = w.iter().map(|x| x * x).sum();
        if sum_sq <= 0.0 {
            0.0
        } else {
            sum * sum / sum_sq
        }
    }
}

/// The true effects, known because we generated the data.
///
/// These are *different numbers* whenever effects are heterogeneous, which is
/// exactly why finding C3 matters: an estimator that reports a LATE as an ATE is
/// answering a different question than the one asked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundTruth {
    /// Average treatment effect over the whole population.
    pub ate: f64,
    /// Average effect among the treated.
    pub att: f64,
    /// Average effect among the untreated.
    pub atc: f64,
    /// Local average treatment effect — compliers only. `None` without an instrument.
    pub late: Option<f64>,
}

/// A world to simulate. Build with [`Dgp::new`] and the `with_*` methods.
#[derive(Debug, Clone)]
pub struct Dgp {
    /// Units to generate.
    pub n: usize,
    /// Observed covariates.
    pub p: usize,
    /// Baseline treatment effect.
    pub effect: f64,
    /// How strongly covariates drive treatment assignment. 0.0 = randomised.
    pub confounding: f64,
    /// Propensity overlap in `(0.0, 1.0]`. `1.0` is good overlap; smaller
    /// values squeeze propensities toward 0 and 1, which is what destroys IPW.
    ///
    /// The assignment index is standardised before this is applied, so the
    /// meaning of a given value does not drift with `p` or `confounding`.
    /// Without that, raising `p` from 3 to 25 silently turned a well-overlapped
    /// world into a positivity violation, and any estimator measured on the
    /// high-dimensional cell was being scored on a different question from the
    /// one the cell claimed to ask.
    pub overlap: f64,
    /// Strength of nonlinear outcome terms. 0.0 = linear model is correct.
    pub nonlinearity: f64,
    /// Whether outcome noise scales with the covariates.
    pub heteroskedastic: bool,
    /// How much the effect varies across units. 0.0 = ATE == ATT == ATC.
    pub heterogeneity: f64,
    /// Append a covariate that is an exact linear combination of the others.
    /// Triggers the rank-deficiency path (finding C1).
    pub collinearity: bool,
    /// Instrument strength. `None` = no instrument generated.
    pub instrument_strength: Option<f64>,
    /// Residual outcome noise scale.
    pub noise: f64,
}

impl Default for Dgp {
    /// A benign world: moderate confounding, good overlap, linear, homoskedastic,
    /// homogeneous effects. Every estimator should succeed here. An estimator
    /// that fails the default DGP is broken outright.
    fn default() -> Self {
        Self {
            n: 2_000,
            p: 3,
            effect: 2.0,
            confounding: 1.0,
            overlap: 1.0,
            nonlinearity: 0.0,
            heteroskedastic: false,
            heterogeneity: 0.0,
            collinearity: false,
            instrument_strength: None,
            noise: 1.0,
        }
    }
}

impl Dgp {
    /// A new DGP with default (benign) settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of units.
    pub fn with_n(mut self, n: usize) -> Self {
        self.n = n;
        self
    }

    /// Set the number of observed covariates.
    pub fn with_p(mut self, p: usize) -> Self {
        self.p = p;
        self
    }

    /// Set the baseline treatment effect (the truth to recover).
    pub fn with_effect(mut self, effect: f64) -> Self {
        self.effect = effect;
        self
    }

    /// Set confounding strength. 0.0 makes assignment ignorable.
    pub fn with_confounding(mut self, c: f64) -> Self {
        self.confounding = c;
        self
    }

    /// Set propensity overlap. `1.0` is good; values near 0 create a positivity
    /// violation.
    pub fn with_overlap(mut self, o: f64) -> Self {
        self.overlap = o.clamp(0.01, 1.0);
        self
    }

    /// Add nonlinear outcome structure, misspecifying a linear outcome model.
    pub fn with_nonlinearity(mut self, k: f64) -> Self {
        self.nonlinearity = k;
        self
    }

    /// Make outcome noise depend on the covariates.
    pub fn heteroskedastic(mut self) -> Self {
        self.heteroskedastic = true;
        self
    }

    /// Make the treatment effect vary across units, separating ATE from ATT.
    pub fn with_heterogeneity(mut self, h: f64) -> Self {
        self.heterogeneity = h;
        self
    }

    /// Append an exactly collinear covariate (finding C1's trigger).
    pub fn collinear(mut self) -> Self {
        self.collinearity = true;
        self
    }

    /// Generate an instrument of the given strength for IV validation.
    pub fn with_instrument(mut self, strength: f64) -> Self {
        self.instrument_strength = Some(strength);
        self
    }

    /// Set residual outcome noise.
    pub fn with_noise(mut self, s: f64) -> Self {
        self.noise = s;
        self
    }

    /// Draw one dataset from this world.
    ///
    /// Deterministic in `seed`: the same seed yields byte-identical data on every
    /// platform, so any failure is reproducible from the seed alone.
    pub fn sample(&self, seed: u64) -> Dataset {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).expect("N(0,1) is always valid");

        let n = self.n;
        let p_obs = self.p;
        let p_total = if self.collinearity { p_obs + 1 } else { p_obs };

        let mut covariates = Array2::zeros((n, p_total));
        let mut treatment = Array1::zeros(n);
        let mut outcome = Array1::zeros(n);
        let mut propensity = Array1::zeros(n);
        let mut instrument = self.instrument_strength.map(|_| Array1::zeros(n));

        // Per-unit effects, needed to compute the true ATT/ATC exactly.
        let mut unit_effects = Vec::with_capacity(n);
        let mut treated_effects = Vec::new();
        let mut control_effects = Vec::new();
        // Compliers: units whose treatment status the instrument actually moves.
        let mut complier_effects = Vec::new();

        for i in 0..n {
            // --- covariates -------------------------------------------------
            let mut x = vec![0.0; p_total];
            for xj in x.iter_mut().take(p_obs) {
                *xj = normal.sample(&mut rng);
            }
            if self.collinearity {
                // Exactly the sum of the first two — rank deficient by construction.
                x[p_obs] = if p_obs >= 2 { x[0] + x[1] } else { x[0] };
            }
            for (j, &xj) in x.iter().enumerate() {
                covariates[[i, j]] = xj;
            }

            // --- treatment assignment ---------------------------------------
            //
            // The index is standardised by sqrt(p) so that its variance does
            // not grow with the number of covariates. Without this, the logit's
            // spread scaled as sqrt(p) and the `high-dim` cell became a severe
            // positivity violation purely because it had more columns — which
            // meant it was no longer testing dimensionality, it was testing
            // overlap, and reporting the answer under the wrong label.
            #[allow(clippy::cast_precision_loss)]
            let scale = (p_obs.max(1) as f64).sqrt();
            let index: f64 = x.iter().take(p_obs).sum::<f64>() * self.confounding / scale;
            // `overlap` now maps directly onto the logit slope: 1.0 keeps
            // propensities in roughly [0.2, 0.8]; small values push them to the
            // extremes, where IPW weights explode.
            let e_i = logistic(index * 0.5 / self.overlap.max(1e-3));
            propensity[i] = e_i;

            let (t, is_complier, z) = match self.instrument_strength {
                Some(strength) => {
                    let z_i = if rng.random::<f64>() < 0.5 { 1.0 } else { 0.0 };
                    // Compliers respond to the instrument; the rest are always-
                    // or never-takers determined by the propensity alone.
                    let complier = rng.random::<f64>() < strength;
                    let t_i = if complier {
                        z_i
                    } else if rng.random::<f64>() < e_i {
                        1.0
                    } else {
                        0.0
                    };
                    (t_i, complier, Some(z_i))
                }
                None => {
                    let t_i = if rng.random::<f64>() < e_i { 1.0 } else { 0.0 };
                    (t_i, false, None)
                }
            };
            treatment[i] = t;
            if let (Some(inst), Some(z_i)) = (instrument.as_mut(), z) {
                inst[i] = z_i;
            }

            // --- outcome -----------------------------------------------------
            // Unit-level effect: base plus a covariate-driven deviation.
            let tau_i = self.effect + self.heterogeneity * x[0];
            unit_effects.push(tau_i);
            if t > 0.5 {
                treated_effects.push(tau_i);
            } else {
                control_effects.push(tau_i);
            }
            if is_complier {
                complier_effects.push(tau_i);
            }

            let linear: f64 = x.iter().take(p_obs).map(|v| 1.5 * v).sum();
            let nonlinear = self.nonlinearity * x[0].powi(2);
            let sigma = if self.heteroskedastic {
                self.noise * (1.0 + x[0].abs())
            } else {
                self.noise
            };
            outcome[i] = linear + nonlinear + tau_i * t + sigma * normal.sample(&mut rng);
        }

        let truth = GroundTruth {
            ate: mean(&unit_effects),
            att: mean(&treated_effects),
            atc: mean(&control_effects),
            late: if complier_effects.is_empty() {
                None
            } else {
                Some(mean(&complier_effects))
            },
        };

        Dataset {
            treatment,
            outcome,
            covariates,
            instrument,
            propensity,
            truth,
            seed,
        }
    }
}

/// A factorial sweep over the conditions that break estimators.
///
/// Running a validation across the grid produces a *map* of where an estimator
/// is trustworthy and where it degrades. Publishing that map — including the
/// cells where it degrades — is the credibility artifact.
#[derive(Debug, Clone)]
pub struct DgpGrid {
    /// Every cell in the sweep, each with a human-readable label.
    pub cells: Vec<(String, Dgp)>,
}

impl Default for DgpGrid {
    fn default() -> Self {
        Self::standard()
    }
}

impl DgpGrid {
    /// The standard grid used by the published coverage tables.
    ///
    /// Deliberately small enough to run in CI. The full factorial sweep used for
    /// release artifacts is generated by [`DgpGrid::factorial`].
    pub fn standard() -> Self {
        let cells = vec![
            ("benign".into(), Dgp::new()),
            (
                "strong-confounding".into(),
                Dgp::new().with_confounding(3.0),
            ),
            // 4% of units extreme; every estimator should still cope.
            ("moderate-overlap".into(), Dgp::new().with_overlap(0.35)),
            // 71% extreme, ESS ~3 of 2000. Weighting is not valid here, and an
            // estimator that returns a confident number is the failure.
            ("weak-overlap".into(), Dgp::new().with_overlap(0.06)),
            ("nonlinear".into(), Dgp::new().with_nonlinearity(2.0)),
            ("heteroskedastic".into(), Dgp::new().heteroskedastic()),
            (
                "heterogeneous-effects".into(),
                Dgp::new().with_heterogeneity(1.5),
            ),
            ("small-n".into(), Dgp::new().with_n(120)),
            ("high-dim".into(), Dgp::new().with_p(25).with_n(400)),
            ("weak-instrument".into(), Dgp::new().with_instrument(0.05)),
            ("strong-instrument".into(), Dgp::new().with_instrument(0.7)),
        ];
        Self { cells }
    }

    /// Full factorial sweep across the listed axes. Large; for release runs.
    pub fn factorial() -> Self {
        let mut cells = Vec::new();
        for &conf in &[0.0, 1.0, 3.0] {
            for &ovl in &[0.35, 0.12] {
                for &nonlin in &[0.0, 2.0] {
                    for &het in &[0.0, 1.5] {
                        for &n in &[500usize, 5_000] {
                            let label = format!("conf{conf}_ovl{ovl}_nl{nonlin}_het{het}_n{n}");
                            cells.push((
                                label,
                                Dgp::new()
                                    .with_confounding(conf)
                                    .with_overlap(ovl)
                                    .with_nonlinearity(nonlin)
                                    .with_heterogeneity(het)
                                    .with_n(n),
                            ));
                        }
                    }
                }
            }
        }
        Self { cells }
    }
}

fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_identical_data() {
        let dgp = Dgp::new().with_n(200);
        let a = dgp.sample(42);
        let b = dgp.sample(42);
        assert_eq!(a.treatment, b.treatment);
        assert_eq!(a.outcome, b.outcome);
        assert_eq!(a.covariates, b.covariates);
    }

    #[test]
    fn different_seeds_give_different_data() {
        let dgp = Dgp::new().with_n(200);
        assert_ne!(dgp.sample(1).outcome, dgp.sample(2).outcome);
    }

    #[test]
    fn homogeneous_effects_collapse_all_estimands() {
        // With no heterogeneity, ATE == ATT == ATC exactly. This is the
        // condition under which finding C3 would be harmless.
        let d = Dgp::new().with_heterogeneity(0.0).with_n(3_000).sample(7);
        assert!((d.truth.ate - d.truth.att).abs() < 1e-9);
        assert!((d.truth.ate - d.truth.atc).abs() < 1e-9);
    }

    #[test]
    fn heterogeneous_effects_separate_ate_from_att() {
        // With confounding AND heterogeneity, the treated are selected on the
        // same covariate that drives the effect, so ATT must differ from ATE.
        // This is the situation that makes mislabelling an estimand a real error.
        let d = Dgp::new()
            .with_heterogeneity(2.0)
            .with_confounding(2.0)
            .with_n(5_000)
            .sample(11);
        assert!(
            (d.truth.ate - d.truth.att).abs() > 0.1,
            "ATE {} and ATT {} should diverge under selection on effect modifier",
            d.truth.ate,
            d.truth.att
        );
    }

    #[test]
    fn weak_overlap_produces_extreme_propensities() {
        // Low overlap must push *individual* propensities toward 0 and 1. Note
        // it does NOT move the marginal treated share: the covariates are
        // symmetric about zero, so roughly half the units are pinned near 1 and
        // half near 0, and the overall split stays near 50/50.
        //
        // That is exactly why the marginal share is the wrong thing to measure —
        // a dataset can look perfectly balanced in aggregate while every single
        // inverse-probability weight is enormous.
        let balanced = Dgp::new().with_overlap(0.35).with_n(2_000).sample(3);
        let extreme = Dgp::new().with_overlap(0.03).with_n(2_000).sample(3);

        assert!(
            extreme.extreme_propensity_fraction(0.05)
                > balanced.extreme_propensity_fraction(0.05) + 0.2,
            "weak overlap should pin propensities at the extremes: balanced {:.3} vs extreme {:.3}",
            balanced.extreme_propensity_fraction(0.05),
            extreme.extreme_propensity_fraction(0.05)
        );
    }

    /// Effective sample size is **non-monotonic** in overlap, and therefore
    /// cannot be used on its own to detect a positivity violation.
    ///
    /// Measured on this DGP at n=2000, seed 3:
    ///
    /// ```text
    /// overlap   extreme fraction   ESS
    ///   1.00          0.000       1874     healthy, and really is
    ///   0.35          0.040       1213
    ///   0.20          0.236        403     ESS collapsing
    ///   0.10          0.550          2     ESS at its worst
    ///   0.03          0.847          1
    ///   0.01          0.947        613     ESS RECOVERS while overlap is gone
    /// ```
    ///
    /// The recovery at the bottom is the trap. Under near-deterministic
    /// assignment almost every unit lands in the arm it was nearly certain to
    /// get, so its inverse-probability weight is ≈1 and ESS looks healthy — at
    /// the exact point where the estimand is least identified. A monitor
    /// thresholding on ESS alone passes the worst case and fails the middling
    /// one.
    ///
    /// This is why an overlap diagnostic needs the extreme-propensity fraction,
    /// which is monotone, alongside ESS, which reports precision *once* overlap
    /// is established.
    #[test]
    fn ess_is_non_monotonic_in_overlap_and_cannot_diagnose_it_alone() {
        let sweep = [1.0, 0.35, 0.2, 0.1, 0.03, 0.01];
        let measured: Vec<(f64, f64, f64)> = sweep
            .iter()
            .map(|&o| {
                let d = Dgp::new().with_overlap(o).with_n(2_000).sample(3);
                (o, d.extreme_propensity_fraction(0.05), d.true_weight_ess())
            })
            .collect();

        // The extreme fraction IS monotone, which is what makes it usable.
        for w in measured.windows(2) {
            assert!(
                w[1].1 >= w[0].1,
                "extreme fraction must rise as overlap worsens: {:?} then {:?}",
                w[0],
                w[1]
            );
        }

        // ESS is not. It falls, then recovers at the most severe violation.
        let worst_ess = measured.iter().map(|m| m.2).fold(f64::INFINITY, f64::min);
        let most_extreme = measured.last().expect("non-empty");
        assert!(
            most_extreme.1 > 0.9,
            "precondition: the last cell really has broken overlap ({:.3})",
            most_extreme.1
        );
        assert!(
            most_extreme.2 > worst_ess * 10.0,
            "ESS should recover at the extreme (worst {worst_ess:.0}, most-extreme {:.0});              if this ever becomes monotone, ESS alone would suffice and the guidance on              Dataset::true_weight_ess should change",
            most_extreme.2
        );
    }

    #[test]
    fn true_propensity_is_recorded_and_in_range() {
        let d = Dgp::new().with_n(500).sample(21);
        assert_eq!(d.propensity.len(), d.n());
        assert!(d.propensity.iter().all(|&e| e > 0.0 && e < 1.0));
    }

    #[test]
    fn randomised_assignment_gives_propensity_one_half() {
        // confounding = 0 means assignment ignores covariates entirely.
        let d = Dgp::new().with_confounding(0.0).with_n(300).sample(31);
        assert!(
            d.propensity.iter().all(|&e| (e - 0.5).abs() < 1e-9),
            "with no confounding every propensity must be exactly 0.5"
        );
    }

    #[test]
    fn collinear_flag_adds_a_dependent_column() {
        let d = Dgp::new().with_p(3).collinear().with_n(50).sample(5);
        assert_eq!(d.p(), 4, "collinear DGP appends one extra column");
        // Column 3 is exactly col0 + col1.
        for i in 0..d.n() {
            let expected = d.covariates[[i, 0]] + d.covariates[[i, 1]];
            assert!((d.covariates[[i, 3]] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn instrument_dgp_defines_a_late() {
        let d = Dgp::new().with_instrument(0.6).with_n(2_000).sample(9);
        assert!(d.instrument.is_some());
        assert!(d.truth.late.is_some());
    }

    #[test]
    fn standard_grid_covers_the_named_failure_modes() {
        let grid = DgpGrid::standard();
        let labels: Vec<&str> = grid.cells.iter().map(|(l, _)| l.as_str()).collect();
        for required in [
            "benign",
            "strong-confounding",
            "weak-overlap",
            "nonlinear",
            "heteroskedastic",
            "heterogeneous-effects",
            "weak-instrument",
        ] {
            assert!(labels.contains(&required), "grid missing {required}");
        }
    }
}
