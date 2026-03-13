//! MCMC samplers for non-conjugate models.
//!
//! Currently provides:
//! - Metropolis-Hastings with configurable proposal distribution
//!
//! Future:
//! - Hamiltonian Monte Carlo (HMC) via Burn autodiff
//! - No U-Turn Sampler (NUTS)

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Result of an MCMC sampling run.
#[derive(Debug, Clone)]
pub struct SamplerResult {
    /// The chain of samples.
    pub samples: Vec<f64>,
    /// Acceptance rate (should be ~0.234 for optimal MH in high dimensions).
    pub acceptance_rate: f64,
}

/// Metropolis-Hastings sampler for arbitrary log-density functions.
pub struct MetropolisHastings {
    /// Standard deviation of the Gaussian proposal.
    pub proposal_std: f64,
    /// Number of warmup (burn-in) iterations.
    pub warmup: usize,
    /// Number of sampling iterations.
    pub n_samples: usize,
}

impl MetropolisHastings {
    /// Create a new Metropolis-Hastings sampler.
    pub fn new(proposal_std: f64, warmup: usize, n_samples: usize) -> Self {
        Self {
            proposal_std,
            warmup,
            n_samples,
        }
    }

    /// Run the sampler on a log-density function.
    ///
    /// `log_density`: function computing the log of the unnormalized posterior
    /// `initial`: starting point for the chain
    pub fn sample<F>(&self, log_density: F, initial: f64) -> SamplerResult
    where
        F: Fn(f64) -> f64,
    {
        let mut rng = rand::rng();
        let proposal = Normal::new(0.0, self.proposal_std).unwrap();

        let mut current = initial;
        let mut current_log_p = log_density(current);
        let mut samples = Vec::with_capacity(self.n_samples);
        let mut accepted = 0u64;
        let total_iter = self.warmup + self.n_samples;

        for i in 0..total_iter {
            let candidate = current + proposal.sample(&mut rng);
            let candidate_log_p = log_density(candidate);

            let log_alpha = candidate_log_p - current_log_p;
            let u: f64 = rng.random();

            if u.ln() < log_alpha {
                current = candidate;
                current_log_p = candidate_log_p;
                if i >= self.warmup {
                    accepted += 1;
                }
            }

            if i >= self.warmup {
                samples.push(current);
            }
        }

        SamplerResult {
            acceptance_rate: accepted as f64 / self.n_samples as f64,
            samples,
        }
    }
}

/// Result of a multi-dimensional MCMC sampling run.
#[derive(Debug, Clone)]
pub struct MultiSamplerResult {
    /// Each inner Vec is one sample (a point in N-dimensional space).
    pub samples: Vec<Vec<f64>>,
    /// Acceptance rate during the sampling phase.
    pub acceptance_rate: f64,
    /// Dimensionality of the sample space.
    pub n_dims: usize,
}

/// Multi-dimensional Metropolis-Hastings with diagonal Gaussian proposal.
///
/// Proposes by adding independent N(0, proposal_std_i) to each dimension.
pub struct MultiDimMH {
    /// Per-dimension proposal standard deviations.
    pub proposal_stds: Vec<f64>,
    /// Number of warmup (burn-in) iterations.
    pub warmup: usize,
    /// Number of sampling iterations.
    pub n_samples: usize,
}

impl MultiDimMH {
    /// Create a new multi-dimensional Metropolis-Hastings sampler.
    pub fn new(proposal_stds: Vec<f64>, warmup: usize, n_samples: usize) -> Self {
        assert!(!proposal_stds.is_empty(), "proposal_stds must not be empty");
        Self {
            proposal_stds,
            warmup,
            n_samples,
        }
    }

    /// Run the sampler on a multi-dimensional log-density function.
    ///
    /// `log_density`: function computing the log of the unnormalized posterior given a point
    /// `initial`: starting point for the chain (must match dimensionality of proposal_stds)
    pub fn sample<F>(&self, log_density: F, initial: Vec<f64>) -> MultiSamplerResult
    where
        F: Fn(&[f64]) -> f64,
    {
        let n_dims = self.proposal_stds.len();
        assert_eq!(
            initial.len(),
            n_dims,
            "initial point dimensionality must match proposal_stds"
        );

        let mut rng = rand::rng();
        let proposals: Vec<Normal<f64>> = self
            .proposal_stds
            .iter()
            .map(|&s| Normal::new(0.0, s).unwrap())
            .collect();

        let mut current = initial;
        let mut current_log_p = log_density(&current);
        let mut samples = Vec::with_capacity(self.n_samples);
        let mut accepted = 0u64;
        let total_iter = self.warmup + self.n_samples;

        for i in 0..total_iter {
            let candidate: Vec<f64> = current
                .iter()
                .zip(proposals.iter())
                .map(|(&x, prop)| x + prop.sample(&mut rng))
                .collect();
            let candidate_log_p = log_density(&candidate);

            let log_alpha = candidate_log_p - current_log_p;
            let u: f64 = rng.random();

            if u.ln() < log_alpha {
                current = candidate;
                current_log_p = candidate_log_p;
                if i >= self.warmup {
                    accepted += 1;
                }
            }

            if i >= self.warmup {
                samples.push(current.clone());
            }
        }

        MultiSamplerResult {
            acceptance_rate: accepted as f64 / self.n_samples as f64,
            samples,
            n_dims,
        }
    }
}

/// Adaptive Metropolis-Hastings that tunes proposal variance during warmup.
///
/// Uses the Robbins-Monro algorithm to target a specific acceptance rate.
/// During warmup, the proposal standard deviation is adjusted on a log scale:
/// `log_std += step_size * (acceptance - target)`.
pub struct AdaptiveMH {
    /// Target acceptance rate (typically 0.234 for high-dim, 0.44 for 1D).
    pub target_acceptance: f64,
    /// Initial proposal standard deviation.
    pub initial_proposal_std: f64,
    /// Number of warmup (adaptation) iterations.
    pub warmup: usize,
    /// Number of sampling iterations (proposal is fixed during sampling).
    pub n_samples: usize,
}

impl AdaptiveMH {
    /// Create a new Adaptive Metropolis-Hastings sampler.
    pub fn new(target_acceptance: f64, warmup: usize, n_samples: usize) -> Self {
        Self {
            target_acceptance,
            initial_proposal_std: 1.0,
            warmup,
            n_samples,
        }
    }

    /// Run the sampler on a 1D log-density function.
    ///
    /// `log_density`: function computing the log of the unnormalized posterior
    /// `initial`: starting point for the chain
    pub fn sample<F>(&self, log_density: F, initial: f64) -> SamplerResult
    where
        F: Fn(f64) -> f64,
    {
        let mut rng = rand::rng();
        let mut log_std = self.initial_proposal_std.ln();
        let mut current = initial;
        let mut current_log_p = log_density(current);
        let mut samples = Vec::with_capacity(self.n_samples);
        let mut accepted_sampling = 0u64;
        let total_iter = self.warmup + self.n_samples;

        for i in 0..total_iter {
            let proposal_std = log_std.exp();
            let proposal = Normal::new(0.0, proposal_std).unwrap();
            let candidate = current + proposal.sample(&mut rng);
            let candidate_log_p = log_density(candidate);

            let log_alpha = candidate_log_p - current_log_p;
            let u: f64 = rng.random();
            let accept = u.ln() < log_alpha;

            if accept {
                current = candidate;
                current_log_p = candidate_log_p;
            }

            // Adapt during warmup using Robbins-Monro
            if i < self.warmup {
                let step_size = 1.0 / (1.0 + i as f64).sqrt();
                let acceptance_indicator = if accept { 1.0 } else { 0.0 };
                log_std += step_size * (acceptance_indicator - self.target_acceptance);
            }

            if i >= self.warmup {
                if accept {
                    accepted_sampling += 1;
                }
                samples.push(current);
            }
        }

        SamplerResult {
            acceptance_rate: accepted_sampling as f64 / self.n_samples as f64,
            samples,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mh_samples_from_standard_normal() {
        // Target: standard normal N(0,1)
        let log_density = |x: f64| -0.5 * x * x;

        let sampler = MetropolisHastings::new(1.0, 1000, 5000);
        let result = sampler.sample(log_density, 0.0);

        assert_eq!(result.samples.len(), 5000);

        // Check that mean is approximately 0
        let mean: f64 = result.samples.iter().sum::<f64>() / result.samples.len() as f64;
        assert!(
            mean.abs() < 0.2,
            "Mean should be near 0, got {mean}"
        );

        // Check acceptance rate is reasonable
        assert!(
            result.acceptance_rate > 0.1 && result.acceptance_rate < 0.9,
            "Acceptance rate {:.2} is outside reasonable range",
            result.acceptance_rate
        );
    }

    #[test]
    fn multidim_mh_2d_standard_normal() {
        // Target: independent 2D standard normal
        let log_density = |x: &[f64]| -0.5 * (x[0] * x[0] + x[1] * x[1]);

        let sampler = MultiDimMH::new(vec![1.0, 1.0], 2000, 5000);
        let result = sampler.sample(log_density, vec![0.0, 0.0]);

        assert_eq!(result.samples.len(), 5000);
        assert_eq!(result.n_dims, 2);

        let mean_x: f64 = result.samples.iter().map(|s| s[0]).sum::<f64>() / 5000.0;
        let mean_y: f64 = result.samples.iter().map(|s| s[1]).sum::<f64>() / 5000.0;
        assert!(mean_x.abs() < 0.2, "Mean X should be near 0, got {mean_x}");
        assert!(mean_y.abs() < 0.2, "Mean Y should be near 0, got {mean_y}");
    }

    #[test]
    fn multidim_mh_correlated_bivariate() {
        // Target: bivariate normal with correlation rho = 0.5
        // Precision matrix for [[1, 0.5],[0.5, 1]] is [[4/3, -2/3],[-2/3, 4/3]]
        let log_density = |x: &[f64]| {
            let prec00 = 4.0 / 3.0;
            let prec01 = -2.0 / 3.0;
            let prec11 = 4.0 / 3.0;
            -0.5 * (prec00 * x[0] * x[0] + 2.0 * prec01 * x[0] * x[1] + prec11 * x[1] * x[1])
        };

        let sampler = MultiDimMH::new(vec![0.8, 0.8], 3000, 8000);
        let result = sampler.sample(log_density, vec![0.0, 0.0]);

        assert_eq!(result.samples.len(), 8000);

        // Check means are near 0
        let mean_x: f64 = result.samples.iter().map(|s| s[0]).sum::<f64>() / 8000.0;
        let mean_y: f64 = result.samples.iter().map(|s| s[1]).sum::<f64>() / 8000.0;
        assert!(mean_x.abs() < 0.3, "Mean X should be near 0, got {mean_x}");
        assert!(mean_y.abs() < 0.3, "Mean Y should be near 0, got {mean_y}");

        // Check positive correlation
        let cov: f64 = result
            .samples
            .iter()
            .map(|s| (s[0] - mean_x) * (s[1] - mean_y))
            .sum::<f64>()
            / 8000.0;
        assert!(cov > 0.0, "Covariance should be positive, got {cov}");
    }

    #[test]
    fn adaptive_mh_standard_normal() {
        let log_density = |x: f64| -0.5 * x * x;

        let sampler = AdaptiveMH::new(0.44, 2000, 5000);
        let result = sampler.sample(log_density, 0.0);

        assert_eq!(result.samples.len(), 5000);

        let mean: f64 = result.samples.iter().sum::<f64>() / 5000.0;
        assert!(mean.abs() < 0.2, "Mean should be near 0, got {mean}");

        // Acceptance rate should be somewhat reasonable after adaptation
        assert!(
            result.acceptance_rate > 0.1 && result.acceptance_rate < 0.9,
            "Acceptance rate {:.2} is outside reasonable range",
            result.acceptance_rate
        );
    }

    #[test]
    fn adaptive_mh_narrow_target() {
        // Target: N(0, 0.01) — very narrow, should adapt proposal down
        let log_density = |x: f64| -0.5 * x * x / 0.01;

        let sampler = AdaptiveMH::new(0.44, 3000, 5000);
        let result = sampler.sample(log_density, 0.0);

        let mean: f64 = result.samples.iter().sum::<f64>() / 5000.0;
        assert!(
            mean.abs() < 0.1,
            "Mean should be near 0, got {mean}"
        );

        let var: f64 =
            result.samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 5000.0;
        // Variance should be approximately 0.01
        assert!(
            var < 0.05,
            "Variance should be small (~0.01), got {var}"
        );
    }
}
