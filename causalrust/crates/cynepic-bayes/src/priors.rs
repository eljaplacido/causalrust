use serde::{Deserialize, Serialize};

/// Beta-Binomial conjugate prior for binary outcomes.
///
/// Models the probability of success (e.g., conversion rate, CTR).
/// Prior: Beta(α, β), Likelihood: Binomial, Posterior: Beta(α + s, β + f)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaBinomial {
    pub alpha: f64,
    pub beta: f64,
}

impl BetaBinomial {
    /// Create a new Beta-Binomial model with the given prior parameters.
    pub fn new(alpha: f64, beta: f64) -> Self {
        assert!(alpha > 0.0, "alpha must be positive");
        assert!(beta > 0.0, "beta must be positive");
        Self { alpha, beta }
    }

    /// Uniform (non-informative) prior: Beta(1, 1).
    pub fn uniform() -> Self {
        Self::new(1.0, 1.0)
    }

    /// Update the posterior with observed successes and failures.
    pub fn update(&mut self, successes: u64, failures: u64) {
        self.alpha += successes as f64;
        self.beta += failures as f64;
    }

    /// Posterior mean: α / (α + β).
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Posterior variance.
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        (self.alpha * self.beta) / (ab * ab * (ab + 1.0))
    }

    /// Posterior mode (MAP estimate).
    pub fn mode(&self) -> f64 {
        if self.alpha > 1.0 && self.beta > 1.0 {
            (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        } else {
            self.mean() // Fallback for edge cases
        }
    }

    /// 95% credible interval (approximate via normal approximation).
    pub fn credible_interval_95(&self) -> (f64, f64) {
        let mean = self.mean();
        let std = self.variance().sqrt();
        let lower = (mean - 1.96 * std).max(0.0);
        let upper = (mean + 1.96 * std).min(1.0);
        (lower, upper)
    }
}

/// Normal-Normal conjugate prior for continuous outcomes.
///
/// Models the mean of a normally distributed variable with known variance.
/// Prior: N(μ₀, σ₀²), Likelihood: N(μ, σ²), Posterior: N(μ₁, σ₁²)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalNormal {
    /// Prior mean.
    pub mu: f64,
    /// Prior precision (1/σ²).
    pub precision: f64,
    /// Known observation precision (1/σ_obs²).
    pub obs_precision: f64,
}

impl NormalNormal {
    /// Create a new Normal-Normal model.
    pub fn new(mu: f64, prior_variance: f64, obs_variance: f64) -> Self {
        assert!(prior_variance > 0.0, "prior variance must be positive");
        assert!(obs_variance > 0.0, "observation variance must be positive");
        Self {
            mu,
            precision: 1.0 / prior_variance,
            obs_precision: 1.0 / obs_variance,
        }
    }

    /// Update the posterior with a batch of observations.
    pub fn update(&mut self, observations: &[f64]) {
        let n = observations.len() as f64;
        let obs_mean: f64 = observations.iter().sum::<f64>() / n;
        let new_precision = self.precision + n * self.obs_precision;
        let new_mu = (self.precision * self.mu + n * self.obs_precision * obs_mean) / new_precision;
        self.mu = new_mu;
        self.precision = new_precision;
    }

    /// Posterior mean.
    pub fn mean(&self) -> f64 {
        self.mu
    }

    /// Posterior variance.
    pub fn variance(&self) -> f64 {
        1.0 / self.precision
    }

    /// 95% credible interval.
    pub fn credible_interval_95(&self) -> (f64, f64) {
        let std = self.variance().sqrt();
        (self.mu - 1.96 * std, self.mu + 1.96 * std)
    }
}

/// Dirichlet-Multinomial conjugate prior for categorical data.
///
/// Prior: Dirichlet(alpha_1, alpha_2, ..., alpha_k)
/// Likelihood: Multinomial
/// Posterior: Dirichlet(alpha_1 + n_1, alpha_2 + n_2, ..., alpha_k + n_k)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirichletMultinomial {
    /// Concentration parameters (one per category).
    pub alphas: Vec<f64>,
}

impl DirichletMultinomial {
    /// Create a new Dirichlet-Multinomial model with the given concentration parameters.
    pub fn new(alphas: Vec<f64>) -> Self {
        assert!(!alphas.is_empty(), "alphas must not be empty");
        assert!(
            alphas.iter().all(|&a| a > 0.0),
            "all alphas must be positive"
        );
        Self { alphas }
    }

    /// Symmetric uniform prior: Dirichlet(1, 1, ..., 1) with k categories.
    pub fn uniform(k: usize) -> Self {
        assert!(k > 0, "k must be at least 1");
        Self {
            alphas: vec![1.0; k],
        }
    }

    /// Update the posterior with observed category counts.
    pub fn update(&mut self, counts: &[usize]) {
        assert_eq!(
            counts.len(),
            self.alphas.len(),
            "counts length must match number of categories"
        );
        for (alpha, &count) in self.alphas.iter_mut().zip(counts.iter()) {
            *alpha += count as f64;
        }
    }

    /// Posterior mean: normalized concentration parameters.
    pub fn mean(&self) -> Vec<f64> {
        let total: f64 = self.alphas.iter().sum();
        self.alphas.iter().map(|a| a / total).collect()
    }

    /// Posterior mode (MAP estimate).
    ///
    /// Returns (alpha_i - 1) / (sum - k) when all alpha_i > 1.
    /// Falls back to mean when any alpha_i <= 1.
    pub fn mode(&self) -> Vec<f64> {
        let k = self.alphas.len() as f64;
        let total: f64 = self.alphas.iter().sum();
        if self.alphas.iter().all(|&a| a > 1.0) {
            let denom = total - k;
            self.alphas.iter().map(|a| (a - 1.0) / denom).collect()
        } else {
            self.mean()
        }
    }

    /// Number of categories.
    pub fn k(&self) -> usize {
        self.alphas.len()
    }

    /// Sum of all concentration parameters.
    pub fn total_count(&self) -> f64 {
        self.alphas.iter().sum()
    }

    /// Marginal mean for a single category.
    pub fn marginal_mean(&self, i: usize) -> f64 {
        let total: f64 = self.alphas.iter().sum();
        self.alphas[i] / total
    }

    /// Marginal variance for a single category.
    pub fn marginal_variance(&self, i: usize) -> f64 {
        let total: f64 = self.alphas.iter().sum();
        let ai = self.alphas[i];
        (ai * (total - ai)) / (total * total * (total + 1.0))
    }
}

/// Gamma-Poisson conjugate prior for count data.
///
/// Models the rate parameter of a Poisson process.
/// Prior: Gamma(α, β), Likelihood: Poisson(λ), Posterior: Gamma(α + Σx, β + n)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GammaPoisson {
    /// Shape parameter.
    pub alpha: f64,
    /// Rate parameter.
    pub beta: f64,
}

impl GammaPoisson {
    /// Create a new Gamma-Poisson model.
    pub fn new(alpha: f64, beta: f64) -> Self {
        assert!(alpha > 0.0, "alpha must be positive");
        assert!(beta > 0.0, "beta must be positive");
        Self { alpha, beta }
    }

    /// Update with observed counts.
    pub fn update(&mut self, counts: &[u64]) {
        let sum: u64 = counts.iter().sum();
        let n = counts.len() as f64;
        self.alpha += sum as f64;
        self.beta += n;
    }

    /// Posterior mean: α / β.
    pub fn mean(&self) -> f64 {
        self.alpha / self.beta
    }

    /// Posterior variance: α / β².
    pub fn variance(&self) -> f64 {
        self.alpha / (self.beta * self.beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_binomial_update() {
        let mut model = BetaBinomial::uniform();
        model.update(7, 3);
        // Posterior: Beta(8, 4), mean = 8/12 ≈ 0.667
        assert!((model.mean() - 0.6667).abs() < 0.01);
    }

    #[test]
    fn beta_binomial_credible_interval() {
        let mut model = BetaBinomial::uniform();
        model.update(50, 50);
        let (lower, upper) = model.credible_interval_95();
        assert!(lower < 0.5);
        assert!(upper > 0.5);
    }

    #[test]
    fn normal_normal_update() {
        let mut model = NormalNormal::new(0.0, 10.0, 1.0);
        model.update(&[5.0, 5.0, 5.0, 5.0, 5.0]);
        // Posterior should be pulled toward 5.0
        assert!((model.mean() - 5.0).abs() < 0.5);
    }

    #[test]
    fn gamma_poisson_update() {
        let mut model = GammaPoisson::new(1.0, 1.0);
        model.update(&[3, 4, 5, 6, 7]);
        // Posterior: Gamma(1+25, 1+5) = Gamma(26, 6), mean ≈ 4.33
        assert!((model.mean() - 4.33).abs() < 0.1);
    }

    #[test]
    fn dirichlet_multinomial_uniform_mean() {
        let model = DirichletMultinomial::uniform(3);
        let mean = model.mean();
        assert_eq!(mean.len(), 3);
        for m in &mean {
            assert!((m - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn dirichlet_multinomial_update() {
        let mut model = DirichletMultinomial::uniform(3);
        model.update(&[10, 5, 5]);
        // Posterior: Dirichlet(11, 6, 6), mean = [11/23, 6/23, 6/23]
        let mean = model.mean();
        assert!((mean[0] - 11.0 / 23.0).abs() < 1e-10);
        assert!((mean[1] - 6.0 / 23.0).abs() < 1e-10);
        assert_eq!(model.k(), 3);
        assert!((model.total_count() - 23.0).abs() < 1e-10);
    }

    #[test]
    fn dirichlet_multinomial_mode_and_variance() {
        let model = DirichletMultinomial::new(vec![3.0, 2.0, 5.0]);
        let mode = model.mode();
        // (3-1)/(10-3)=2/7, (2-1)/(10-3)=1/7, (5-1)/(10-3)=4/7
        assert!((mode[0] - 2.0 / 7.0).abs() < 1e-10);
        assert!((mode[2] - 4.0 / 7.0).abs() < 1e-10);

        // Marginal variance for category 0: 3*(10-3)/(10*10*11) = 21/1100
        let var0 = model.marginal_variance(0);
        assert!((var0 - 21.0 / 1100.0).abs() < 1e-10);
    }
}
