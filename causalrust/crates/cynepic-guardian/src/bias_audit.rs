//! Bias auditing via chi-squared goodness-of-fit tests.
//!
//! Tests whether observed decision distributions deviate significantly from
//! expected distributions, detecting potential bias in routing, policy
//! evaluation, or classification across groups.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of a bias audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasAuditResult {
    /// The chi-squared test statistic.
    pub chi_squared: f64,
    /// Degrees of freedom (number of categories - 1).
    pub degrees_of_freedom: usize,
    /// Whether bias was detected at the configured significance level.
    pub bias_detected: bool,
    /// The critical value used for comparison.
    pub critical_value: f64,
    /// Observed counts per category.
    pub observed_counts: HashMap<String, u64>,
    /// Expected counts per category.
    pub expected_counts: HashMap<String, f64>,
    /// Significance level used.
    pub significance_level: f64,
}

/// Audits decision distributions for statistical bias.
pub struct BiasAuditor {
    /// Significance level for the chi-squared test (e.g., 0.05).
    significance_level: f64,
}

impl BiasAuditor {
    /// Create a new auditor with the given significance level.
    ///
    /// Common values: 0.05 (95% confidence), 0.01 (99% confidence).
    pub fn new(significance_level: f64) -> Self {
        Self { significance_level }
    }

    /// Test whether observed counts follow a uniform distribution.
    ///
    /// Each category is expected to have equal probability.
    pub fn test_uniformity(&self, counts: &HashMap<String, u64>) -> BiasAuditResult {
        if counts.is_empty() {
            return BiasAuditResult {
                chi_squared: 0.0,
                degrees_of_freedom: 0,
                bias_detected: false,
                critical_value: 0.0,
                observed_counts: counts.clone(),
                expected_counts: HashMap::new(),
                significance_level: self.significance_level,
            };
        }

        let total: u64 = counts.values().sum();
        let k = counts.len();
        let expected_per_category = total as f64 / k as f64;

        let expected: HashMap<String, f64> = counts
            .keys()
            .map(|key| (key.clone(), expected_per_category))
            .collect();

        self.test_distribution(counts, &expected)
    }

    /// Test observed counts against an expected distribution.
    ///
    /// Expected values should be absolute counts (not probabilities).
    pub fn test_distribution(
        &self,
        observed: &HashMap<String, u64>,
        expected: &HashMap<String, f64>,
    ) -> BiasAuditResult {
        if observed.is_empty() || expected.is_empty() {
            return BiasAuditResult {
                chi_squared: 0.0,
                degrees_of_freedom: 0,
                bias_detected: false,
                critical_value: 0.0,
                observed_counts: observed.clone(),
                expected_counts: expected.clone(),
                significance_level: self.significance_level,
            };
        }

        let df = observed.len().saturating_sub(1);
        if df == 0 {
            return BiasAuditResult {
                chi_squared: 0.0,
                degrees_of_freedom: 0,
                bias_detected: false,
                critical_value: 0.0,
                observed_counts: observed.clone(),
                expected_counts: expected.clone(),
                significance_level: self.significance_level,
            };
        }

        // Compute chi-squared statistic: Σ (O_i - E_i)² / E_i
        let chi_sq: f64 = observed
            .iter()
            .map(|(key, &obs)| {
                let exp = expected.get(key).copied().unwrap_or(1.0).max(1e-10);
                let diff = obs as f64 - exp;
                diff * diff / exp
            })
            .sum();

        let critical = Self::critical_value(df, self.significance_level);

        BiasAuditResult {
            chi_squared: chi_sq,
            degrees_of_freedom: df,
            bias_detected: chi_sq > critical,
            critical_value: critical,
            observed_counts: observed.clone(),
            expected_counts: expected.clone(),
            significance_level: self.significance_level,
        }
    }

    /// Audit approval rates across groups for disparate treatment.
    ///
    /// Each entry is `(group_name, was_approved)`. Tests whether approval
    /// rates are uniform across groups.
    pub fn audit_approval_rates(&self, decisions: &[(String, bool)]) -> BiasAuditResult {
        if decisions.is_empty() {
            return BiasAuditResult {
                chi_squared: 0.0,
                degrees_of_freedom: 0,
                bias_detected: false,
                critical_value: 0.0,
                observed_counts: HashMap::new(),
                expected_counts: HashMap::new(),
                significance_level: self.significance_level,
            };
        }

        // Count approvals per group
        let mut group_totals: HashMap<String, u64> = HashMap::new();
        let mut group_approvals: HashMap<String, u64> = HashMap::new();

        for (group, approved) in decisions {
            *group_totals.entry(group.clone()).or_insert(0) += 1;
            if *approved {
                *group_approvals.entry(group.clone()).or_insert(0) += 1;
            }
        }

        // Overall approval rate
        let total_approved: u64 = group_approvals.values().sum();
        let total: u64 = group_totals.values().sum();
        let overall_rate = total_approved as f64 / total as f64;

        // Expected approvals per group (under null hypothesis of equal rates)
        let expected: HashMap<String, f64> = group_totals
            .iter()
            .map(|(group, &count)| (group.clone(), count as f64 * overall_rate))
            .collect();

        // Observed approvals
        let observed: HashMap<String, u64> = group_totals
            .keys()
            .map(|group| {
                (
                    group.clone(),
                    group_approvals.get(group).copied().unwrap_or(0),
                )
            })
            .collect();

        self.test_distribution(&observed, &expected)
    }

    /// Chi-squared critical value using the Wilson-Hilferty approximation.
    ///
    /// Approximates the inverse chi-squared CDF for the given degrees of
    /// freedom and significance level.
    fn critical_value(df: usize, alpha: f64) -> f64 {
        if df == 0 {
            return 0.0;
        }

        // Standard normal quantile for common alpha levels
        let z = Self::normal_quantile(1.0 - alpha);

        // Wilson-Hilferty approximation: χ²_α,k ≈ k × (1 - 2/(9k) + z × √(2/(9k)))³
        let k = df as f64;
        let term = 1.0 - 2.0 / (9.0 * k) + z * (2.0 / (9.0 * k)).sqrt();
        (k * term * term * term).max(0.0)
    }

    /// Approximate standard normal quantile using Beasley-Springer-Moro algorithm.
    fn normal_quantile(p: f64) -> f64 {
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }

        // Rational approximation for central region
        let t = if p < 0.5 {
            (-2.0 * p.ln()).sqrt()
        } else {
            (-2.0 * (1.0 - p).ln()).sqrt()
        };

        // Abramowitz and Stegun approximation 26.2.23
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

        if p < 0.5 { -result } else { result }
    }
}

impl Default for BiasAuditor {
    fn default() -> Self {
        Self::new(0.05)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_distribution_no_bias() {
        let auditor = BiasAuditor::new(0.05);

        let mut counts = HashMap::new();
        counts.insert("group_a".into(), 100u64);
        counts.insert("group_b".into(), 102);
        counts.insert("group_c".into(), 98);

        let result = auditor.test_uniformity(&counts);
        assert!(!result.bias_detected);
        assert!(result.chi_squared < result.critical_value);
    }

    #[test]
    fn skewed_distribution_detects_bias() {
        let auditor = BiasAuditor::new(0.05);

        let mut counts = HashMap::new();
        counts.insert("group_a".into(), 200u64);
        counts.insert("group_b".into(), 50);
        counts.insert("group_c".into(), 50);

        let result = auditor.test_uniformity(&counts);
        assert!(result.bias_detected);
        assert!(result.chi_squared > result.critical_value);
    }

    #[test]
    fn approval_rate_bias_detection() {
        let auditor = BiasAuditor::new(0.05);

        // Group A: 90% approval, Group B: 30% approval → clear bias
        let mut decisions = Vec::new();
        for _ in 0..90 {
            decisions.push(("group_a".into(), true));
        }
        for _ in 0..10 {
            decisions.push(("group_a".into(), false));
        }
        for _ in 0..30 {
            decisions.push(("group_b".into(), true));
        }
        for _ in 0..70 {
            decisions.push(("group_b".into(), false));
        }

        let result = auditor.audit_approval_rates(&decisions);
        assert!(result.bias_detected);
    }

    #[test]
    fn equal_approval_rates_no_bias() {
        let auditor = BiasAuditor::new(0.05);

        let mut decisions = Vec::new();
        for _ in 0..50 {
            decisions.push(("group_a".into(), true));
        }
        for _ in 0..50 {
            decisions.push(("group_a".into(), false));
        }
        for _ in 0..48 {
            decisions.push(("group_b".into(), true));
        }
        for _ in 0..52 {
            decisions.push(("group_b".into(), false));
        }

        let result = auditor.audit_approval_rates(&decisions);
        assert!(!result.bias_detected);
    }

    #[test]
    fn empty_input_no_panic() {
        let auditor = BiasAuditor::new(0.05);

        let result = auditor.test_uniformity(&HashMap::new());
        assert!(!result.bias_detected);

        let result = auditor.audit_approval_rates(&[]);
        assert!(!result.bias_detected);
    }

    #[test]
    fn critical_value_reasonable() {
        // For df=1, alpha=0.05, expected ~3.84
        let cv = BiasAuditor::critical_value(1, 0.05);
        assert!((cv - 3.84).abs() < 0.5);

        // For df=4, alpha=0.05, expected ~9.49
        let cv = BiasAuditor::critical_value(4, 0.05);
        assert!((cv - 9.49).abs() < 0.5);
    }
}
