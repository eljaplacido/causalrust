//! KL-divergence based drift detection for routing distributions.
//!
//! Monitors how the distribution of classified domains changes over time
//! compared to a reference (baseline) distribution. Significant drift may
//! indicate data quality issues, model degradation, or genuine shifts in
//! query complexity.

use cynepic_core::CynefinDomain;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Monitors routing distribution drift via KL-divergence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetector {
    /// Reference distribution (baseline probabilities per domain).
    reference: HashMap<CynefinDomain, f64>,
    /// Current window observation counts.
    current_counts: HashMap<CynefinDomain, u64>,
    /// Total observations in current window.
    current_total: u64,
    /// KL-divergence threshold above which drift is flagged.
    threshold: f64,
}

/// Report from a drift check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    /// Computed KL-divergence (bits) from reference to current.
    pub kl_divergence: f64,
    /// Whether drift exceeds the configured threshold.
    pub drift_detected: bool,
    /// The reference (baseline) distribution.
    pub reference_distribution: HashMap<CynefinDomain, f64>,
    /// The current observed distribution.
    pub current_distribution: HashMap<CynefinDomain, f64>,
    /// Total observations in the current window.
    pub total_observations: u64,
}

impl DriftDetector {
    /// Create a new drift detector with the given KL-divergence threshold.
    ///
    /// A threshold of 0.05 is a reasonable starting point; values above
    /// this indicate meaningful distribution shift.
    pub fn new(threshold: f64) -> Self {
        Self {
            reference: HashMap::new(),
            current_counts: HashMap::new(),
            current_total: 0,
            threshold,
        }
    }

    /// Set the reference distribution explicitly (probabilities must sum to ~1.0).
    pub fn set_reference(&mut self, distribution: HashMap<CynefinDomain, f64>) {
        self.reference = distribution;
    }

    /// Snapshot the current observation counts as the new reference distribution.
    pub fn set_reference_from_counts(&mut self) {
        if self.current_total > 0 {
            self.reference = Self::normalize(&self.current_counts, self.current_total);
        }
    }

    /// Record an observed domain classification.
    pub fn record(&mut self, domain: CynefinDomain) {
        *self.current_counts.entry(domain).or_insert(0) += 1;
        self.current_total += 1;
    }

    /// Check for drift between the reference and current distributions.
    pub fn check(&self) -> DriftReport {
        let current_dist = if self.current_total > 0 {
            Self::normalize(&self.current_counts, self.current_total)
        } else {
            HashMap::new()
        };

        let kl = if self.reference.is_empty() || self.current_total == 0 {
            0.0
        } else {
            Self::kl_divergence(&current_dist, &self.reference)
        };

        DriftReport {
            kl_divergence: kl,
            drift_detected: kl > self.threshold,
            reference_distribution: self.reference.clone(),
            current_distribution: current_dist,
            total_observations: self.current_total,
        }
    }

    /// Reset the current window counts (keeps the reference).
    pub fn reset_window(&mut self) {
        self.current_counts.clear();
        self.current_total = 0;
    }

    /// Reset everything including the reference distribution.
    pub fn reset(&mut self) {
        self.reference.clear();
        self.current_counts.clear();
        self.current_total = 0;
    }

    /// Get the current observation count for a domain.
    pub fn count(&self, domain: CynefinDomain) -> u64 {
        self.current_counts.get(&domain).copied().unwrap_or(0)
    }

    /// Total observations in the current window.
    pub fn total(&self) -> u64 {
        self.current_total
    }

    /// Compute KL-divergence: KL(P || Q) = Σ p_i * ln(p_i / q_i).
    ///
    /// Uses additive smoothing (epsilon) to avoid division by zero.
    fn kl_divergence(p: &HashMap<CynefinDomain, f64>, q: &HashMap<CynefinDomain, f64>) -> f64 {
        let epsilon = 1e-10;
        let all_domains = CynefinDomain::all();

        all_domains
            .iter()
            .map(|domain| {
                let p_i = p.get(domain).copied().unwrap_or(0.0).max(epsilon);
                let q_i = q.get(domain).copied().unwrap_or(0.0).max(epsilon);
                p_i * (p_i / q_i).ln()
            })
            .sum()
    }

    /// Normalize counts to a probability distribution.
    fn normalize(counts: &HashMap<CynefinDomain, u64>, total: u64) -> HashMap<CynefinDomain, f64> {
        counts
            .iter()
            .map(|(&domain, &count)| (domain, count as f64 / total as f64))
            .collect()
    }
}

impl Default for DriftDetector {
    fn default() -> Self {
        Self::new(0.05)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_drift_same_distribution() {
        let mut detector = DriftDetector::new(0.05);

        // Build a reference
        for _ in 0..50 {
            detector.record(CynefinDomain::Clear);
        }
        for _ in 0..30 {
            detector.record(CynefinDomain::Complicated);
        }
        for _ in 0..20 {
            detector.record(CynefinDomain::Complex);
        }
        detector.set_reference_from_counts();
        detector.reset_window();

        // Observe same distribution
        for _ in 0..50 {
            detector.record(CynefinDomain::Clear);
        }
        for _ in 0..30 {
            detector.record(CynefinDomain::Complicated);
        }
        for _ in 0..20 {
            detector.record(CynefinDomain::Complex);
        }

        let report = detector.check();
        assert!(!report.drift_detected);
        assert!(report.kl_divergence < 0.01);
    }

    #[test]
    fn detects_significant_drift() {
        let mut detector = DriftDetector::new(0.05);

        // Reference: mostly Clear
        let mut reference = HashMap::new();
        reference.insert(CynefinDomain::Clear, 0.7);
        reference.insert(CynefinDomain::Complicated, 0.2);
        reference.insert(CynefinDomain::Complex, 0.1);
        detector.set_reference(reference);

        // Current: mostly Chaotic (dramatic shift)
        for _ in 0..70 {
            detector.record(CynefinDomain::Chaotic);
        }
        for _ in 0..20 {
            detector.record(CynefinDomain::Clear);
        }
        for _ in 0..10 {
            detector.record(CynefinDomain::Complicated);
        }

        let report = detector.check();
        assert!(report.drift_detected);
        assert!(report.kl_divergence > 0.05);
    }

    #[test]
    fn empty_window_no_drift() {
        let mut detector = DriftDetector::new(0.05);
        let mut reference = HashMap::new();
        reference.insert(CynefinDomain::Clear, 0.5);
        reference.insert(CynefinDomain::Complicated, 0.5);
        detector.set_reference(reference);

        let report = detector.check();
        assert!(!report.drift_detected);
        assert_eq!(report.total_observations, 0);
    }

    #[test]
    fn reset_clears_state() {
        let mut detector = DriftDetector::new(0.05);
        detector.record(CynefinDomain::Clear);
        detector.set_reference_from_counts();

        assert_eq!(detector.total(), 1);
        detector.reset();
        assert_eq!(detector.total(), 0);

        let report = detector.check();
        assert!(!report.drift_detected);
    }
}
