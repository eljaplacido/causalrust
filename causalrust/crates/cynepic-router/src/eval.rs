use cynepic_core::CynefinDomain;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Tracks classification accuracy for evaluation and monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierMetrics {
    /// Confusion matrix: predicted -> actual -> count.
    confusion: HashMap<String, HashMap<String, u64>>,
    /// Total predictions.
    total: u64,
    /// Correct predictions.
    correct: u64,
}

impl ClassifierMetrics {
    /// Create a new empty metrics tracker.
    pub fn new() -> Self {
        Self {
            confusion: HashMap::new(),
            total: 0,
            correct: 0,
        }
    }

    /// Record a prediction vs ground truth.
    pub fn record(&mut self, predicted: CynefinDomain, actual: CynefinDomain) {
        let predicted_key = predicted.to_string();
        let actual_key = actual.to_string();

        *self
            .confusion
            .entry(predicted_key)
            .or_default()
            .entry(actual_key)
            .or_insert(0) += 1;

        self.total += 1;
        if predicted == actual {
            self.correct += 1;
        }
    }

    /// Overall accuracy.
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }

    /// Per-domain precision: TP / (TP + FP).
    ///
    /// TP = times we predicted this domain and it was correct.
    /// FP = times we predicted this domain but it was actually something else.
    pub fn precision(&self, domain: CynefinDomain) -> f64 {
        let domain_key = domain.to_string();

        let predicted_row = match self.confusion.get(&domain_key) {
            Some(row) => row,
            None => return 0.0,
        };

        let tp = *predicted_row.get(&domain_key).unwrap_or(&0);
        let total_predicted: u64 = predicted_row.values().sum();

        if total_predicted == 0 {
            0.0
        } else {
            tp as f64 / total_predicted as f64
        }
    }

    /// Per-domain recall: TP / (TP + FN).
    ///
    /// TP = times we predicted this domain and it was correct.
    /// FN = times the actual domain was this but we predicted something else.
    pub fn recall(&self, domain: CynefinDomain) -> f64 {
        let domain_key = domain.to_string();

        let tp = self
            .confusion
            .get(&domain_key)
            .and_then(|row| row.get(&domain_key))
            .copied()
            .unwrap_or(0);

        // Total times this domain was the actual answer (across all predictions).
        let total_actual: u64 = self
            .confusion
            .values()
            .map(|row| *row.get(&domain_key).unwrap_or(&0))
            .sum();

        if total_actual == 0 {
            0.0
        } else {
            tp as f64 / total_actual as f64
        }
    }

    /// Per-domain F1 score: harmonic mean of precision and recall.
    pub fn f1(&self, domain: CynefinDomain) -> f64 {
        let p = self.precision(domain);
        let r = self.recall(domain);

        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Total number of predictions.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Misrouting cost analysis: weighted by CostTier.
    ///
    /// Returns the total "wasted cost" from misclassifications,
    /// weighted by the cost tier of the incorrect route.
    /// Uses a domain-to-tier mapping where more complex domains cost more.
    pub fn misrouting_cost(&self, cost_map: &super::budget::CostMap) -> f64 {
        use crate::config::CostTier;

        let domain_tier = |domain: &str| -> CostTier {
            match domain {
                "Clear" => CostTier::Free,
                "Complicated" => CostTier::Low,
                "Complex" => CostTier::Medium,
                "Chaotic" => CostTier::High,
                _ => CostTier::Free,
            }
        };

        let mut total_cost = 0.0;

        for (predicted, actuals) in &self.confusion {
            let tier = domain_tier(predicted);
            let cost = cost_map.cost_for(&tier);

            for (actual, count) in actuals {
                if predicted != actual {
                    total_cost += cost * (*count as f64);
                }
            }
        }

        total_cost
    }

    /// Get the confusion matrix as a nested map.
    pub fn confusion_matrix(&self) -> &HashMap<String, HashMap<String, u64>> {
        &self.confusion
    }
}

impl Default for ClassifierMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_accuracy() {
        let mut metrics = ClassifierMetrics::new();

        metrics.record(CynefinDomain::Clear, CynefinDomain::Clear);
        metrics.record(CynefinDomain::Complicated, CynefinDomain::Complicated);
        metrics.record(CynefinDomain::Complex, CynefinDomain::Complex);
        metrics.record(CynefinDomain::Chaotic, CynefinDomain::Chaotic);

        assert!((metrics.accuracy() - 1.0).abs() < 1e-10);
        assert_eq!(metrics.total(), 4);
        assert!((metrics.precision(CynefinDomain::Clear) - 1.0).abs() < 1e-10);
        assert!((metrics.recall(CynefinDomain::Clear) - 1.0).abs() < 1e-10);
        assert!((metrics.f1(CynefinDomain::Clear) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mixed_predictions_precision_recall() {
        let mut metrics = ClassifierMetrics::new();

        // Predicted Complicated, actual Complicated (TP for Complicated)
        metrics.record(CynefinDomain::Complicated, CynefinDomain::Complicated);
        metrics.record(CynefinDomain::Complicated, CynefinDomain::Complicated);
        // Predicted Complicated, actual Clear (FP for Complicated, FN for Clear)
        metrics.record(CynefinDomain::Complicated, CynefinDomain::Clear);
        // Predicted Clear, actual Clear (TP for Clear)
        metrics.record(CynefinDomain::Clear, CynefinDomain::Clear);
        // Predicted Clear, actual Complicated (FP for Clear, FN for Complicated)
        metrics.record(CynefinDomain::Clear, CynefinDomain::Complicated);

        assert_eq!(metrics.total(), 5);
        // Accuracy: 3 correct out of 5
        assert!((metrics.accuracy() - 0.6).abs() < 1e-10);

        // Complicated precision: 2 TP / (2 TP + 1 FP) = 2/3
        assert!((metrics.precision(CynefinDomain::Complicated) - 2.0 / 3.0).abs() < 1e-10);
        // Complicated recall: 2 TP / (2 TP + 1 FN) = 2/3
        assert!((metrics.recall(CynefinDomain::Complicated) - 2.0 / 3.0).abs() < 1e-10);

        // Clear precision: 1 TP / (1 TP + 1 FP) = 1/2
        assert!((metrics.precision(CynefinDomain::Clear) - 0.5).abs() < 1e-10);
        // Clear recall: 1 TP / (1 TP + 1 FN) = 1/2
        assert!((metrics.recall(CynefinDomain::Clear) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn misrouting_cost_calculation() {
        use crate::budget::CostMap;

        let mut metrics = ClassifierMetrics::new();

        // Correct predictions (no cost)
        metrics.record(CynefinDomain::Clear, CynefinDomain::Clear);
        // Misclassification: predicted Chaotic (High tier, cost 0.20) but actual was Clear
        metrics.record(CynefinDomain::Chaotic, CynefinDomain::Clear);
        metrics.record(CynefinDomain::Chaotic, CynefinDomain::Clear);
        // Misclassification: predicted Complicated (Low tier, cost 0.01) but actual was Complex
        metrics.record(CynefinDomain::Complicated, CynefinDomain::Complex);

        let cost_map = CostMap::default();
        let cost = metrics.misrouting_cost(&cost_map);

        // 2 * 0.20 (Chaotic misroutes) + 1 * 0.01 (Complicated misroute) = 0.41
        assert!((cost - 0.41).abs() < 1e-10);
    }
}
