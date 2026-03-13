use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::config::CostTier;

/// Tracks routing costs over time for budget enforcement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetTracker {
    /// Total budget limit (in arbitrary cost units).
    pub total_budget: f64,
    /// Spent so far.
    pub total_spent: f64,
    /// Per-tier spending.
    pub tier_spending: HashMap<String, f64>,
    /// Number of routes per tier.
    pub tier_counts: HashMap<String, u64>,
    /// Total number of routing decisions.
    pub total_routes: u64,
}

/// Cost mapping for each CostTier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostMap {
    /// Cost for the Free tier.
    pub free: f64,
    /// Cost for the Low tier.
    pub low: f64,
    /// Cost for the Medium tier.
    pub medium: f64,
    /// Cost for the High tier.
    pub high: f64,
}

impl Default for CostMap {
    fn default() -> Self {
        Self {
            free: 0.0,
            low: 0.01,
            medium: 0.05,
            high: 0.20,
        }
    }
}

impl CostMap {
    /// Get the cost for a given tier.
    pub fn cost_for(&self, tier: &CostTier) -> f64 {
        match tier {
            CostTier::Free => self.free,
            CostTier::Low => self.low,
            CostTier::Medium => self.medium,
            CostTier::High => self.high,
        }
    }
}

/// Result of a budget check or recording.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BudgetDecision {
    /// Within budget, proceed normally.
    WithinBudget {
        /// Remaining budget after this decision.
        remaining: f64,
    },
    /// Over budget — suggest downgrading to a cheaper tier.
    OverBudget {
        /// Amount over budget.
        overage: f64,
        /// Suggested cheaper tier to use instead.
        suggested_tier: CostTier,
    },
}

impl BudgetTracker {
    /// Create a new budget tracker with the given total budget.
    pub fn new(total_budget: f64) -> Self {
        Self {
            total_budget,
            total_spent: 0.0,
            tier_spending: HashMap::new(),
            tier_counts: HashMap::new(),
            total_routes: 0,
        }
    }

    /// Record a routing cost. Returns budget decision.
    pub fn record(&mut self, tier: &CostTier, cost_map: &CostMap) -> BudgetDecision {
        let cost = cost_map.cost_for(tier);
        let tier_name = format!("{:?}", tier);

        self.total_spent += cost;
        self.total_routes += 1;
        *self.tier_spending.entry(tier_name.clone()).or_insert(0.0) += cost;
        *self.tier_counts.entry(tier_name).or_insert(0) += 1;

        if self.total_spent <= self.total_budget {
            BudgetDecision::WithinBudget {
                remaining: self.total_budget - self.total_spent,
            }
        } else {
            BudgetDecision::OverBudget {
                overage: self.total_spent - self.total_budget,
                suggested_tier: self.suggest_downgrade(),
            }
        }
    }

    /// Check if a route would exceed the budget without recording it.
    pub fn check(&self, tier: &CostTier, cost_map: &CostMap) -> BudgetDecision {
        let cost = cost_map.cost_for(tier);
        let projected_spent = self.total_spent + cost;

        if projected_spent <= self.total_budget {
            BudgetDecision::WithinBudget {
                remaining: self.total_budget - projected_spent,
            }
        } else {
            BudgetDecision::OverBudget {
                overage: projected_spent - self.total_budget,
                suggested_tier: self.suggest_downgrade(),
            }
        }
    }

    /// Get remaining budget.
    pub fn remaining(&self) -> f64 {
        self.total_budget - self.total_spent
    }

    /// Get utilization as a percentage (0.0 to 1.0+).
    pub fn utilization(&self) -> f64 {
        if self.total_budget == 0.0 {
            0.0
        } else {
            self.total_spent / self.total_budget
        }
    }

    /// Get average cost per route.
    pub fn average_cost(&self) -> f64 {
        if self.total_routes == 0 {
            0.0
        } else {
            self.total_spent / self.total_routes as f64
        }
    }

    /// Reset the tracker (e.g., for a new billing period).
    pub fn reset(&mut self) {
        self.total_spent = 0.0;
        self.tier_spending.clear();
        self.tier_counts.clear();
        self.total_routes = 0;
    }

    /// Find the cheapest tier that would keep spending under budget.
    /// Defaults to Free if all tiers would exceed the budget.
    fn suggest_downgrade(&self) -> CostTier {
        let default_cost_map = CostMap::default();
        let tiers = [CostTier::Free, CostTier::Low, CostTier::Medium, CostTier::High];

        for tier in &tiers {
            let cost = default_cost_map.cost_for(tier);
            if self.total_spent + cost <= self.total_budget {
                return *tier;
            }
        }

        CostTier::Free
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn within_budget() {
        let mut tracker = BudgetTracker::new(1.0);
        let cost_map = CostMap::default();

        let decision = tracker.record(&CostTier::Low, &cost_map);
        match decision {
            BudgetDecision::WithinBudget { remaining } => {
                assert!((remaining - 0.99).abs() < 1e-10);
            }
            _ => panic!("Expected WithinBudget"),
        }
        assert_eq!(tracker.total_routes, 1);
    }

    #[test]
    fn over_budget_detection() {
        let mut tracker = BudgetTracker::new(0.10);
        let cost_map = CostMap::default();

        // First High route costs 0.20, which exceeds the 0.10 budget.
        let decision = tracker.record(&CostTier::High, &cost_map);
        match decision {
            BudgetDecision::OverBudget { overage, suggested_tier } => {
                assert!((overage - 0.10).abs() < 1e-10);
                // Should suggest Free since we're already over budget
                assert_eq!(suggested_tier, CostTier::Free);
            }
            _ => panic!("Expected OverBudget"),
        }
    }

    #[test]
    fn utilization_tracking() {
        let mut tracker = BudgetTracker::new(1.0);
        let cost_map = CostMap::default();

        tracker.record(&CostTier::Medium, &cost_map); // 0.05
        tracker.record(&CostTier::Medium, &cost_map); // 0.05
        // Total spent: 0.10 out of 1.0 = 10%

        assert!((tracker.utilization() - 0.10).abs() < 1e-10);
        assert!((tracker.average_cost() - 0.05).abs() < 1e-10);
        assert_eq!(tracker.total_routes, 2);
    }

    #[test]
    fn reset_clears_state() {
        let mut tracker = BudgetTracker::new(1.0);
        let cost_map = CostMap::default();

        tracker.record(&CostTier::High, &cost_map);
        tracker.record(&CostTier::Medium, &cost_map);
        assert!(tracker.total_spent > 0.0);
        assert!(tracker.total_routes > 0);

        tracker.reset();

        assert_eq!(tracker.total_spent, 0.0);
        assert_eq!(tracker.total_routes, 0);
        assert!(tracker.tier_spending.is_empty());
        assert!(tracker.tier_counts.is_empty());
        assert_eq!(tracker.remaining(), 1.0);
    }
}
