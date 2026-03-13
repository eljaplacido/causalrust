use crate::budget::{BudgetDecision, BudgetTracker, CostMap};
use crate::classifier::{ClassificationResult, ClassifierError, QueryClassifier};
use crate::config::{RouterConfig, RouteTarget};
use std::sync::Arc;

/// The Cynefin router: classifies queries and returns routing decisions.
pub struct CynefinRouter {
    classifier: Arc<dyn QueryClassifier>,
    config: RouterConfig,
    /// Optional budget tracker for cost-aware routing.
    budget: Option<BudgetTracker>,
    /// Optional cost map for budget calculations.
    cost_map: Option<CostMap>,
}

/// Extended routing decision with cost tracking.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// The classification result.
    pub classification: ClassificationResult,
    /// The target to route to (if a route exists for the domain).
    pub target: Option<RouteTarget>,
    /// Whether the classification was confident enough.
    pub confident: bool,
    /// Budget status after this routing decision (if budget tracking is enabled).
    pub budget_status: Option<BudgetDecision>,
}

impl CynefinRouter {
    /// Create a new router with the given classifier and config.
    pub fn new(classifier: Arc<dyn QueryClassifier>, config: RouterConfig) -> Self {
        Self {
            classifier,
            config,
            budget: None,
            cost_map: None,
        }
    }

    /// Create a new router with budget tracking.
    pub fn with_budget(
        classifier: Arc<dyn QueryClassifier>,
        config: RouterConfig,
        budget: BudgetTracker,
        cost_map: CostMap,
    ) -> Self {
        Self {
            classifier,
            config,
            budget: Some(budget),
            cost_map: Some(cost_map),
        }
    }

    /// Classify a query and produce a routing decision.
    pub async fn route(&self, query: &str) -> Result<RoutingDecision, ClassifierError> {
        let classification = self.classifier.classify(query).await?;

        let confident = classification.confidence >= self.config.confidence_threshold;
        let effective_domain = if confident {
            classification.domain
        } else {
            self.config.fallback_domain
        };

        let target = self.config.routes.get(&effective_domain).cloned();

        tracing::info!(
            query = query,
            domain = %effective_domain,
            confidence = classification.confidence,
            confident = confident,
            has_route = target.is_some(),
            "Routing decision"
        );

        Ok(RoutingDecision {
            classification,
            target,
            confident,
            budget_status: None,
        })
    }

    /// Route with budget enforcement.
    ///
    /// If over budget, may downgrade to a cheaper route.
    pub async fn route_with_budget(
        &mut self,
        query: &str,
    ) -> Result<RoutingDecision, ClassifierError> {
        let classification = self.classifier.classify(query).await?;

        let confident = classification.confidence >= self.config.confidence_threshold;
        let effective_domain = if confident {
            classification.domain
        } else {
            self.config.fallback_domain
        };

        let target = self.config.routes.get(&effective_domain).cloned();

        // Determine cost tier from the target, defaulting to Free.
        let cost_tier = target
            .as_ref()
            .map(|t| t.cost_tier)
            .unwrap_or_default();

        let budget_status = match (&mut self.budget, &self.cost_map) {
            (Some(budget), Some(cost_map)) => {
                let decision = budget.record(&cost_tier, cost_map);
                Some(decision)
            }
            _ => None,
        };

        // If over budget, try to downgrade to a cheaper route.
        let (final_target, final_budget_status) = match &budget_status {
            Some(BudgetDecision::OverBudget { suggested_tier, .. }) => {
                // Look for a route with the suggested (cheaper) tier.
                let downgraded = self
                    .config
                    .routes
                    .values()
                    .find(|rt| rt.cost_tier == *suggested_tier)
                    .cloned();
                (downgraded.or(target), budget_status)
            }
            _ => (target, budget_status),
        };

        tracing::info!(
            query = query,
            domain = %effective_domain,
            confidence = classification.confidence,
            confident = confident,
            has_route = final_target.is_some(),
            "Routing decision (budget-aware)"
        );

        Ok(RoutingDecision {
            classification,
            target: final_target,
            confident,
            budget_status: final_budget_status,
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Get current budget status.
    pub fn budget(&self) -> Option<&BudgetTracker> {
        self.budget.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::{BudgetTracker, CostMap};
    use crate::classifier::KeywordClassifier;
    use crate::config::CostTier;
    use cynepic_core::CynefinDomain;
    use std::collections::HashMap;

    #[tokio::test]
    async fn routes_causal_query() {
        let classifier = Arc::new(KeywordClassifier::default_patterns());
        let mut routes = HashMap::new();
        routes.insert(
            CynefinDomain::Complicated,
            RouteTarget {
                url: "http://causal-engine:8080".into(),
                timeout_ms: 5000,
                cost_tier: Default::default(),
            },
        );

        let config = RouterConfig {
            routes,
            confidence_threshold: 0.1,
            ..Default::default()
        };

        let router = CynefinRouter::new(classifier, config);
        let decision = router.route("Why did the effect change?").await.unwrap();
        assert!(decision.target.is_some());
    }

    #[tokio::test]
    async fn route_with_budget_tracking() {
        let classifier = Arc::new(KeywordClassifier::default_patterns());
        let mut routes = HashMap::new();
        routes.insert(
            CynefinDomain::Complicated,
            RouteTarget {
                url: "http://causal-engine:8080".into(),
                timeout_ms: 5000,
                cost_tier: CostTier::Medium,
            },
        );

        let config = RouterConfig {
            routes,
            confidence_threshold: 0.1,
            ..Default::default()
        };

        let budget = BudgetTracker::new(10.0);
        let cost_map = CostMap::default();
        let mut router = CynefinRouter::with_budget(classifier, config, budget, cost_map);

        let decision = router
            .route_with_budget("Why did the effect change?")
            .await
            .unwrap();

        assert!(decision.target.is_some());
        assert!(decision.budget_status.is_some());

        match decision.budget_status.unwrap() {
            BudgetDecision::WithinBudget { remaining } => {
                assert!(remaining > 0.0);
            }
            _ => panic!("Expected WithinBudget"),
        }

        // Budget tracker should reflect the route.
        let tracker = router.budget().unwrap();
        assert_eq!(tracker.total_routes, 1);
        assert!(tracker.total_spent > 0.0);
    }

    #[tokio::test]
    async fn budget_enforced_downgrade() {
        let classifier = Arc::new(KeywordClassifier::default_patterns());
        let mut routes = HashMap::new();
        routes.insert(
            CynefinDomain::Complicated,
            RouteTarget {
                url: "http://expensive-engine:8080".into(),
                timeout_ms: 5000,
                cost_tier: CostTier::High,
            },
        );
        routes.insert(
            CynefinDomain::Clear,
            RouteTarget {
                url: "http://cheap-engine:8080".into(),
                timeout_ms: 5000,
                cost_tier: CostTier::Free,
            },
        );

        let config = RouterConfig {
            routes,
            confidence_threshold: 0.1,
            ..Default::default()
        };

        // Very small budget: 0.10 total, High tier costs 0.20 per route.
        let budget = BudgetTracker::new(0.10);
        let cost_map = CostMap::default();
        let mut router = CynefinRouter::with_budget(classifier, config, budget, cost_map);

        let decision = router
            .route_with_budget("Why did the effect change?")
            .await
            .unwrap();

        // Should be over budget since High tier (0.20) > budget (0.10).
        match decision.budget_status.as_ref().unwrap() {
            BudgetDecision::OverBudget { overage, .. } => {
                assert!(*overage > 0.0);
            }
            _ => panic!("Expected OverBudget decision"),
        }
    }
}
