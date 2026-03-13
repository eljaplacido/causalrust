use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use cynepic_core::CynefinDomain;

/// Configuration for the Cynefin router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Mapping from Cynefin domain to upstream service URL.
    pub routes: HashMap<CynefinDomain, RouteTarget>,
    /// Minimum confidence threshold for classification.
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,
    /// Fallback domain when confidence is below threshold.
    #[serde(default)]
    pub fallback_domain: CynefinDomain,
}

/// Target for routing a classified query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteTarget {
    /// Upstream URL to forward the request to.
    pub url: String,
    /// Optional timeout in milliseconds.
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    /// Cost tier (used for cost-aware routing decisions).
    #[serde(default)]
    pub cost_tier: CostTier,
}

/// Cost tier for LLM routing decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CostTier {
    /// Free or very cheap (local SLM).
    #[default]
    Free,
    /// Low cost (e.g., OpenAI GPT-3.5, Gemini Flash).
    Low,
    /// Medium cost (e.g., GPT-4o-mini, Claude Haiku).
    Medium,
    /// High cost (e.g., GPT-4, Claude Opus, o1).
    High,
}

fn default_confidence_threshold() -> f64 {
    0.3
}

fn default_timeout() -> u64 {
    30_000
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            routes: HashMap::new(),
            confidence_threshold: default_confidence_threshold(),
            fallback_domain: CynefinDomain::Disorder,
        }
    }
}
