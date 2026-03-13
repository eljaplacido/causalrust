use async_trait::async_trait;

/// Core trait for any analytical engine in the cynepic pipeline.
///
/// Each Cynefin domain maps to an engine that implements this trait:
/// - Clear → deterministic lookup engine
/// - Complicated → causal inference engine (`cynepic-causal`)
/// - Complex → Bayesian active inference engine (`cynepic-bayes`)
/// - Chaotic → circuit breaker (returns error/fallback)
/// - Disorder → human escalation handler
#[async_trait]
pub trait AnalyticalEngine: Send + Sync {
    /// The input type this engine accepts.
    type Input: Send;
    /// The output type this engine produces.
    type Output: Send;
    /// The error type for failures.
    type Error: std::error::Error + Send + Sync;

    /// Execute analysis on the given input.
    async fn analyze(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;

    /// Human-readable name of this engine (e.g., "CausalInferenceEngine").
    fn name(&self) -> &str;

    /// The Cynefin domain this engine is designed for.
    fn target_domain(&self) -> crate::CynefinDomain;
}
