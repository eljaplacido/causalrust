use async_trait::async_trait;
use cynepic_core::CynefinDomain;
use serde::{Deserialize, Serialize};

/// Result of classifying a query into a Cynefin domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// The classified domain.
    pub domain: CynefinDomain,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
    /// Scores for all domains, sorted descending.
    pub all_scores: Vec<(CynefinDomain, f64)>,
}

/// Trait for query classifiers that map natural language to Cynefin domains.
#[async_trait]
pub trait QueryClassifier: Send + Sync {
    /// Classify a query string into a Cynefin domain with confidence.
    async fn classify(&self, query: &str) -> Result<ClassificationResult, ClassifierError>;
}

/// A simple keyword-based classifier for bootstrapping and testing.
///
/// Production systems should use embedding-based classifiers (Candle + HNSW).
pub struct KeywordClassifier {
    patterns: Vec<(Vec<String>, CynefinDomain)>,
}

impl KeywordClassifier {
    /// Create a classifier with default keyword patterns.
    pub fn default_patterns() -> Self {
        Self {
            patterns: vec![
                // Clear: simple lookups and definitions
                (
                    vec!["what is".into(), "define".into(), "look up".into(), "how many".into()],
                    CynefinDomain::Clear,
                ),
                // Complicated: causal and analytical questions
                (
                    vec![
                        "why did".into(), "cause".into(), "effect".into(), "impact".into(),
                        "correlation".into(), "regression".into(), "because".into(),
                    ],
                    CynefinDomain::Complicated,
                ),
                // Complex: uncertainty and exploration
                (
                    vec![
                        "uncertain".into(), "probability".into(), "might".into(), "explore".into(),
                        "what if".into(), "scenario".into(), "predict".into(),
                    ],
                    CynefinDomain::Complex,
                ),
                // Chaotic: crisis and emergency
                (
                    vec![
                        "emergency".into(), "crisis".into(), "outage".into(), "breach".into(),
                        "urgent".into(), "critical failure".into(),
                    ],
                    CynefinDomain::Chaotic,
                ),
            ],
        }
    }

    fn score_domain(&self, query: &str, keywords: &[String]) -> f64 {
        let query_lower = query.to_lowercase();
        let matches = keywords
            .iter()
            .filter(|kw| query_lower.contains(kw.as_str()))
            .count();
        if keywords.is_empty() {
            0.0
        } else {
            matches as f64 / keywords.len() as f64
        }
    }
}

#[async_trait]
impl QueryClassifier for KeywordClassifier {
    async fn classify(&self, query: &str) -> Result<ClassificationResult, ClassifierError> {
        let mut all_scores: Vec<(CynefinDomain, f64)> = self
            .patterns
            .iter()
            .map(|(keywords, domain)| (*domain, self.score_domain(query, keywords)))
            .collect();

        all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (domain, confidence) = if all_scores.is_empty() || all_scores[0].1 == 0.0 {
            (CynefinDomain::Disorder, 0.0)
        } else {
            all_scores[0]
        };

        Ok(ClassificationResult {
            domain,
            confidence,
            all_scores,
        })
    }
}

/// Errors from the classification system.
#[derive(Debug, thiserror::Error)]
pub enum ClassifierError {
    #[error("Embedding model failed: {0}")]
    EmbeddingFailed(String),

    #[error("Classification produced no confident result")]
    NoConfidentResult,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn keyword_classifier_causal_query() {
        let classifier = KeywordClassifier::default_patterns();
        let result = classifier
            .classify("Why did the sales increase? What was the cause?")
            .await
            .unwrap();
        assert_eq!(result.domain, CynefinDomain::Complicated);
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn keyword_classifier_unknown_falls_to_disorder() {
        let classifier = KeywordClassifier::default_patterns();
        let result = classifier.classify("asdfg12345").await.unwrap();
        assert_eq!(result.domain, CynefinDomain::Disorder);
    }

    #[tokio::test]
    async fn keyword_classifier_emergency() {
        let classifier = KeywordClassifier::default_patterns();
        let result = classifier
            .classify("There is an emergency outage in production!")
            .await
            .unwrap();
        assert_eq!(result.domain, CynefinDomain::Chaotic);
    }
}
