//! # cynepic-router
//!
//! Cynefin-domain query classification and cost-aware routing.
//!
//! Classifies incoming queries by complexity domain and routes them to the
//! appropriate analytical engine or upstream LLM service. Deployable as a
//! embeddable library. There is no standalone binary.
//!
//! # Key Features
//!
//! - **Semantic classification**: Embeds queries and classifies by Cynefin domain
//! - **Cost-aware routing**: Routes simple queries to cheap/local models
//! - **Confidence scoring**: Every classification includes a confidence score
//! - **Embedded**: call the classifier and router directly from your service

pub mod budget;
pub mod classifier;
pub mod config;
pub mod eval;
pub mod lexical;
pub mod router;

pub use budget::{BudgetDecision, BudgetTracker, CostMap};
pub use classifier::{ClassificationResult, ClassifierError, KeywordClassifier, QueryClassifier};
pub use config::{CostTier, RouteTarget, RouterConfig};
pub use eval::ClassifierMetrics;
pub use lexical::LexicalClassifier;
pub use router::{CynefinRouter, RoutingDecision};
