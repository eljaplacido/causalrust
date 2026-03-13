//! # cynepic-router
//!
//! A Cynefin-based semantic query router and AI proxy server.
//!
//! Classifies incoming queries by complexity domain and routes them to the
//! appropriate analytical engine or upstream LLM service. Deployable as a
//! standalone binary (`cynefin-proxy`) — like Nginx, but for AI routing.
//!
//! # Key Features
//!
//! - **Semantic classification**: Embeds queries and classifies by Cynefin domain
//! - **Cost-aware routing**: Routes simple queries to cheap/local models
//! - **Confidence scoring**: Every classification includes a confidence score
//! - **Standalone proxy**: Run as an HTTP reverse proxy for AI services

pub mod budget;
pub mod classifier;
pub mod config;
pub mod eval;
pub mod router;

pub use budget::{BudgetDecision, BudgetTracker, CostMap};
pub use classifier::{ClassificationResult, QueryClassifier};
pub use eval::ClassifierMetrics;
pub use router::{CynefinRouter, RoutingDecision};
