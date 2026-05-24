//! cynepic-server: Axum HTTP API for cynepic-rs decision intelligence.
//!
//! Exposes CARF/CYNEPIC capabilities as REST endpoints:
//! - `GET  /health` — health check
//! - `POST /router/classify` — Cynefin domain classification
//! - `POST /causal/estimate` — Causal effect estimation
//! - `POST /bayesian/update` — Bayesian belief update
//! - `POST /guardian/evaluate` — Policy enforcement evaluation
//! - `POST /graph/execute` — Workflow execution

use axum::{http::StatusCode, routing::{get, post}, Json, Router};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

// ── Request / Response types ───────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ClassifyRequest {
    query: String,
}

#[derive(Debug, Serialize)]
struct ClassifyResponse {
    domain: String,
    confidence: f64,
    entropy: f64,
}

#[derive(Debug, Deserialize)]
struct CausalRequest {
    treatment: Vec<f64>,
    outcome: Vec<f64>,
    #[serde(default)]
    covariates: Vec<Vec<f64>>,
}

#[derive(Debug, Serialize)]
struct CausalResponse {
    ate: f64,
    std_error: f64,
    method: String,
}

#[derive(Debug, Deserialize)]
struct BayesianRequest {
    successes: u64,
    trials: u64,
}

#[derive(Debug, Serialize)]
struct BayesianResponse {
    posterior_mean: f64,
    credible_interval: (f64, f64),
}

#[derive(Debug, Deserialize)]
struct GuardianRequest {
    domain: String,
    amount: Option<f64>,
}

#[derive(Debug, Serialize)]
struct GuardianResponse {
    verdict: String,
    reason: Option<String>,
}

// ── Handlers ────────────────────────────────────────────────────────────

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "cynepic-server",
        "version": "0.2.0"
    }))
}

async fn classify(Json(req): Json<ClassifyRequest>) -> Result<Json<ClassifyResponse>, StatusCode> {
    let router = cynepic_router::CynefinRouter::new();
    let result = router.classify(&req.query);
    Ok(Json(ClassifyResponse {
        domain: format!("{}", result.domain),
        confidence: result.confidence,
        entropy: result.entropy,
    }))
}

async fn causal_estimate(Json(req): Json<CausalRequest>) -> Result<Json<CausalResponse>, StatusCode> {
    let est = cynepic_causal::LinearATEEstimator::difference_in_means(&req.treatment, &req.outcome);
    Ok(Json(CausalResponse {
        ate: est.ate,
        std_error: est.std_error,
        method: "difference_in_means".into(),
    }))
}

async fn bayesian_update(Json(req): Json<BayesianRequest>) -> Result<Json<BayesianResponse>, StatusCode> {
    use cynepic_bayes::BetaBinomial;
    let mut model = BetaBinomial::uniform();
    model.update(req.successes, req.trials - req.successes);
    let ci = model.credible_interval_95();
    Ok(Json(BayesianResponse {
        posterior_mean: model.mean(),
        credible_interval: ci,
    }))
}

async fn guardian_evaluate(Json(req): Json<GuardianRequest>) -> Json<GuardianResponse> {
    let chain = cynepic_guardian::PolicyChain::default();
    let decision = chain.evaluate(&cynepic_core::PolicyContext::default());
    Json(GuardianResponse {
        verdict: format!("{:?}", decision),
        reason: None,
    })
}

async fn graph_execute() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "executed",
        "steps": 0
    }))
}

// ── Main ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/health", get(health))
        .route("/router/classify", post(classify))
        .route("/causal/estimate", post(causal_estimate))
        .route("/bayesian/update", post(bayesian_update))
        .route("/guardian/evaluate", post(guardian_evaluate))
        .route("/graph/execute", post(graph_execute));

    let addr = SocketAddr::from(([127, 0, 0, 1], 4310));
    tracing::info!("cynepic-server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
