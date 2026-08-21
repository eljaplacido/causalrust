//! `cynepic-server`: an HTTP surface over the cynepic-rs crates.
//!
//! ```text
//! GET  /health              liveness
//! POST /router/classify     Cynefin domain classification
//! POST /causal/estimate     causal effect estimation
//! POST /bayes/update        Bayesian belief update
//! POST /guardian/evaluate   policy evaluation
//! ```
//!
//! # Errors are part of the API
//!
//! The crates underneath return `Result` for conditions that used to be
//! plausible-looking numbers: a rank-deficient design, an empty treatment arm,
//! insufficient overlap for weighting. Those are **not** 500s. They are 422s
//! with the reason attached, because they describe the caller's data rather
//! than a fault in the service, and a caller who is told "your treatment arm is
//! empty" can act where one who is told "internal server error" cannot.
//!
//! # Every estimate carries its provenance
//!
//! Responses include the estimand (ATE / ATT / ATC / LATE), the kind of standard
//! error, and the confidence interval — not a bare number. `cynepic-causal`
//! makes `ATEResult` unconstructible without those, and this API would be
//! discarding the guarantee if it serialised only the point estimate.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

use cynepic_bayes::priors::BetaBinomial;
use cynepic_causal::estimate::linear::LinearATEEstimator;
use cynepic_causal::estimate::propensity::PropensityScoreEstimator;
use cynepic_causal::{ATEResult, EstimationError};
use cynepic_core::CynefinDomain;
use cynepic_router::classifier::{KeywordClassifier, QueryClassifier};

// ── Errors ──────────────────────────────────────────────────────────────

/// An error the caller can act on.
#[derive(Debug, Serialize)]
struct ApiError {
    error: String,
    detail: String,
}

/// Wrapper so estimator errors map to a status that describes them.
///
/// `Debug` so a test that expects success can print what came back instead.
#[derive(Debug)]
struct Failure(StatusCode, String, String);

impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        let Failure(status, error, detail) = self;
        (status, Json(ApiError { error, detail })).into_response()
    }
}

impl From<EstimationError> for Failure {
    /// Estimation errors describe the caller's *data*, not a service fault, so
    /// they are 422 rather than 500. The distinction matters operationally: a
    /// 500 pages someone, a 422 tells the caller what to fix.
    fn from(e: EstimationError) -> Self {
        let kind = match e {
            EstimationError::LengthMismatch { .. } => "length_mismatch",
            EstimationError::NoObservations => "no_observations",
            EstimationError::EmptyArm { .. } => "empty_arm",
            EstimationError::InsufficientData { .. } => "insufficient_data",
            EstimationError::CrossFittingNotApplicable { .. } => "cross_fitting_not_applicable",
            EstimationError::RankDeficient { .. } => "rank_deficient",
            EstimationError::NotConverged { .. } => "not_converged",
            EstimationError::Separation { .. } => "separation",
            EstimationError::InsufficientOverlap { .. } => "insufficient_overlap",
            EstimationError::WeakInstrument { .. } => "weak_instrument",
            EstimationError::ConstantTreatment { .. } => "constant_treatment",
            EstimationError::NotFinite { .. } => "not_finite",
        };
        Failure(
            StatusCode::UNPROCESSABLE_ENTITY,
            kind.to_string(),
            e.to_string(),
        )
    }
}

// ── Router ──────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ClassifyRequest {
    query: String,
}

#[derive(Debug, Serialize)]
struct ClassifyResponse {
    domain: String,
    confidence: f64,
    /// Normalised Shannon entropy of the score distribution.
    ///
    /// The signal an escalation policy triggers on: high entropy means the
    /// classifier could not separate the domains, which is a different state
    /// from a confident answer and must not be collapsed into one.
    entropy: f64,
    /// True when the classifier declined to guess.
    abstained: bool,
}

async fn classify(Json(req): Json<ClassifyRequest>) -> Result<Json<ClassifyResponse>, Failure> {
    let classifier = KeywordClassifier::default_patterns();
    let result = classifier.classify(&req.query).await.map_err(|e| {
        Failure(
            StatusCode::INTERNAL_SERVER_ERROR,
            "classifier_failed".into(),
            e.to_string(),
        )
    })?;

    Ok(Json(ClassifyResponse {
        domain: format!("{:?}", result.domain),
        confidence: result.confidence,
        entropy: result.entropy,
        abstained: result.domain == CynefinDomain::Disorder,
    }))
}

// ── Causal ──────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct CausalRequest {
    treatment: Vec<f64>,
    outcome: Vec<f64>,
    /// Row-major covariates. Empty means an unadjusted estimate.
    #[serde(default)]
    covariates: Vec<Vec<f64>>,
    /// `"ols"` (default), `"ipw"` or `"att"`.
    #[serde(default)]
    method: Option<String>,
}

#[derive(Debug, Serialize)]
struct CausalResponse {
    ate: f64,
    std_error: f64,
    /// Which quantity this is. ATE, ATT and LATE are different populations and
    /// coincide only under effect homogeneity.
    estimand: String,
    /// Which population that estimand describes, in words.
    population: String,
    std_error_kind: String,
    confidence_interval: Option<(f64, f64)>,
    n_obs: usize,
    method: String,
    diagnostics: DiagnosticsResponse,
}

#[derive(Debug, Serialize)]
struct DiagnosticsResponse {
    rank: Option<(usize, usize)>,
    /// Kish effective sample size — the honest `n` behind a weighted estimate.
    effective_n: Option<f64>,
    propensity_range: Option<(f64, f64)>,
    /// Effective degrees of freedom of the *variance estimate*.
    ///
    /// A value in the tens rather than the hundreds means the interval rests on
    /// very few effective observations, whatever `n_obs` says.
    variance_dof: Option<f64>,
    converged: Option<bool>,
    arm_sizes: Option<(usize, usize)>,
}

fn to_response(r: &ATEResult, method: &str) -> CausalResponse {
    let d = r.diagnostics();
    CausalResponse {
        ate: r.ate(),
        std_error: r.std_error(),
        estimand: r.estimand().label().to_string(),
        population: r.estimand().population().to_string(),
        std_error_kind: format!("{:?}", r.std_error_kind()),
        confidence_interval: r.confidence_interval(0.95),
        n_obs: r.n_obs(),
        method: method.to_string(),
        diagnostics: DiagnosticsResponse {
            rank: d.rank,
            effective_n: d.effective_n,
            propensity_range: d.propensity_range,
            variance_dof: d.variance_dof,
            converged: d.convergence.map(|c| c.converged),
            arm_sizes: d.arm_sizes,
        },
    }
}

/// Build an `n x p` covariate matrix from row-major JSON.
fn covariate_matrix(rows: &[Vec<f64>], n: usize) -> Result<Array2<f64>, Failure> {
    if rows.is_empty() {
        return Ok(Array2::zeros((n, 0)));
    }
    if rows.len() != n {
        return Err(Failure(
            StatusCode::UNPROCESSABLE_ENTITY,
            "length_mismatch".into(),
            format!("{} covariate rows for {n} units", rows.len()),
        ));
    }
    let p = rows[0].len();
    if rows.iter().any(|r| r.len() != p) {
        return Err(Failure(
            StatusCode::UNPROCESSABLE_ENTITY,
            "ragged_covariates".into(),
            "every covariate row must have the same length".into(),
        ));
    }
    Ok(Array2::from_shape_fn((n, p), |(i, j)| rows[i][j]))
}

async fn causal_estimate(Json(req): Json<CausalRequest>) -> Result<Json<CausalResponse>, Failure> {
    let n = req.treatment.len();
    let treatment = Array1::from_vec(req.treatment.clone());
    let outcome = Array1::from_vec(req.outcome.clone());
    let covariates = covariate_matrix(&req.covariates, n)?;

    let method = req.method.as_deref().unwrap_or("ols");
    let result = match method {
        "ipw" => PropensityScoreEstimator::ipw(&treatment, &outcome, &covariates)?,
        "att" => PropensityScoreEstimator::att(&treatment, &outcome, &covariates)?,
        "ols" if covariates.ncols() > 0 => {
            LinearATEEstimator::ols_adjusted(&treatment, &outcome, &covariates)?
        }
        "ols" => LinearATEEstimator::difference_in_means(&treatment, &outcome)?,
        other => {
            return Err(Failure(
                StatusCode::BAD_REQUEST,
                "unknown_method".into(),
                format!("'{other}' is not one of: ols, ipw, att"),
            ));
        }
    };

    Ok(Json(to_response(&result, method)))
}

// ── Bayes ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct BayesRequest {
    successes: u64,
    trials: u64,
}

#[derive(Debug, Serialize)]
struct BayesResponse {
    posterior_mean: f64,
    /// Exact Beta quantile interval, not a normal approximation to one.
    credible_interval: (f64, f64),
    alpha: f64,
    beta: f64,
}

async fn bayes_update(Json(req): Json<BayesRequest>) -> Result<Json<BayesResponse>, Failure> {
    if req.successes > req.trials {
        return Err(Failure(
            StatusCode::UNPROCESSABLE_ENTITY,
            "successes_exceed_trials".into(),
            format!("{} successes in {} trials", req.successes, req.trials),
        ));
    }

    let mut model = BetaBinomial::uniform();
    model.update(req.successes, req.trials - req.successes);

    Ok(Json(BayesResponse {
        posterior_mean: model.mean(),
        credible_interval: model.credible_interval_95(),
        alpha: model.alpha,
        beta: model.beta,
    }))
}

// ── Guardian ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct GuardianRequest {
    action: String,
    #[serde(default)]
    context: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct GuardianResponse {
    verdict: String,
    reason: Option<String>,
}

async fn guardian_evaluate(
    Json(req): Json<GuardianRequest>,
) -> Result<Json<GuardianResponse>, Failure> {
    use cynepic_core::PolicyDecision;
    use cynepic_guardian::policy::PolicyChain;

    // An empty chain approves by default. Exposed so a deployment can see what
    // it has configured rather than assume; a guardrail nobody has configured
    // is a guardrail that is not guarding.
    let chain = PolicyChain::new();
    let decision = chain
        .evaluate(&req.action, &req.context)
        .await
        .map_err(|e| {
            Failure(
                StatusCode::INTERNAL_SERVER_ERROR,
                "policy_error".into(),
                e.to_string(),
            )
        })?;

    let (verdict, reason) = match decision {
        PolicyDecision::Approve => ("approve", None),
        PolicyDecision::Reject { reason } => ("reject", Some(reason)),
        PolicyDecision::Escalate { target } => ("escalate", Some(format!("{target:?}"))),
    };

    Ok(Json(GuardianResponse {
        verdict: verdict.to_string(),
        reason,
    }))
}

// ── Health ──────────────────────────────────────────────────────────────

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "cynepic-server",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = Router::new()
        .route("/health", get(health))
        .route("/router/classify", post(classify))
        .route("/causal/estimate", post(causal_estimate))
        .route("/bayes/update", post(bayes_update))
        .route("/guardian/evaluate", post(guardian_evaluate));

    // Bind address is configurable so the service is not pinned to loopback in
    // a container, where loopback is unreachable from outside the namespace.
    let addr: SocketAddr = std::env::var("CYNEPIC_BIND")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| SocketAddr::from(([127, 0, 0, 1], 4310)));

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(%addr, error = %e, "failed to bind");
            std::process::exit(1);
        }
    };
    tracing::info!(%addr, "cynepic-server listening");

    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!(error = %e, "server terminated");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Recover status and body from a handler's `Failure` without a listener.
    ///
    /// The handlers are ordinary async functions over `Json<T>`, so they can be
    /// called directly. Spinning up a TCP listener to test them would measure
    /// axum's routing table rather than this crate's behaviour, and would make
    /// the suite dependent on a free port.
    fn failure_parts(f: Failure) -> (StatusCode, String) {
        let Failure(status, kind, _) = f;
        (status, kind)
    }

    #[tokio::test]
    async fn health_names_the_service_and_its_version() {
        let Json(v) = health().await;
        assert_eq!(v["status"], "healthy");
        assert_eq!(v["service"], "cynepic-server");
        // A health endpoint that does not report a version cannot answer the
        // question it is usually asked during an incident: what is deployed?
        assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
    }

    // ── The error contract ──────────────────────────────────────────────
    //
    // These are the tests that earn their keep. The mapping from estimator
    // errors to HTTP is the API's promise that "your data cannot support this"
    // is distinguishable from "the service broke", and it is an exhaustive
    // match that a new error variant will not compile without.

    #[tokio::test]
    async fn data_errors_are_422_not_500() {
        // A 500 pages someone at 3am. A 422 tells the caller what to fix. Every
        // estimation error describes the caller's data, so every one is a 422.
        for e in [
            EstimationError::NoObservations,
            EstimationError::EmptyArm {
                arm: "treated",
                n_treated: 0,
                n_control: 10,
            },
            EstimationError::InsufficientData { n: 3, p: 5 },
            EstimationError::CrossFittingNotApplicable {
                available: 12,
                required: 80,
                parameters: 4,
            },
        ] {
            let (status, _) = failure_parts(Failure::from(e));
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        }
    }

    #[tokio::test]
    async fn every_error_kind_is_distinct() {
        // Two different failures reported under one name are indistinguishable
        // to a caller writing a retry policy, which is the whole point of
        // carrying a kind alongside the status.
        let kinds: Vec<String> = vec![
            EstimationError::NoObservations,
            EstimationError::EmptyArm {
                arm: "treated",
                n_treated: 0,
                n_control: 10,
            },
            EstimationError::InsufficientData { n: 3, p: 5 },
            EstimationError::CrossFittingNotApplicable {
                available: 12,
                required: 80,
                parameters: 4,
            },
            EstimationError::LengthMismatch {
                name_a: "treatment",
                len_a: 3,
                name_b: "outcome",
                len_b: 4,
            },
        ]
        .into_iter()
        .map(|e| failure_parts(Failure::from(e)).1)
        .collect();

        let mut unique = kinds.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(
            unique.len(),
            kinds.len(),
            "duplicate error kinds: {kinds:?}"
        );
        assert!(
            kinds.iter().all(|k| !k.is_empty()),
            "an empty kind tells a caller nothing"
        );
    }

    // ── Causal ──────────────────────────────────────────────────────────

    fn causal_request(method: &str) -> CausalRequest {
        // Twenty units, a real effect of 2.0, one covariate.
        let treatment: Vec<f64> = (0..20).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        let covariates: Vec<Vec<f64>> = (0..20).map(|i| vec![f64::from(i % 5)]).collect();
        let outcome: Vec<f64> = (0..20)
            .map(|i| 2.0 * f64::from(u8::from(i % 2 == 0)) + f64::from(i % 5) * 0.5)
            .collect();
        CausalRequest {
            treatment,
            outcome,
            covariates,
            method: Some(method.to_string()),
        }
    }

    #[tokio::test]
    async fn an_estimate_carries_its_provenance() {
        let Json(r) = causal_estimate(Json(causal_request("ols")))
            .await
            .expect("well-posed");
        // The point of `ATEResult` having no public constructor is that a number
        // cannot be separated from what it means. This asserts the API does not
        // undo that by serialising only the point estimate.
        assert!(!r.estimand.is_empty(), "estimand must be reported");
        assert!(!r.population.is_empty(), "population must be reported");
        assert!(!r.std_error_kind.is_empty(), "SE kind must be reported");
        assert!(r.confidence_interval.is_some(), "interval must be reported");
        assert_eq!(r.n_obs, 20);
        assert_eq!(r.method, "ols");
        assert!((r.ate - 2.0).abs() < 0.5, "ate {} is far from 2.0", r.ate);
    }

    #[tokio::test]
    async fn mismatched_input_lengths_are_rejected_with_a_reason() {
        let mut req = causal_request("ols");
        req.outcome.truncate(5);
        let (status, kind) = failure_parts(
            causal_estimate(Json(req))
                .await
                .expect_err("mismatched lengths cannot be estimated"),
        );
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(kind, "length_mismatch");
    }

    #[tokio::test]
    async fn ragged_covariates_are_rejected_before_reaching_an_estimator() {
        let mut req = causal_request("ols");
        req.covariates[3] = vec![1.0, 2.0];
        let (status, kind) = failure_parts(
            causal_estimate(Json(req))
                .await
                .expect_err("ragged rows are not a matrix"),
        );
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(kind, "ragged_covariates");
    }

    #[tokio::test]
    async fn an_unknown_method_is_a_400_that_lists_the_known_ones() {
        let (status, kind) = failure_parts(
            causal_estimate(Json(causal_request("magic")))
                .await
                .expect_err("no such method"),
        );
        // 400, not 422: the request itself is malformed, not the data in it.
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(kind, "unknown_method");
    }

    #[tokio::test]
    async fn every_advertised_method_works() {
        // The module docs name three. A method that 404s at runtime is worse
        // than one that was never documented.
        for method in ["ols", "ipw", "att"] {
            let r = causal_estimate(Json(causal_request(method))).await;
            assert!(
                r.is_ok(),
                "method '{method}' failed on well-posed data: {:?}",
                r.err().map(failure_parts)
            );
        }
    }

    // ── Bayes ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn a_belief_update_returns_an_exact_interval() {
        let Json(r) = bayes_update(Json(BayesRequest {
            successes: 7,
            trials: 10,
        }))
        .await
        .expect("valid counts");
        assert!(
            (r.posterior_mean - 8.0 / 12.0).abs() < 1e-9,
            "{}",
            r.posterior_mean
        );
        let (lo, hi) = r.credible_interval;
        assert!(lo > 0.0 && hi < 1.0 && lo < hi, "interval ({lo}, {hi})");
        assert!(
            lo < r.posterior_mean && r.posterior_mean < hi,
            "the mean must lie inside its own interval"
        );
    }

    #[tokio::test]
    async fn more_successes_than_trials_is_rejected() {
        let (status, kind) = failure_parts(
            bayes_update(Json(BayesRequest {
                successes: 11,
                trials: 10,
            }))
            .await
            .expect_err("impossible counts"),
        );
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(kind, "successes_exceed_trials");
    }

    // ── Router ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn classification_reports_entropy_alongside_the_domain() {
        let Json(r) = classify(Json(ClassifyRequest {
            query: "what is the current retry limit".to_string(),
        }))
        .await
        .expect("infallible");
        assert!(!r.domain.is_empty());
        assert!((0.0..=1.0).contains(&r.confidence));
        // Entropy is the signal an escalation policy triggers on. An API that
        // returned only the domain would leave a caller unable to tell a
        // confident route from a coin flip.
        assert!((0.0..=1.0).contains(&r.entropy));
    }

    #[tokio::test]
    async fn an_unroutable_query_says_so_rather_than_guessing() {
        let Json(r) = classify(Json(ClassifyRequest {
            query: "zzzz qqqq".to_string(),
        }))
        .await
        .expect("infallible");
        assert_eq!(r.domain, format!("{:?}", CynefinDomain::Disorder));
        assert!(r.confidence == 0.0, "confidence {}", r.confidence);
    }

    // ── Guardian ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn an_empty_policy_chain_approves_and_the_response_shows_it() {
        let Json(r) = guardian_evaluate(Json(GuardianRequest {
            action: "deploy".to_string(),
            context: serde_json::json!({"role": "admin"}),
        }))
        .await
        .expect("infallible");
        // Documented behaviour: a chain nobody configured approves. The test
        // exists so that is a decision on the record rather than a surprise.
        assert_eq!(r.verdict, "approve");
        assert!(r.reason.is_none());
    }
}
