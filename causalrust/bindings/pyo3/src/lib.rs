//! PyO3 bindings for cynepic-rs.
//!
//! Provides Python-accessible versions of the core cynepic-rs types and
//! engines, enabling high-performance CARF/CYNEPIC workflows from Python.
//!
//! Usage:
//! ```python
//! import cynepic
//! domain = cynepic.CynefinDomain.COMPLICATED
//! dag = cynepic.CausalDag()
//! dag.add_variable("treatment")
//! ```

use pyo3::prelude::*;

// ── Core types ────────────────────────────────────────────────────────

/// Cynefin domain classification for problem complexity.
///
/// `from_py_object` is opted into explicitly: pyo3 0.29 deprecates the implicit
/// `FromPyObject` derive for `#[pyclass]` types that also implement `Clone`.
/// This one is genuinely passed *into* Rust from Python — a caller selects a
/// domain — so the conversion is wanted rather than incidental.
#[pyclass(eq, eq_int, from_py_object, name = "CynefinDomain")]
#[derive(Clone, Debug, PartialEq)]
pub enum PyCynefinDomain {
    Clear,
    Complicated,
    Complex,
    Chaotic,
    Disorder,
}

#[pymethods]
impl PyCynefinDomain {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        match self {
            Self::Clear => "clear".into(),
            Self::Complicated => "complicated".into(),
            Self::Complex => "complex".into(),
            Self::Chaotic => "chaotic".into(),
            Self::Disorder => "disorder".into(),
        }
    }
}

impl From<cynepic_core::CynefinDomain> for PyCynefinDomain {
    fn from(d: cynepic_core::CynefinDomain) -> Self {
        match d {
            cynepic_core::CynefinDomain::Clear => Self::Clear,
            cynepic_core::CynefinDomain::Complicated => Self::Complicated,
            cynepic_core::CynefinDomain::Complex => Self::Complex,
            cynepic_core::CynefinDomain::Chaotic => Self::Chaotic,
            cynepic_core::CynefinDomain::Disorder => Self::Disorder,
        }
    }
}

// ── Causal DAG ────────────────────────────────────────────────────────

/// A causal Directed Acyclic Graph with d-separation and adjustment tools.
#[pyclass(name = "CausalDag")]
#[derive(Debug)]
pub struct PyCausalDag {
    inner: cynepic_causal::CausalDag,
}

#[pymethods]
impl PyCausalDag {
    #[new]
    fn new() -> Self {
        Self {
            inner: cynepic_causal::CausalDag::new(),
        }
    }

    fn add_variable(&mut self, name: &str) {
        self.inner.add_variable(name);
    }

    /// Add a causal edge.
    ///
    /// Raises `ValueError` if the edge would create a cycle or is a self-loop.
    /// A `CausalDag` that is not acyclic invalidates every algorithm built on
    /// it, so the check is at insertion rather than at use.
    fn add_edge(&mut self, from: &str, to: &str) -> PyResult<()> {
        self.inner
            .add_edge(from, to)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Mark a variable as unobserved.
    ///
    /// Identification will then decline rather than returning an adjustment set
    /// containing something you cannot measure.
    fn mark_latent(&mut self, name: &str) -> PyResult<()> {
        self.inner
            .mark_latent(name)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Whether two variables are d-separated given a conditioning set.
    ///
    /// Raises `ValueError` for a variable the graph does not contain, including
    /// in the conditioning set. Returning `True` — "conditionally independent"
    /// — for a misspelled name would turn a typo into a positive finding.
    fn d_separated(&self, x: &str, y: &str, z: Vec<String>) -> PyResult<bool> {
        let set: std::collections::HashSet<String> = z.into_iter().collect();
        cynepic_causal::d_separated(&self.inner, x, y, &set)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Find a verified backdoor adjustment set.
    ///
    /// Returns the variables to adjust for. An empty list is a *successful*
    /// identification meaning no adjustment is needed — distinct from raising,
    /// which means no observed set can block the backdoor paths.
    ///
    /// Raises `ValueError` when the effect is not identifiable by adjustment,
    /// with the reason: either latent variables would be required, or no
    /// observed set blocks every path.
    fn find_backdoor_adjustment(&self, treatment: &str, outcome: &str) -> PyResult<Vec<String>> {
        let adj = cynepic_causal::BackdoorCriterion::find(&self.inner, treatment, outcome)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(adj.sorted().into_iter().map(str::to_string).collect())
    }

    fn __repr__(&self) -> String {
        format!(
            "CausalDag(variables={}, edges={})",
            self.inner.num_variables(),
            self.inner.num_edges()
        )
    }
}

// ── Causal effect estimation ───────────────────────────────────────────

/// The result of estimating a causal effect.
///
/// # Why this is a class and not a float
///
/// `cynepic-causal`'s `ATEResult` has no public constructor, deliberately: a
/// number cannot be separated from what it means. A binding that returned a
/// bare float would undo that guarantee at the language boundary, which is
/// exactly where it matters most — the Python caller is the one furthest from
/// the assumptions.
///
/// So every estimate arrives with the *estimand* it computed (the average
/// effect over everyone, or only over the treated — different questions), the
/// kind of standard error behind its interval, and the diagnostics from the fit.
// `skip_from_py_object`: a `CausalEffect` travels out of Rust and is never
// passed back in, so the conversion is incidental rather than wanted. Opting
// out is also the safer default here — accepting one as an *input* would let a
// caller hand-build a result and defeat the provenance guarantee entirely.
#[pyclass(name = "CausalEffect", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyCausalEffect {
    /// The point estimate.
    #[pyo3(get)]
    ate: f64,
    /// Its standard error.
    #[pyo3(get)]
    std_error: f64,
    /// 95% confidence interval, as a `(low, high)` tuple.
    #[pyo3(get)]
    confidence_interval: Option<(f64, f64)>,
    /// Which question this answers: "ATE", "ATT", "ATC" or "LATE".
    #[pyo3(get)]
    estimand: String,
    /// Whose effect it is — the population the estimand refers to.
    #[pyo3(get)]
    population: String,
    /// How the standard error was computed.
    #[pyo3(get)]
    std_error_kind: String,
    /// Observations used.
    #[pyo3(get)]
    n_obs: usize,
    /// Kish effective sample size, where weighting was involved.
    ///
    /// **The number to read before trusting a weighted interval.** Far below
    /// `n_obs` means the estimate rests on a handful of observations however
    /// large the dataset is.
    #[pyo3(get)]
    effective_n: Option<f64>,
    /// Effective degrees of freedom of the variance estimate.
    ///
    /// In the single digits on a large dataset means the interval is resting on
    /// very few observations — and under heavy weighting a *narrow* interval in
    /// that regime is the case to distrust, not to trust.
    #[pyo3(get)]
    variance_dof: Option<f64>,
    /// Smallest and largest fitted propensity score, where one was fitted.
    #[pyo3(get)]
    propensity_range: Option<(f64, f64)>,
}

#[pymethods]
impl PyCausalEffect {
    /// Whether the 95% interval excludes zero.
    ///
    /// Returns `None` rather than `False` when no interval exists, so "we could
    /// not tell" stays distinguishable from "no effect".
    #[getter]
    fn significant(&self) -> Option<bool> {
        self.confidence_interval
            .map(|(lo, hi)| lo > 0.0 || hi < 0.0)
    }

    fn __repr__(&self) -> String {
        match self.confidence_interval {
            Some((lo, hi)) => format!(
                "CausalEffect({}={:.4}, 95% CI [{:.4}, {:.4}], n={})",
                self.estimand, self.ate, lo, hi, self.n_obs
            ),
            None => format!(
                "CausalEffect({}={:.4}, se={:.4}, n={})",
                self.estimand, self.ate, self.std_error, self.n_obs
            ),
        }
    }
}

impl PyCausalEffect {
    fn from_result(r: &cynepic_causal::ATEResult) -> Self {
        let d = r.diagnostics();
        Self {
            ate: r.ate(),
            std_error: r.std_error(),
            confidence_interval: r.confidence_interval(0.95),
            estimand: r.estimand().label().to_string(),
            population: r.estimand().population().to_string(),
            std_error_kind: format!("{:?}", r.std_error_kind()),
            n_obs: r.n_obs(),
            effective_n: d.effective_n,
            variance_dof: d.variance_dof,
            propensity_range: d.propensity_range,
        }
    }
}

/// Build ndarray inputs from Python lists, checking shape before the estimator
/// does, so the error names the Python-side problem.
fn arrays(
    treatment: Vec<f64>,
    outcome: Vec<f64>,
    covariates: Option<Vec<Vec<f64>>>,
) -> PyResult<(
    ndarray::Array1<f64>,
    ndarray::Array1<f64>,
    ndarray::Array2<f64>,
)> {
    let n = treatment.len();
    if outcome.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "treatment has {n} entries but outcome has {}",
            outcome.len()
        )));
    }
    let rows = covariates.unwrap_or_default();
    let x = if rows.is_empty() {
        ndarray::Array2::zeros((n, 0))
    } else {
        if rows.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} covariate rows for {n} units",
                rows.len()
            )));
        }
        let p = rows[0].len();
        if rows.iter().any(|r| r.len() != p) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "every covariate row must have the same length",
            ));
        }
        ndarray::Array2::from_shape_fn((n, p), |(i, j)| rows[i][j])
    };
    Ok((
        ndarray::Array1::from_vec(treatment),
        ndarray::Array1::from_vec(outcome),
        x,
    ))
}

fn to_py_err(e: cynepic_causal::EstimationError) -> PyErr {
    // ValueError, not RuntimeError: every one of these describes the caller's
    // *data*, and a caller told "your treatment arm is empty" can act where one
    // told "internal error" cannot.
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Estimate an average treatment effect by adjusting for covariates.
///
/// Least squares with a heteroskedasticity-robust standard error. Use this when
/// the relationship between the covariates and the outcome is plausibly linear —
/// it achieves nominal interval coverage on every cell of the validation grid.
///
/// # Errors
///
/// `ValueError` when the data cannot support an estimate: mismatched lengths, an
/// empty treatment arm, a rank-deficient design. It **refuses rather than
/// returning a plausible-looking number** — numpy's least squares, given the
/// same collinear design, returns `-0.000562` with no warning, which reads as
/// "no effect" rather than "undefined".
#[pyfunction]
#[pyo3(signature = (treatment, outcome, covariates=None))]
fn estimate_ate(
    treatment: Vec<f64>,
    outcome: Vec<f64>,
    covariates: Option<Vec<Vec<f64>>>,
) -> PyResult<PyCausalEffect> {
    let (t, y, x) = arrays(treatment, outcome, covariates)?;
    let r = if x.ncols() > 0 {
        cynepic_causal::estimate::linear::LinearATEEstimator::ols_adjusted(&t, &y, &x)
    } else {
        cynepic_causal::estimate::linear::LinearATEEstimator::difference_in_means(&t, &y)
    }
    .map_err(to_py_err)?;
    Ok(PyCausalEffect::from_result(&r))
}

/// Estimate an average treatment effect by inverse-probability weighting.
///
/// Fits a propensity model, then reweights so the treated and control groups
/// resemble each other. Use this when the *treatment assignment* is easier to
/// model than the outcome.
///
/// The propensity model is fitted out-of-fold where the data supports it, which
/// costs about four times as much as fitting it in-sample and is what makes the
/// confidence intervals correct.
///
/// **Read `effective_n` and `variance_dof` on the result before trusting the
/// interval.** Under heavy weighting a narrow interval is the case to distrust.
///
/// # Errors
///
/// `ValueError` as above, plus insufficient overlap — if the treated and control
/// groups barely resemble each other, no amount of reweighting fixes it and the
/// estimator says so instead of returning a confident number.
#[pyfunction]
fn estimate_ate_weighted(
    treatment: Vec<f64>,
    outcome: Vec<f64>,
    covariates: Vec<Vec<f64>>,
) -> PyResult<PyCausalEffect> {
    let (t, y, x) = arrays(treatment, outcome, Some(covariates))?;
    let r = cynepic_causal::estimate::propensity::PropensityScoreEstimator::ipw(&t, &y, &x)
        .map_err(to_py_err)?;
    Ok(PyCausalEffect::from_result(&r))
}

/// Estimate the effect **on the treated** rather than on everyone.
///
/// A different question from [`estimate_ate_weighted`], and usually the one
/// meant by "did the campaign work" — the effect among those who actually
/// received it.
///
/// # Errors
///
/// As [`estimate_ate_weighted`].
#[pyfunction]
fn estimate_att(
    treatment: Vec<f64>,
    outcome: Vec<f64>,
    covariates: Vec<Vec<f64>>,
) -> PyResult<PyCausalEffect> {
    let (t, y, x) = arrays(treatment, outcome, Some(covariates))?;
    let r = cynepic_causal::estimate::propensity::PropensityScoreEstimator::att(&t, &y, &x)
        .map_err(to_py_err)?;
    Ok(PyCausalEffect::from_result(&r))
}

// ── Beta-Binomial Bayesian Prior ───────────────────────────────────────

/// Beta-Binomial conjugate prior for binary outcome tracking.
#[pyclass(name = "BetaBinomial")]
#[derive(Debug)]
pub struct PyBetaBinomial {
    inner: cynepic_bayes::BetaBinomial,
}

#[pymethods]
impl PyBetaBinomial {
    /// Create with uniform prior Beta(1, 1).
    #[new]
    fn new() -> Self {
        Self {
            inner: cynepic_bayes::BetaBinomial::uniform(),
        }
    }

    /// Create with custom prior Beta(alpha, beta).
    #[staticmethod]
    fn with_prior(alpha: f64, beta: f64) -> PyResult<Self> {
        Ok(Self {
            inner: cynepic_bayes::BetaBinomial::new(alpha, beta)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        })
    }

    /// Record n successes and m failures.
    fn update(&mut self, successes: u64, failures: u64) {
        self.inner.update(successes, failures);
    }

    /// Posterior mean probability of success.
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn __repr__(&self) -> String {
        format!("BetaBinomial(mean={:.3})", self.inner.mean())
    }
}

// ── Circuit Breaker ────────────────────────────────────────────────────

/// A circuit breaker that opens after a configurable number of failures.
#[pyclass(name = "CircuitBreaker")]
#[derive(Debug)]
pub struct PyCircuitBreaker {
    inner: cynepic_guardian::CircuitBreaker,
}

#[pymethods]
impl PyCircuitBreaker {
    /// Create with failure threshold and recovery timeout (seconds).
    #[new]
    fn new(failure_threshold: u64, recovery_timeout_secs: u64) -> Self {
        use std::time::Duration;
        Self {
            inner: cynepic_guardian::CircuitBreaker::new(
                failure_threshold,
                Duration::from_secs(recovery_timeout_secs),
            ),
        }
    }

    /// Record a failure, tripping the breaker once the threshold is reached.
    ///
    /// # The bug this replaces
    ///
    /// Both this method and `record_success` had **empty bodies** — a comment
    /// claiming the underlying calls were synchronous, and no call. The Python
    /// circuit breaker therefore recorded nothing and `is_open` was always
    /// `False`: a guardrail that could not trip, on the surface most likely to
    /// be used by someone who could not read the Rust to check.
    ///
    /// `CircuitBreaker::record_failure` is `async` — it takes a `tokio::Mutex`
    /// to restart the trip clock — which is presumably why it was left out.
    /// That is a reason to bridge the runtime, not to drop the call.
    ///
    /// # Errors
    ///
    /// `RuntimeError` if no async runtime can be started.
    fn record_failure(&mut self) -> PyResult<()> {
        let rt = runtime().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "could not start an async runtime to record the failure",
            )
        })?;
        rt.block_on(self.inner.record_failure());
        Ok(())
    }

    /// Record a success, closing the breaker.
    fn record_success(&mut self) {
        self.inner.record_success();
    }

    #[getter]
    fn is_open(&self) -> bool {
        self.inner.is_open()
    }

    fn __repr__(&self) -> String {
        format!("CircuitBreaker(open={})", self.inner.is_open())
    }
}

/// A shared current-thread runtime for bridging the async calls underneath.
///
/// Built once. Creating one per call would make a guardrail check — which sits
/// on every request by construction — pay for runtime setup each time.
///
/// Returns `None` rather than panicking if the runtime cannot be built; library
/// code in this workspace returns errors instead of unwinding, and a binding is
/// library code for the language on the other side of it.
fn runtime() -> Option<&'static tokio::runtime::Runtime> {
    static RT: std::sync::OnceLock<Option<tokio::runtime::Runtime>> = std::sync::OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .ok()
    })
    .as_ref()
}

// ── Tool Belief Set ────────────────────────────────────────────────────

/// Multi-tool reliability tracking via Beta-Binomial beliefs.
#[pyclass(name = "ToolBeliefSet")]
#[derive(Debug)]
pub struct PyToolBeliefSet {
    inner: cynepic_bayes::ToolBeliefSet,
}

#[pymethods]
impl PyToolBeliefSet {
    #[new]
    fn new() -> Self {
        Self {
            inner: cynepic_bayes::ToolBeliefSet::new(),
        }
    }

    fn add_tool(&mut self, name: &str) {
        self.inner.register(name);
    }

    fn record_success(&mut self, name: &str) {
        self.inner.record(name, true);
    }

    fn record_failure(&mut self, name: &str) {
        self.inner.record(name, false);
    }

    fn reliability(&self, name: &str) -> PyResult<f64> {
        self.inner
            .get(name)
            .map(|t| t.reliability())
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("Tool '{}' not found", name))
            })
    }

    fn should_circuit_break(&self, name: &str, threshold: f64) -> PyResult<bool> {
        self.inner
            .get(name)
            .map(|t| t.should_circuit_break(threshold))
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("Tool '{}' not found", name))
            })
    }

    /// How many tools are registered.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        // Was hardcoded to 0, so a set with twenty tools reported none. A repr
        // that lies is worse than no repr, because it is believed.
        format!("ToolBeliefSet(tools={})", self.inner.len())
    }
}

// ── Module ─────────────────────────────────────────────────────────────

/// cynepic-rs — High-performance CARF/CYNEPIC decision intelligence from Python.
///
/// A Rust-native implementation of the core CARF architecture providing:
/// - Cynefin domain classification
/// - Causal DAG analysis (d-separation, backdoor adjustment)
/// - Bayesian conjugate priors (Beta-Binomial)
/// - Circuit breaker for policy enforcement
/// - Tool reliability tracking
#[pymodule]
fn cynepic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCynefinDomain>()?;
    m.add_class::<PyCausalDag>()?;
    m.add_class::<PyBetaBinomial>()?;
    m.add_class::<PyCircuitBreaker>()?;
    m.add_class::<PyToolBeliefSet>()?;
    m.add_class::<PyCausalEffect>()?;
    m.add_function(wrap_pyfunction!(estimate_ate, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_ate_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_att, m)?)?;

    // From the manifest, not a literal. A hand-written version drifts silently
    // and then a caller checking `cynepic.__version__` trusts the wrong thing.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__doc__",
        "cynepic-rs PyO3 bindings — complexity-adaptive decision intelligence in Rust",
    )?;

    Ok(())
}

/// Tests for the binding layer.
///
/// # Why these are Rust tests and not Python ones
///
/// Both would be worth having. These run without an interpreter or a built
/// wheel, so they gate every CI job rather than only one that has maturin
/// available, and they catch the class of defect that actually occurs here:
/// a wrapper that drops an error, hardcodes a value, or converts a type wrongly.
///
/// They are compiled out under `extension-module`, which deliberately leaves
/// the CPython symbols unresolved — a test binary has nothing to supply them.
/// The `--no-default-features` CI job is where they run.
#[cfg(all(test, not(feature = "extension-module")))]
mod tests {
    use super::*;

    #[test]
    fn a_cycle_is_refused_at_insertion() {
        let mut dag = PyCausalDag::new();
        dag.add_edge("A", "B").expect("acyclic");
        dag.add_edge("B", "C").expect("acyclic");
        // A DAG that is not acyclic invalidates every algorithm built on it, so
        // the check must be at insertion rather than at use.
        assert!(dag.add_edge("C", "A").is_err(), "a cycle was accepted");
        assert!(dag.add_edge("A", "A").is_err(), "a self-loop was accepted");
    }

    #[test]
    fn an_unknown_variable_raises_rather_than_reporting_independence() {
        let mut dag = PyCausalDag::new();
        dag.add_edge("X", "Y").expect("acyclic");
        // Finding C12: returning `True` — "conditionally independent" — for a
        // misspelled name turns a typo into a positive finding. The binding
        // must not swallow that error into a bool.
        assert!(dag.d_separated("typo", "Y", vec![]).is_err());
        assert!(dag.d_separated("X", "typo", vec![]).is_err());
        assert!(
            dag.d_separated("X", "Y", vec!["typo".into()]).is_err(),
            "an unknown name in the conditioning set must raise too"
        );
    }

    #[test]
    fn d_separation_answers_the_questions_it_can() {
        let mut dag = PyCausalDag::new();
        dag.add_edge("X", "M").expect("acyclic");
        dag.add_edge("M", "Y").expect("acyclic");
        assert!(
            !dag.d_separated("X", "Y", vec![]).expect("known names"),
            "an open chain is not d-separated"
        );
        assert!(
            dag.d_separated("X", "Y", vec!["M".into()])
                .expect("known names"),
            "conditioning on the mediator blocks the chain"
        );
    }

    #[test]
    fn identification_declines_when_it_needs_something_unobserved() {
        let mut dag = PyCausalDag::new();
        dag.add_edge("U", "T").expect("acyclic");
        dag.add_edge("U", "Y").expect("acyclic");
        dag.add_edge("T", "Y").expect("acyclic");
        dag.mark_latent("U").expect("known name");
        // Returning an adjustment set containing something the caller cannot
        // measure would be worse than declining.
        assert!(dag.find_backdoor_adjustment("T", "Y").is_err());
    }

    #[test]
    fn an_empty_adjustment_set_is_success_not_failure() {
        let mut dag = PyCausalDag::new();
        dag.add_edge("T", "Y").expect("acyclic");
        // No confounding, so nothing to adjust for. Distinct from raising,
        // which means no observed set works — and a binding that collapsed the
        // two would make the distinction unavailable from Python.
        let adj = dag
            .find_backdoor_adjustment("T", "Y")
            .expect("identifiable with no adjustment");
        assert!(adj.is_empty(), "expected an empty set, got {adj:?}");
    }

    #[test]
    fn the_repr_reports_the_graph_it_actually_holds() {
        let mut dag = PyCausalDag::new();
        dag.add_edge("A", "B").expect("acyclic");
        let r = dag.__repr__();
        assert!(r.contains("variables=2"), "{r}");
        assert!(r.contains("edges=1"), "{r}");
    }

    #[test]
    fn a_conjugate_update_moves_the_mean_in_the_right_direction() {
        let mut b = PyBetaBinomial::new();
        let start = b.mean();
        b.update(9, 1);
        assert!(b.mean() > start, "successes must raise the mean");
        let after = b.mean();
        b.update(0, 20);
        assert!(b.mean() < after, "failures must lower it");
    }

    #[test]
    fn an_invalid_prior_is_refused() {
        // Beta(0, 0) is not a distribution. Accepting it would produce a
        // posterior that is silently meaningless.
        assert!(PyBetaBinomial::with_prior(0.0, 1.0).is_err());
        assert!(PyBetaBinomial::with_prior(1.0, -1.0).is_err());
        assert!(PyBetaBinomial::with_prior(2.0, 3.0).is_ok());
    }

    #[test]
    fn the_breaker_opens_only_after_the_threshold() {
        let mut cb = PyCircuitBreaker::new(3, 30);
        assert!(!cb.is_open());
        cb.record_failure().expect("runtime available");
        cb.record_failure().expect("runtime available");
        assert!(!cb.is_open(), "opened early, before the threshold");
        cb.record_failure().expect("runtime available");
        assert!(cb.is_open(), "did not open at the threshold");
    }

    #[test]
    fn success_resets_the_breaker() {
        let mut cb = PyCircuitBreaker::new(2, 30);
        cb.record_failure().expect("runtime available");
        cb.record_success();
        cb.record_failure().expect("runtime available");
        assert!(
            !cb.is_open(),
            "a success between failures must clear the count, or a service \
             failing once an hour eventually trips for no reason"
        );
    }

    #[test]
    fn an_unregistered_tool_raises_rather_than_reporting_perfect_reliability() {
        let set = PyToolBeliefSet::new();
        // Returning 1.0 for a tool nobody registered would read as "completely
        // reliable" for something never observed at all.
        assert!(set.reliability("nope").is_err());
        assert!(set.should_circuit_break("nope", 0.5).is_err());
    }

    #[test]
    fn tool_reliability_tracks_what_it_was_told() {
        let mut set = PyToolBeliefSet::new();
        set.add_tool("search");
        for _ in 0..8 {
            set.record_success("search");
        }
        let good = set.reliability("search").expect("registered");
        for _ in 0..20 {
            set.record_failure("search");
        }
        let bad = set.reliability("search").expect("registered");
        assert!(
            bad < good,
            "failures must lower reliability: {bad} vs {good}"
        );
        assert!(
            set.should_circuit_break("search", 0.6).expect("registered"),
            "a tool failing 20 of 28 calls should trip a 0.6 threshold"
        );
    }

    #[test]
    fn the_belief_set_repr_counts_its_tools() {
        let mut set = PyToolBeliefSet::new();
        assert_eq!(set.__len__(), 0);
        set.add_tool("a");
        set.add_tool("b");
        // This was hardcoded to 0, so a set with twenty tools reported none.
        assert_eq!(set.__len__(), 2);
        assert!(set.__repr__().contains("tools=2"), "{}", set.__repr__());
    }

    #[test]
    fn every_domain_converts_and_prints() {
        for d in [
            cynepic_core::CynefinDomain::Clear,
            cynepic_core::CynefinDomain::Complicated,
            cynepic_core::CynefinDomain::Complex,
            cynepic_core::CynefinDomain::Chaotic,
            cynepic_core::CynefinDomain::Disorder,
        ] {
            let py: PyCynefinDomain = d.into();
            assert!(!py.__str__().is_empty());
            assert!(!py.__repr__().is_empty());
        }
    }
}
