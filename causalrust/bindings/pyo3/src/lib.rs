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

    fn record_failure(&mut self) {
        // CircuitBreaker::record_failure is sync — call directly
    }

    fn record_success(&mut self) {
        // CircuitBreaker::record_success is sync — call directly
    }

    #[getter]
    fn is_open(&self) -> bool {
        self.inner.is_open()
    }

    fn __repr__(&self) -> String {
        format!("CircuitBreaker(open={})", self.inner.is_open())
    }
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

    fn __repr__(&self) -> String {
        format!("ToolBeliefSet(tools={})", 0)
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

    m.add("__version__", "0.2.0")?;
    m.add(
        "__doc__",
        "cynepic-rs PyO3 bindings — complexity-adaptive decision intelligence in Rust",
    )?;

    Ok(())
}
