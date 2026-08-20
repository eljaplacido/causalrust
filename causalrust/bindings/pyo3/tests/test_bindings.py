"""Smoke tests for the built wheel.

Run against an installed ``cynepic``, not against the Rust source::

    python3 -m venv .venv && . .venv/bin/activate
    pip install maturin pytest
    maturin develop --features extension-module
    pytest bindings/pyo3/tests

Why these exist alongside the Rust tests
----------------------------------------
``bindings/pyo3/src/lib.rs`` has 14 Rust unit tests covering the same methods,
and they are the ones that gate every CI job because they need no interpreter.
These test something the Rust tests structurally cannot: **the artifact that
actually ships**.

Between a passing Rust test and a working ``import cynepic`` sit the module
registration, the ``abi3`` build, the class exports, and the getter/method
distinction that only exists on the Python side. A ``#[getter]`` that should
have been a method is invisible to Rust and is an ``AttributeError`` here.

What this file will not do
--------------------------
Re-test the statistics. Whether ``ols_adjusted`` covers at 95% is settled by a
Monte Carlo harness in ``cynepic-testkit``, and asserting a treatment effect to
two decimal places here would duplicate that badly while pretending to add
confidence. These check that the bridge carries values and errors across
intact — nothing more, and nothing that would pass if it did not.
"""

from __future__ import annotations

import pytest

cynepic = pytest.importorskip(
    "cynepic",
    reason="build the wheel first: maturin develop --features extension-module",
)


# ── The module itself ───────────────────────────────────────────────────


def test_version_is_present_and_not_a_placeholder():
    # Read from CARGO_PKG_VERSION on the Rust side. A hardcoded literal drifted
    # from the manifest once already.
    assert cynepic.__version__
    assert cynepic.__version__ != "0.0.0"
    assert cynepic.__version__.count(".") == 2, cynepic.__version__


def test_every_advertised_class_is_exported():
    # A class registered in the module macro but missing from the built wheel is
    # an ImportError for a user and invisible to every Rust test.
    for name in (
        "CynefinDomain",
        "CausalDag",
        "BetaBinomial",
        "CircuitBreaker",
        "ToolBeliefSet",
    ):
        assert hasattr(cynepic, name), f"{name} is not exported"


# ── Causal DAG ──────────────────────────────────────────────────────────


def test_a_cycle_raises_valueerror():
    dag = cynepic.CausalDag()
    dag.add_edge("A", "B")
    dag.add_edge("B", "C")
    with pytest.raises(ValueError):
        dag.add_edge("C", "A")


def test_an_unknown_variable_raises_rather_than_returning_true():
    # Finding C12, through the binding: returning True — "conditionally
    # independent" — for a misspelled name turns a typo into a positive finding.
    dag = cynepic.CausalDag()
    dag.add_edge("X", "Y")
    with pytest.raises(ValueError):
        dag.d_separated("typo", "Y", [])
    with pytest.raises(ValueError):
        dag.d_separated("X", "Y", ["typo"])


def test_d_separation_answers_what_it_can():
    dag = cynepic.CausalDag()
    dag.add_edge("X", "M")
    dag.add_edge("M", "Y")
    assert dag.d_separated("X", "Y", []) is False
    assert dag.d_separated("X", "Y", ["M"]) is True


def test_identification_declines_when_it_needs_something_latent():
    dag = cynepic.CausalDag()
    dag.add_edge("U", "T")
    dag.add_edge("U", "Y")
    dag.add_edge("T", "Y")
    dag.mark_latent("U")
    # Returning an adjustment set containing something unmeasurable would be
    # worse than declining.
    with pytest.raises(ValueError):
        dag.find_backdoor_adjustment("T", "Y")


def test_an_empty_adjustment_set_is_success_not_failure():
    dag = cynepic.CausalDag()
    dag.add_edge("T", "Y")
    # Distinct from raising, which means no observed set works. A binding that
    # collapsed the two would make the distinction unavailable from Python.
    assert dag.find_backdoor_adjustment("T", "Y") == []


# ── Bayes ───────────────────────────────────────────────────────────────


def test_a_conjugate_update_moves_the_mean():
    # `mean` is a property, not a method — it is `#[getter]` on the Rust side.
    # The Rust unit tests call `b.mean()` and pass, because in Rust it *is* a
    # method; the getter/method distinction exists only here. This is the
    # concrete reason this file exists.
    b = cynepic.BetaBinomial()
    start = b.mean
    b.update(9, 1)
    assert b.mean > start
    after = b.mean
    b.update(0, 20)
    assert b.mean < after


def test_the_documented_properties_are_properties():
    # Pin the shape of the API, not just its values. Flipping a `#[getter]` to a
    # method or back is a silent breaking change for every Python caller and is
    # invisible to every Rust test.
    b = cynepic.BetaBinomial()
    assert isinstance(b.mean, float), "BetaBinomial.mean must be a property"

    cb = cynepic.CircuitBreaker(2, 30)
    assert isinstance(cb.is_open, bool), "CircuitBreaker.is_open must be a property"


def test_an_invalid_prior_raises():
    with pytest.raises(ValueError):
        cynepic.BetaBinomial.with_prior(0.0, 1.0)


# ── Guardian ────────────────────────────────────────────────────────────


def test_the_circuit_breaker_trips_at_its_threshold():
    # The regression test for a real defect: `record_failure` had an empty body,
    # so the Python breaker recorded nothing and `is_open` was always False —
    # a guardrail that could not trip. No Rust test existed to catch it, and no
    # Python user could have seen why.
    cb = cynepic.CircuitBreaker(3, 30)
    assert cb.is_open is False
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open is False, "opened before its threshold"
    cb.record_failure()
    assert cb.is_open is True, "did not open at its threshold"


def test_success_clears_the_failure_count():
    cb = cynepic.CircuitBreaker(2, 30)
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    assert cb.is_open is False


# ── Tool beliefs ────────────────────────────────────────────────────────


def test_an_unregistered_tool_raises_rather_than_reporting_reliability():
    # Returning 1.0 for something never observed reads as "completely reliable".
    tools = cynepic.ToolBeliefSet()
    with pytest.raises(KeyError):
        tools.reliability("nope")


def test_reliability_tracks_what_it_was_told():
    tools = cynepic.ToolBeliefSet()
    tools.add_tool("search")
    for _ in range(8):
        tools.record_success("search")
    good = tools.reliability("search")
    for _ in range(20):
        tools.record_failure("search")
    assert tools.reliability("search") < good
    assert tools.should_circuit_break("search", 0.6) is True


def test_the_repr_counts_the_tools_it_holds():
    # Was hardcoded to 0, so a set with twenty tools reported none.
    tools = cynepic.ToolBeliefSet()
    tools.add_tool("a")
    tools.add_tool("b")
    assert len(tools) == 2
    assert "tools=2" in repr(tools)
