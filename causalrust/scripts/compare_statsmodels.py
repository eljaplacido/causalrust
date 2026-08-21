#!/usr/bin/env python3
"""Measure statsmodels and numpy on the estimation tasks cynepic-causal performs.

This is the comparison a data scientist actually weighs. NetworkX and LangGraph
answer "is the Rust faster"; this one answers "is it faster than the thing I
already have open", which is a different and more demanding question, because
statsmodels' OLS is LAPACK underneath and LAPACK is not slow.

Running it
----------
    python3 -m venv /tmp/smbench
    /tmp/smbench/bin/pip install numpy statsmodels
    /tmp/smbench/bin/python scripts/compare_statsmodels.py

Then, on the same machine:

    cargo run -p cynepic-causal --example causal_latency --release

What is compared
----------------
Like for like, which takes some care:

* ``ols_adjusted`` regresses the outcome on treatment plus covariates and
  reports the treatment coefficient with a robust standard error. The
  statsmodels equivalent is ``OLS(...).fit(cov_type="HC1")`` — NOT the default
  ``fit()``, which computes classical errors and less work. Comparing against
  the cheaper call would flatter the Rust side for free.
* ``ipw`` fits a logistic propensity model and forms a Hájek weighted contrast.
  The statsmodels equivalent is ``Logit(...).fit()`` plus the weighting, so the
  timing includes both.
* ``difference_in_means`` is a two-sample contrast with an unequal-variance
  standard error, i.e. ``scipy.stats.ttest_ind(equal_var=False)``.

Model construction is inside the timed region on both sides, because a caller
pays it per estimate. Data generation is outside it on both sides.

What this does NOT claim
------------------------
That the answers are the same. That is a separate and more important question,
and it is settled by ``crates/cynepic-causal/tests/parity.rs``, which pins the
Rust output against numpy and scipy across 258 cases. Speed is only interesting
once agreement is established.
"""

from __future__ import annotations

import platform
import sys
import time

import numpy as np
import statsmodels
import statsmodels.api as sm
from scipy import stats

SEED = 20260821
SIZES = ((1_000, 3), (10_000, 3), (10_000, 25), (100_000, 3))


def make(n: int, p: int, rng: np.random.Generator):
    """Confounded observational data with a known effect of 2.0."""
    x = rng.normal(size=(n, p))
    logit = 0.8 * x[:, 0]
    e = 1.0 / (1.0 + np.exp(-logit))
    t = (rng.uniform(size=n) < e).astype(float)
    y = 2.0 * t + 0.75 * x[:, 0] + rng.normal(size=n)
    return t, y, x


def fmt(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f}ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f}us"
    return f"{ns / 1_000_000:.2f}ms"


def timeit(fn, reps: int) -> tuple[float, float]:
    for _ in range(max(3, reps // 10)):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
    times.sort()
    return times[len(times) // 2], times[int((len(times) - 1) * 0.99)]


def ols_hc1(t, y, x):
    """Treatment coefficient with a heteroskedasticity-robust standard error."""
    design = sm.add_constant(np.column_stack([t, x]))
    fit = sm.OLS(y, design).fit(cov_type="HC1")
    return fit.params[1], fit.bse[1]


def ipw(t, y, x):
    """Logistic propensity, then a Hajek weighted contrast."""
    design = sm.add_constant(x)
    scores = sm.Logit(t, design).fit(disp=0).predict(design)
    scores = np.clip(scores, 0.02, 0.98)
    w1 = t / scores
    w0 = (1.0 - t) / (1.0 - scores)
    return (w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum()


def diff_in_means(t, y):
    a, b = y[t > 0.5], y[t <= 0.5]
    res = stats.ttest_ind(a, b, equal_var=False)
    return a.mean() - b.mean(), res


def main() -> int:
    print("statsmodels / numpy — the same estimation tasks cynepic-causal performs\n")
    print(f"  python       {sys.version.split()[0]}")
    print(f"  numpy        {np.__version__}")
    print(f"  statsmodels  {statsmodels.__version__}")
    print(f"  machine      {platform.machine()} / {platform.processor() or 'unknown'}\n")

    rng = np.random.default_rng(SEED)

    print(f"  {'task':<34} {'p50':>10} {'p99':>10}")

    for n, p in SIZES:
        t, y, x = make(n, p, rng)
        reps = 300 if n <= 10_000 else 40

        p50, p99 = timeit(lambda: ols_hc1(t, y, x), reps)
        print(f"  {f'ols + HC1, n={n} p={p}':<34} {fmt(p50):>10} {fmt(p99):>10}")

    print()
    for n, p in ((1_000, 3), (10_000, 3)):
        t, y, x = make(n, p, rng)
        reps = 100 if n <= 1_000 else 30
        p50, p99 = timeit(lambda: ipw(t, y, x), reps)
        print(f"  {f'logit + IPW, n={n} p={p}':<34} {fmt(p50):>10} {fmt(p99):>10}")

    print()
    for n in (1_000, 100_000):
        t, y, _ = make(n, 3, rng)
        reps = 2_000 if n <= 1_000 else 200
        p50, p99 = timeit(lambda: diff_in_means(t, y), reps)
        print(f"  {f'welch difference, n={n}':<34} {fmt(p50):>10} {fmt(p99):>10}")

    # Sanity: the estimates must be near the truth, or the timings are timing
    # the wrong thing. A benchmark that measures a broken call is worse than no
    # benchmark, because it looks like evidence.
    t, y, x = make(10_000, 3, rng)
    ate_ols, _ = ols_hc1(t, y, x)
    ate_ipw = ipw(t, y, x)
    print(f"\n  sanity — true effect 2.0, ols {ate_ols:.3f}, ipw {ate_ipw:.3f}")
    assert abs(ate_ols - 2.0) < 0.15, ate_ols
    assert abs(ate_ipw - 2.0) < 0.25, ate_ipw

    print("\n  Compare against `cargo run -p cynepic-causal --example causal_latency")
    print("  --release` on this same machine. A cross-machine ratio is not a result.")
    print("\n  Note what is NOT claimed here: that the answers agree. That is a")
    print("  separate and more important question, settled by tests/parity.rs")
    print("  across 258 cases. Speed only becomes interesting after agreement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
