#!/usr/bin/env python3
"""Generate golden reference fixtures for the cross-implementation parity tests.

Why this exists
---------------
The README carries assumed speedups over NetworkX, PyMC and friends. A speedup
claim is only interesting if both sides compute the *same thing*, and nothing
checked that. Before "1000x faster than NetworkX" means anything, "identical to
NetworkX" has to be true.

This script computes reference answers with the established Python
implementations and writes them to
`crates/cynepic-causal/tests/fixtures/parity.json`. The Rust
parity suite then asserts agreement against that file.

The references are committed, so CI needs no Python. Regenerate only when adding
cases -- and if a regenerated value *changes* for an existing case, that is a
finding to investigate, not a fixture to overwrite.

Usage
-----
    python3 scripts/generate_parity_fixtures.py

Requires numpy, scipy and networkx. Versions used are recorded in the output so
a future disagreement can be attributed.
"""

from __future__ import annotations

import json
import pathlib
import sys

import networkx as nx
import numpy as np
import scipy.special
import scipy.stats

# Lives with the suite that consumes it: crates/cynepic-causal/tests/parity.rs.
OUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "crates" / "cynepic-causal" / "tests" / "fixtures" / "parity.json"
)

# Fixed seed: the fixtures must be byte-reproducible from this script alone.
SEED = 20260820


def ols_cases() -> list[dict]:
    """Least-squares treatment effects, referenced against `numpy.linalg.lstsq`.

    The estimand is the coefficient on treatment in `Y ~ 1 + T + X`, which is
    what `LinearATEEstimator::ols_adjusted` returns. numpy solves the same
    system by SVD where the Rust side uses pivoted Householder QR, so agreement
    is evidence about the numerics rather than a restatement of one algorithm.
    """
    rng = np.random.default_rng(SEED)
    cases = []

    for label, n, p, noise in [
        ("small", 40, 2, 1.0),
        ("medium", 400, 3, 1.0),
        ("wide", 200, 12, 0.5),
        ("noisy", 300, 2, 5.0),
        ("near-collinear", 150, 3, 1.0),
    ]:
        x = rng.normal(size=(n, p))
        if label == "near-collinear":
            # Third column almost, but not exactly, the sum of the first two.
            # Exactly collinear would be rejected by both sides; *nearly* so is
            # where a solver's conditioning shows.
            x[:, 2] = x[:, 0] + x[:, 1] + 1e-6 * rng.normal(size=n)

        t = (rng.random(n) < 0.5).astype(float)
        beta_x = rng.normal(size=p)
        y = 1.0 + 2.5 * t + x @ beta_x + noise * rng.normal(size=n)

        design = np.column_stack([np.ones(n), t, x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)

        # Classical OLS standard error on the treatment coefficient.
        residual = y - design @ coef
        dof = n - design.shape[1]
        sigma2 = float(residual @ residual) / dof
        xtx_inv = np.linalg.inv(design.T @ design)
        se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))

        cases.append(
            {
                "label": label,
                "treatment": t.tolist(),
                "outcome": y.tolist(),
                "covariates": x.tolist(),
                "expected_ate": float(coef[1]),
                "expected_se_classical": se,
            }
        )
    return cases


def difference_in_means_cases() -> list[dict]:
    """Welch two-sample difference, referenced against `scipy.stats.ttest_ind`.

    scipy computes the Welch standard error internally; recovering it from the
    reported statistic checks the Rust side against an independent derivation
    rather than against the same formula written twice.
    """
    rng = np.random.default_rng(SEED + 1)
    cases = []
    for label, n_t, n_c, shift, sd_t, sd_c in [
        ("balanced", 100, 100, 2.0, 1.0, 1.0),
        ("unbalanced", 30, 300, 1.5, 1.0, 1.0),
        ("unequal-variance", 120, 120, 1.0, 3.0, 0.5),
    ]:
        y_t = rng.normal(shift, sd_t, n_t)
        y_c = rng.normal(0.0, sd_c, n_c)

        treatment = np.concatenate([np.ones(n_t), np.zeros(n_c)])
        outcome = np.concatenate([y_t, y_c])

        diff = float(y_t.mean() - y_c.mean())
        result = scipy.stats.ttest_ind(y_t, y_c, equal_var=False)
        se = abs(diff / float(result.statistic))

        cases.append(
            {
                "label": label,
                "treatment": treatment.tolist(),
                "outcome": outcome.tolist(),
                "expected_ate": diff,
                "expected_se_welch": se,
            }
        )
    return cases


def quantile_cases() -> list[dict]:
    """Beta, Gamma and Student-t quantiles, referenced against scipy.

    `cynepic-core::special` implements these with a Lentz continued fraction and
    bisection; scipy uses Boost's incomplete beta and gamma. Different
    algorithms, so agreement is meaningful.
    """
    beta = [
        {
            "kind": "beta",
            "a": a,
            "b": b,
            "p": p,
            "expected": float(scipy.stats.beta.ppf(p, a, b)),
        }
        # Includes the shapes a reliability tracker actually reaches: a perfect
        # record (201, 1), a single observation (1, 2), and the symmetric case.
        for a, b in [(1, 1), (3, 1), (1, 3), (2, 5), (10, 10), (201, 1), (1, 201), (0.5, 0.5)]
        for p in [0.005, 0.025, 0.25, 0.5, 0.75, 0.975, 0.995]
    ]
    gamma = [
        {
            "kind": "gamma",
            "shape": shape,
            "rate": rate,
            "p": p,
            "expected": float(scipy.stats.gamma.ppf(p, shape, scale=1.0 / rate)),
        }
        for shape, rate in [(0.5, 1.0), (1.0, 1.0), (2.0, 3.0), (20.0, 2.0), (100.0, 10.0)]
        for p in [0.025, 0.5, 0.975]
    ]
    student_t = [
        {
            "kind": "t",
            "dof": dof,
            "p": p,
            "expected": float(scipy.stats.t.ppf(p, dof)),
        }
        # Low dof matters: finding C14's Satterthwaite correction lands around 7.
        for dof in [1, 2, 5, 7, 10, 23, 30, 100, 1000]
        for p in [0.025, 0.05, 0.5, 0.95, 0.975]
    ]
    return beta + gamma + student_t


def dsep_cases() -> list[dict]:
    """D-separation verdicts, referenced against `networkx.is_d_separator`.

    The Rust side runs Bayes-Ball; networkx uses a different formulation. The
    collider cases are the ones worth having: conditioning on a collider *opens*
    a path, which is the direction that surprises people and the direction an
    implementation is most likely to get backwards.
    """
    graphs = {
        "chain": [("X", "W"), ("W", "Y")],
        "fork": [("W", "X"), ("W", "Y")],
        "collider": [("X", "W"), ("Y", "W")],
        "collider-with-descendant": [("X", "W"), ("Y", "W"), ("W", "D")],
        "diamond": [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        "long-chain": [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")],
        "m-structure": [("U1", "X"), ("U1", "M"), ("U2", "M"), ("U2", "Y")],
    }

    cases = []
    for name, edges in graphs.items():
        g = nx.DiGraph(edges)
        nodes = sorted(g.nodes())
        for x in nodes:
            for y in nodes:
                if x >= y:
                    continue
                for z in ([], *[[n] for n in nodes if n not in (x, y)]):
                    try:
                        sep = bool(nx.is_d_separator(g, {x}, {y}, set(z)))
                    except Exception:  # noqa: BLE001 - networkx raises on adjacency
                        continue
                    cases.append(
                        {
                            "graph": name,
                            "edges": edges,
                            "x": x,
                            "y": y,
                            "z": z,
                            "expected_d_separated": sep,
                        }
                    )
    return cases


def main() -> int:
    payload = {
        "_comment": (
            "GENERATED by scripts/generate_parity_fixtures.py -- do not edit by "
            "hand. If a regenerated value differs from what is committed here, "
            "that is a finding to investigate, not a fixture to overwrite."
        ),
        "_generated_with": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "networkx": nx.__version__,
            "seed": SEED,
        },
        "ols": ols_cases(),
        "difference_in_means": difference_in_means_cases(),
        "quantiles": quantile_cases(),
        "d_separation": dsep_cases(),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"wrote {OUT.relative_to(pathlib.Path.cwd())}")
    for key in ("ols", "difference_in_means", "quantiles", "d_separation"):
        print(f"  {key:22} {len(payload[key])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
