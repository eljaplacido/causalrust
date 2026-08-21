#!/usr/bin/env python3
"""Measure networkx d-separation latency, for comparison against cynepic-causal.

Item 7 of the benchmarking plan in ``docs/roadmap.md``: throughput, deliberately
last. This covers one row of the README's assumed-speedup table.

Why it is worth running
-----------------------
The README asserted ``~1,000x`` for d-separation against a ``~10ms`` NetworkX
baseline. Both halves were wrong, and in the same direction: NetworkX is much
faster than 10ms, so the ratio was inflated at both ends.

Measured on the same chain graphs the Rust harness uses:

    nodes    networkx      cynepic     speedup
       10      12.1us       0.62us         19x
      100      49.3us        4.6us         11x
      500     213.2us       24.2us          9x

Roughly **9-19x**, not 1000x, and it *falls* as the graph grows -- which says
most of the win is per-call Python overhead rather than the algorithm. That is
still a good result. It is not the advertised one.

Running it
----------
    python3 scripts/compare_networkx.py

Then, on the same machine:

    cargo run -p cynepic-causal --example causal_latency --release

Comparing numbers from different machines is meaningless, so run both or
neither. The CPU is printed by the Rust side; record it with any figure you
quote.
"""

from __future__ import annotations

import platform
import time

import networkx as nx

# Matches `causal_latency.rs`. A chain with the first half of the nodes in the
# conditioning set: long paths, and enough conditioning to make the traversal do
# real work.
SIZES = (10, 100, 500)


def percentile(sorted_ns: list[int], p: float) -> int:
    return sorted_ns[int((len(sorted_ns) - 1) * p)]


def fmt(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f}ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f}us"
    return f"{ns / 1_000_000:.2f}ms"


def main() -> int:
    print("networkx.is_d_separator — chain graphs, first half conditioned\n")
    print(f"  python     {platform.python_version()}")
    print(f"  networkx   {nx.__version__}")
    print(f"  machine    {platform.machine()} / {platform.processor() or 'unknown'}\n")

    print(f"  {'nodes':>6} {'p50':>10} {'p95':>10} {'p99':>10} {'samples':>9}")

    for nodes in SIZES:
        g = nx.DiGraph()
        for i in range(nodes - 1):
            g.add_edge(f"X{i}", f"X{i + 1}")
        last = f"X{nodes - 1}"
        # Endpoints excluded: networkx requires the three sets be disjoint and
        # raises otherwise. cynepic-causal did answer that query -- with `true`,
        # a positive finding of independence for a malformed question. Writing
        # this comparison is what surfaced it; it now rejects the query too, so
        # both sides are timing the same work.
        z = {f"X{i}" for i in range(1, nodes // 2)}

        for _ in range(50):  # warmup
            nx.is_d_separator(g, {"X0"}, {last}, z)

        reps = 2000 if nodes <= 100 else 300
        times = []
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            nx.is_d_separator(g, {"X0"}, {last}, z)
            times.append(time.perf_counter_ns() - t0)
        times.sort()

        print(
            f"  {nodes:>6} {fmt(percentile(times, 0.50)):>10} "
            f"{fmt(percentile(times, 0.95)):>10} "
            f"{fmt(percentile(times, 0.99)):>10} {reps:>9}"
        )

    print("\nCompare against `cargo run -p cynepic-causal --example causal_latency")
    print("--release` on this same machine. A cross-machine ratio is not a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
