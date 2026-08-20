#!/usr/bin/env python3
"""Measure LangGraph per-step dispatch latency, for comparison against cynepic-graph.

Item 7 of the benchmarking plan in ``docs/roadmap.md``. This covers the
``StateGraph step`` row of the README's table -- the one row of the four that
was well-posed as written.

Why it is worth running
-----------------------
The README assumed ``~10x``. Measured on the same workload, on one machine:

    nodes    langgraph/step    cynepic/step    speedup
        5            66.0us           550ns       120x
       20            57.5us           572ns       100x
      100            73.0us           392ns       186x

Roughly **100-186x**, so this row was *understated* by an order of magnitude --
the opposite direction from the NetworkX row, which was overstated by 50x. Two
assumptions, both plausible, both wrong, in opposite directions. That is what
un-instrumented figures look like in aggregate: not biased, just noise dressed
as a claim.

What makes the comparison fair
------------------------------
Both sides build a chain of trivial nodes, compile once, then invoke many
times. Neither side does user work inside a node, so what remains is the
framework's own dispatch overhead -- the thing the row claims to compare.

Three ways this could have been rigged, and how each is avoided:

* Building the graph inside the timed region. ``compile()`` does real work;
  calling it per iteration would report setup as per-step cost. Both sides
  build once.
* Nodes that do something. Real work inside a node dilutes the difference
  toward 1x. Both sides increment an integer.
* Quoting per-graph rather than per-step. A 100-node graph makes any framework
  look slow in absolute terms; one step is the comparable unit.

What it does not say
--------------------
LangGraph carries checkpointing middleware, a channel-based reducer model and
interrupt support through every step. Some of the measured gap is that
machinery rather than waste, and a dispatch-speed win is not a claim that one
library replaces the other. Note also that LangGraph's per-step cost is roughly
flat in ``n`` while cynepic-graph's *falls* -- ours amortises fixed setup over
more steps, theirs is genuinely per-step.

Running it
----------
    python3 -m venv /tmp/lgbench && /tmp/lgbench/bin/pip install langgraph
    /tmp/lgbench/bin/python scripts/compare_langgraph.py

Then, on the same machine:

    cargo run -p cynepic-graph --example latency_report --release

Comparing numbers from different machines is meaningless, so run both or
neither.
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import sys
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

# Matches `crates/cynepic-graph/examples/latency_report.rs`.
SIZES = ((5, 400), (20, 300), (100, 150))


class Counter(TypedDict):
    counter: int


def make_node(i: int):
    def step(state: Counter) -> Counter:
        return {"counter": state["counter"] + 1}

    step.__name__ = f"n{i}"
    return step


def build(n_nodes: int):
    g = StateGraph(Counter)
    for i in range(n_nodes):
        g.add_node(f"n{i}", make_node(i))
    g.add_edge(START, "n0")
    for i in range(n_nodes - 1):
        g.add_edge(f"n{i}", f"n{i + 1}")
    g.add_edge(f"n{n_nodes - 1}", END)
    return g.compile()


def fmt(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f}ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f}us"
    return f"{ns / 1_000_000:.2f}ms"


def main() -> int:
    print("langgraph — linear chain of incrementing nodes, compiled once\n")
    print(f"  python     {sys.version.split()[0]}")
    print(f"  langgraph  {md.version('langgraph')}")
    print(f"  machine    {platform.machine()} / {platform.processor() or 'unknown'}\n")

    print(f"  {'nodes':>6} {'graph p50':>11} {'graph p99':>11} {'per step p50':>14}")

    for nodes, reps in SIZES:
        app = build(nodes)

        for _ in range(5):  # warmup
            app.invoke({"counter": 0})

        times = []
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            out = app.invoke({"counter": 0})
            times.append(time.perf_counter_ns() - t0)
        # The assertion is the point of the fixture: a framework that skipped
        # nodes would look wonderfully fast.
        assert out["counter"] == nodes, f"expected {nodes} steps, ran {out['counter']}"
        times.sort()

        p50 = times[int((len(times) - 1) * 0.50)]
        p99 = times[int((len(times) - 1) * 0.99)]
        print(f"  {nodes:>6} {fmt(p50):>11} {fmt(p99):>11} {fmt(p50 / nodes):>14}")

    print("\nCompare against `cargo run -p cynepic-graph --example latency_report")
    print("--release` on this same machine. A cross-machine ratio is not a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
