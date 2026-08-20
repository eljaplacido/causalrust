#!/usr/bin/env python3
"""Measure OPA sidecar policy evaluation, split into transport and engine.

Item 7 of the benchmarking plan in ``docs/roadmap.md``. This covers the
``Policy evaluation`` row of the README's table -- the row that is real but
does not measure what it appears to.

The problem with the row as written
-----------------------------------
"~100x, OPA sidecar vs in-process regorus" is two claims wearing one number:

1. **Transport.** A sidecar is an HTTP round trip. Deleting it is a real win
   and has nothing to do with which policy engine is better.
2. **Engine.** regorus (Rust) against OPA's Go evaluator, same policy, same
   input.

Quoting the combined figure credits the engine for a win that belongs to the
deployment topology. OPA can be asked for its own ``timer_rego_query_eval_ns``
via ``?metrics=true``, which separates them, so this script reports both.

Running it
----------
    curl -sSL -o /tmp/opa https://openpolicyagent.org/downloads/latest/opa_linux_arm64_static
    chmod +x /tmp/opa
    python3 scripts/compare_opa.py --opa /tmp/opa

Then, on the same machine:

    cargo run -p cynepic-guardian --example latency_report --release --features rego

Use the ``_static`` build or the matching libc build for your platform; the
download URL above is for linux/arm64.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# Kept identical to `POLICY` in
# crates/cynepic-guardian/examples/latency_report.rs. If these drift the
# comparison is meaningless, which is why the policy is short enough to diff by
# eye.
POLICY = """
package policy

default allow := false

allow if {
    input.role == "admin"
}

allow if {
    input.role == "operator"
    input.amount < 1000
}
"""

INPUT = {"role": "operator", "amount": 250}
ADDR = "127.0.0.1:8181"
URL = f"http://{ADDR}/v1/data/policy/allow?metrics=true"


def fmt(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f}ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f}us"
    return f"{ns / 1_000_000:.2f}ms"


def post(payload: dict) -> dict:
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def wait_ready(proc: subprocess.Popen, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"opa exited early with code {proc.returncode}")
        try:
            post({"input": INPUT})
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.1)
    raise RuntimeError("opa did not become ready")


def percentile(sorted_vals: list[float], p: float) -> float:
    return sorted_vals[int((len(sorted_vals) - 1) * p)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opa", required=True, help="path to the opa binary")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()

    opa = Path(args.opa)
    if not opa.exists():
        print(f"opa binary not found at {opa}", file=sys.stderr)
        return 2

    version = subprocess.run(
        [str(opa), "version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]

    with tempfile.TemporaryDirectory() as tmp:
        policy_path = Path(tmp) / "policy.rego"
        policy_path.write_text(POLICY)

        proc = subprocess.Popen(
            [str(opa), "run", "--server", "--addr", ADDR, "--log-level", "error",
             str(policy_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            wait_ready(proc)

            print(f"opa {version} — sidecar over HTTP on {ADDR}\n")

            for _ in range(200):  # warmup
                post({"input": INPUT})

            round_trip: list[float] = []
            engine_only: list[float] = []
            for _ in range(args.reps):
                t0 = time.perf_counter_ns()
                body = post({"input": INPUT})
                round_trip.append(time.perf_counter_ns() - t0)
                # OPA's own timer for evaluating the query, excluding HTTP,
                # JSON parsing of the request and response serialisation.
                engine_only.append(body["metrics"]["timer_rego_query_eval_ns"])
                assert body.get("result") is True, body

            round_trip.sort()
            engine_only.sort()

            print(f"  {'measurement':<34} {'p50':>10} {'p99':>10}")
            print(
                f"  {'round trip (what a caller waits)':<34} "
                f"{fmt(percentile(round_trip, 0.50)):>10} "
                f"{fmt(percentile(round_trip, 0.99)):>10}"
            )
            print(
                f"  {'opa timer_rego_query_eval_ns':<34} "
                f"{fmt(percentile(engine_only, 0.50)):>10} "
                f"{fmt(percentile(engine_only, 0.99)):>10}"
            )
            transport = percentile(round_trip, 0.50) - percentile(engine_only, 0.50)
            print(f"  {'└ transport + serialisation':<34} {fmt(transport):>10}")

            print(
                f"\n  {100 * percentile(engine_only, 0.50) / percentile(round_trip, 0.50):.1f}%"
                " of the round trip is OPA actually evaluating the policy."
            )
            print("  The rest is HTTP, JSON and the loopback hop.")
            print("\n  Compare against `cargo run -p cynepic-guardian --example")
            print("  latency_report --release --features rego` on this machine, and")
            print("  quote BOTH ratios: one is a claim about the policy engine, the")
            print("  other is a claim about not running a sidecar. They are")
            print("  different claims and only one of them is about this code.")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
