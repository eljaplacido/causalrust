#!/usr/bin/env bash
#
# Findings ratchet — workspace-wide.
#
# Every open correctness finding has a failing spec, marked `#[ignore]`, in one
# of the suites listed below. This script counts them and fails if the count has
# gone UP.
#
# Why a ratchet and not a plain test: the specs are SUPPOSED to fail. Running
# them as ordinary tests would make CI permanently red, and a permanently red CI
# is one nobody reads. Counting them instead makes the debt visible, blocks new
# debt, and turns "we fixed it" into an arithmetic claim.
#
# Three invariants are enforced:
#
#   1. The open count may only go down.
#   2. The specs still COMPILE against the current API — otherwise an API change
#      could orphan the lot while the count kept reporting a reassuring number
#      about dead code.
#   3. Every open spec still FAILS. A spec that has started passing is either a
#      fix nobody claimed or a spec that witnesses nothing, and both look
#      identical to outstanding work until something checks. This is the check
#      that caught four mislabelled specs during the first re-baseline.

set -euo pipefail

cd "$(dirname "$0")/.."

# Open specs at the last ratchet. Lower with each fix; never raise without a
# corresponding docs/FINDINGS.md entry in the same commit.
BASELINE=4

# Suites carrying findings specs: "<crate>:<test target>".
SUITES=(
    "cynepic-causal:findings"
    "cynepic-router:routing_accuracy"
    "cynepic-bayes:calibration"
    "cynepic-guardian:guardrails"
    "cynepic-graph:execution_properties"
)

open=0
missing=0

for suite in "${SUITES[@]}"; do
    crate="${suite%%:*}"
    target="${suite##*:}"
    file="crates/${crate}/tests/${target}.rs"

    if [[ ! -f "$file" ]]; then
        echo "findings-ratchet: $file is missing." >&2
        echo "The spec files are the ledger. Deleting one does not close its findings." >&2
        missing=1
        continue
    fi

    # Count `#[ignore]` attributes at line starts only, so the doc comments that
    # discuss `#[ignore]` are not counted.
    count=$(grep -c '^#\[ignore' "$file" || true)
    printf 'findings-ratchet: %-30s %s open\n' "${crate}/${target}" "$count"
    open=$((open + count))
done

if (( missing == 1 )); then
    exit 1
fi

echo "findings-ratchet: ${open} open specs total (baseline ${BASELINE})"

if (( open > BASELINE )); then
    cat >&2 <<EOF

FAIL: open findings went up, ${BASELINE} -> ${open}.

A new #[ignore]d spec means a new confirmed defect. That is allowed, but it
must be deliberate:

  1. Add the finding to docs/FINDINGS.md with its failure mode and fix.
  2. Raise BASELINE in this script, in the same commit.

If you were fixing something and the count rose, the fix introduced a
regression somewhere else.
EOF
    exit 1
fi

if (( open < BASELINE )); then
    cat <<EOF

Open findings went DOWN, ${BASELINE} -> ${open}. Lower BASELINE in
scripts/findings-ratchet.sh to ${open} to lock the improvement in, and mark the
finding closed in docs/FINDINGS.md.

EOF
    exit 1
fi

for suite in "${SUITES[@]}"; do
    crate="${suite%%:*}"
    target="${suite##*:}"

    echo "findings-ratchet: ${crate}/${target} — checking specs compile"
    cargo test -p "$crate" --test "$target" --no-run --quiet

    echo "findings-ratchet: ${crate}/${target} — confirming open specs still fail"
    ignored_out=$(cargo test -p "$crate" --test "$target" -- --ignored 2>&1 || true)
    passing=$(printf '%s\n' "$ignored_out" | grep -cE '^test .* \.\.\. ok$' || true)

    if (( passing > 0 )); then
        cat >&2 <<EOF

FAIL: ${passing} spec(s) in ${crate}/${target} marked #[ignore] now PASS.

EOF
        printf '%s\n' "$ignored_out" | grep -E '^test .* \.\.\. ok$' >&2
        cat >&2 <<'EOF'

A passing spec behind an #[ignore] is one of two things:

  - The finding is fixed. Delete the #[ignore], lower BASELINE, and mark it
    closed in docs/FINDINGS.md.
  - The spec never witnessed the defect. Sharpen it until it fails, or move it
    to the always-run suite as a regression guard and say what it guards.

Do not leave it as it is. It currently reads as outstanding work that nobody is
doing, which is the one thing this ledger exists to prevent.
EOF
        exit 1
    fi

    echo "findings-ratchet: ${crate}/${target} — checking the regression guards hold"
    cargo test -p "$crate" --test "$target" --quiet
done

echo "findings-ratchet: ok — ${open} open, all failing as expected"
