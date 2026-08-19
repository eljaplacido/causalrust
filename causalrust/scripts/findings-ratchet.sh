#!/usr/bin/env bash
#
# Findings ratchet.
#
# Every open correctness finding in docs/FINDINGS.md has a failing spec in
# crates/cynepic-causal/tests/findings.rs, marked `#[ignore]`. This script
# counts them and fails if the count has gone UP.
#
# Why a ratchet and not a plain test: the specs are supposed to fail. Running
# them in CI as ordinary tests would make CI permanently red, and a permanently
# red CI is one nobody reads. Counting them instead makes the debt visible,
# blocks new debt, and turns "we fixed it" into an arithmetic claim.
#
# The count may only ever go down. To lower it, delete an `#[ignore]` in the
# same pull request as the fix and lower BASELINE here. Raising BASELINE
# requires a new entry in docs/FINDINGS.md and should be argued for in review.

set -euo pipefail

cd "$(dirname "$0")/.."

# Open specs at the last ratchet. Lower this with each fix; never raise it
# without a corresponding docs/FINDINGS.md entry.
BASELINE=13

SPEC_FILE="crates/cynepic-causal/tests/findings.rs"

if [[ ! -f "$SPEC_FILE" ]]; then
    echo "findings-ratchet: $SPEC_FILE is missing." >&2
    echo "The spec file is the ledger. Deleting it does not close the findings." >&2
    exit 1
fi

# Count `#[ignore]` attributes on test functions. Restricted to line starts so
# the doc comment at the top of the file — which discusses `#[ignore]` — is not
# counted.
open=$(grep -c '^#\[ignore' "$SPEC_FILE" || true)

echo "findings-ratchet: ${open} open specs (baseline ${BASELINE})"

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
scripts/findings-ratchet.sh to ${open} to lock the improvement in, and mark
the finding closed in docs/FINDINGS.md.

EOF
    exit 1
fi

# The specs must still COMPILE against the current API even though they are
# ignored. Without this, an API change could silently orphan every spec and the
# count above would keep reporting a reassuring number about dead code.
echo "findings-ratchet: checking specs still compile against the current API"
cargo test -p cynepic-causal --test findings --no-run --quiet

# Every #[ignore]d spec must still FAIL. A spec that has started passing is
# either fixed — in which case delete the #[ignore] and claim the win — or it
# never witnessed its finding in the first place. Both are things to know, and
# neither should sit undetected behind an `#[ignore]` that makes it look like
# outstanding work.
#
# This is the check that keeps the ledger honest. Counting alone would let a
# file full of vacuous specs report a large, reassuring number.
echo "findings-ratchet: confirming every open spec still fails"
ignored_out=$(cargo test -p cynepic-causal --test findings -- --ignored 2>&1 || true)
passing=$(printf '%s\n' "$ignored_out" | grep -cE '^test .* \.\.\. ok$' || true)

if (( passing > 0 )); then
    cat >&2 <<EOF

FAIL: ${passing} spec(s) marked #[ignore] now PASS.

EOF
    printf '%s\n' "$ignored_out" | grep -E '^test .* \.\.\. ok$' >&2
    cat >&2 <<'EOF'

A passing spec behind an #[ignore] is one of two things:

  - The finding is fixed. Delete the #[ignore], lower BASELINE, and mark it
    closed in docs/FINDINGS.md.
  - The spec never witnessed the defect. Sharpen it until it fails, or move it
    to the always-run suite as a regression guard and say what it actually
    guards.

Do not leave it as it is. It currently reads as outstanding work that nobody
is doing, which is the one thing this ledger exists to prevent.
EOF
    exit 1
fi

# The always-run half must be green: those are the measured-sound behaviours
# (OLS coverage, the metamorphic relations) and they are regression guards now.
echo "findings-ratchet: checking the regression guards still hold"
cargo test -p cynepic-causal --test findings --quiet

echo "findings-ratchet: ok — ${open} open, all failing as expected"
