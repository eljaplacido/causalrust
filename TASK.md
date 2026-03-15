# Current Task

## Phase 1: Critical Fixes (README code examples + exports)
Fix root README.md code examples and missing re-exports so the landing page doesn't embarrass us.

### Subtasks
1. [x] Audit complete — all issues documented in STATE.md
2. [ ] Fix cynepic-graph lib.rs: re-export FnNode
3. [ ] Fix root README.md: remove .unwrap() from add_edge calls
4. [ ] Fix root README.md: LoopViolation::NodeOvervisit → NodeOvervisited
5. [ ] Fix root README.md: CircuitBreaker record_failure is async — add .await
6. [ ] Fix test counts: guardian=22, total=99 across all docs
7. [ ] Remove phantom deps: axum from router, polars/async-openai from workspace
8. [ ] Fix PITCH.md stale status
9. [ ] Fix integration.md false Polars claim
10. [ ] Fix architecture.md incomplete file trees
11. [ ] Fix CLAUDE.md repository layout + test count
12. [ ] Update .gitignore
13. [ ] Security fixes: CausalDag deser, CircuitBreaker ordering, NormalNormal empty, sampler panics
14. [ ] Fix inner README.md (d_separated signature, is_closed→is_open)
15. [ ] Fix EXPERIMENTS.md (major rewrite needed)
