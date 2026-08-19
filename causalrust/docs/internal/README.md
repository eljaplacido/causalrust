# Internal working notes

Historical planning documents, kept for provenance. **They are not current and
should not be read as guidance.**

They were moved here from the repository root ahead of the open-source release,
where a newcomer would land on them first. `TASK.md` in particular lists fifteen
unchecked subtasks that were all completed months earlier — read as a to-do list
it is simply wrong.

| File | What it was | Superseded by |
|---|---|---|
| `TASK.md` | Phase 1 fix checklist, March 2026 | [CHANGELOG.md](../../../CHANGELOG.md) |
| `STATE.md` | Agent-loop iteration log | `git log` |
| `PRD.md` | Public-launch readiness plan | [docs/roadmap.md](../roadmap.md), [research.md](../../research.md) |

For what is actually true about this workspace today:

- **[docs/FINDINGS.md](../FINDINGS.md)** — every known correctness defect, with
  its measurement. The most useful document in the repository.
- **[CHANGELOG.md](../../../CHANGELOG.md)** — what changed, and what it measured.
- **[CONTRIBUTING.md](../../../CONTRIBUTING.md)** — how to work here.
- **[research.md](../../research.md)** — landscape research mapped to modules,
  with each recommendation scored against measured status.
