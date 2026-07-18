# Documentation

This directory contains the current engineering rationale and future-work map.
The root [README](../README.md) remains the operational starting point.

## Current documents

- [Known issues](../KNOWN_ISSUES.md) — open artifacts and deferred design
  decisions (CLAHE boundary artifact, synthetic-halo rework, cache precision),
  with symptoms, causes and planned fixes.
- [Current design decisions](decisions.md) — what the live implementation does,
  the evidence behind it and the boundaries that should not be changed casually.
  Includes the tensor/scalogram detection path that isolates a behavior without a
  flow cache (the root README's "Live tensor detection" section is the operational
  view).
- [Expanded cache plan](expanded_cache_plan.md) — the dynamics/identity forward
  plan the scalogram explorer came from; its "Status update" records what shipped
  and how the scalogram-storage question was sidestepped.
- [Next steps](next-steps.md) — ranked validation and engineering work, including
  attractive approaches that are deliberately not recommended yet.

## Historical inputs

- [Initial project handoff](archive/initial-project-handoff.md) — the original
  full-frame, automatic-ROI design brief.
- [Standardization handoff](archive/standardization-handoff.md) — the proposal
  that initiated the standardization work.

Both archived handoffs contain useful context, but many prescriptions were
superseded by measurements on the difficult stabilized reference video. They are
not current requirements.
