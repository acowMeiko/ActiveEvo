# Stage 2 Memory Factory

## Purpose

Stage 2 converts Stage-1 learning evidence into searchable Memory. It verifies
whether a Principle is supported, structured and executable; it does not claim
that the Principle improves accuracy. Positive or negative effect is measured
only in Stage 3.

## Data flow

```text
levelN learning packets
  -> build_learning_bundles
  -> validate_learning_bundle
  -> generate_principles
       all correct rollouts -> success consensus
       every wrong rollout -> causal divergence
       similar root causes -> 1-3 failure families
  -> validate_principle_content
  -> audit_principle
  -> repair_principle (at most once, only when decision=repair)
  -> audit_principle again
  -> reconcile_candidate
       add | covered | update | merge | reject
  -> append/version Store
  -> snapshot(levelN)
```

The closest correct trajectory is a concrete contrast, not the sole truth.
The generator must first infer consensus from all correct trajectories. This
avoids treating an accidental correct rollout as the gold reasoning method.

## Function map

| Function | Input | Output | Diagnose here when |
|---|---|---|---|
| `build_learning_bundles` | Stage-1 packet list | one bundle per question | trajectories disappear or duplicate |
| `validate_learning_bundle` | one bundle | explicit errors | source evidence is incomplete |
| `generate_principles` | all rollouts for one question | success consensus and 1-3 families | root cause or Principle is wrong |
| `validate_principle_content` | Principle JSON | normalized content/errors | field format is wrong |
| `audit_principle` | candidate and cited rollouts | per-dimension audit | quality rejection is wrong |
| `repair_principle` | repairable candidate/audit | repaired candidate | wording/scope is fixable |
| `reconcile_candidate` | approved candidate/existing Store | management operation | Memory duplicates or merges incorrectly |
| `build_level_memory` | one Level and configuration | artifacts, Store, summary | orchestration/resume/commit fails |

## Quality boundary

Deterministic rejection covers malformed schema, invalid evidence IDs and
literal source-answer leakage. The independent auditor separately judges source
fidelity, observable trigger/evidence/exclusion, executable action/checks and
supported generalization. Vague content receives one repair; unsupported or
contradictory content is rejected.

## Artifacts

```text
output/stage2/levelN/
  00_learning_bundles/
  04_generated_principles/
  06_quality_audits/
  08_memory_reconciliation/
  summary.json
```

`summary.json` reports candidate count, quality pass count, each Memory
operation and managed coverage. Every active Memory remains traceable to its
question, failure family and rollout IDs.
