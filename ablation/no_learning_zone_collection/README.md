# No Learning-Zone Collection Ablation

This ablation is independent from `single_vector_principle`.

It removes the Stage 1 learning-zone collection gate: every training sample receives
the full Stage 1 sampling budget, and downstream hard-negative mining, Stage 1.3,
Stage 2, and Stage 3 run from that all-sample collection.

Outputs are written under this ablation directory by default and do not overwrite
the main `output/` or `memory/` directories.
