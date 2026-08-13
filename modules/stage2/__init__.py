"""Stage 2 Memory Factory.

Stage 2 turns Stage-1 trajectory evidence into auditable, versioned memories.
It deliberately contains no neighbor-task validation or Solver evaluation.
"""

from modules.stage2.factory import build_level_memory

__all__ = ["build_level_memory"]
