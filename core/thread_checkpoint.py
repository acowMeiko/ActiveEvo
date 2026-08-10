"""Human-readable thread checkpoints for resumable single-problem inference."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class ThreadCheckpointManager:
    """Persist per-thread working state without mixing it into long-term memory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        thread_id: str,
        checkpoint_name: str,
        state: Dict[str, Any],
        parent_checkpoint: Optional[str] = None,
    ) -> Path:
        thread_dir = self.root / thread_id
        thread_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_id = f"{thread_id}:{checkpoint_name}"
        payload = {
            "checkpoint_id": checkpoint_id,
            "thread_id": thread_id,
            "parent_checkpoint": parent_checkpoint,
            "checkpoint_name": checkpoint_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "goal": state.get("goal", "solve the current math problem completely"),
            "plan": state.get("plan", []),
            "current_step": state.get("current_step", checkpoint_name),
            "recent_observation": state.get("recent_observation"),
            "recent_action": state.get("recent_action"),
            "recent_tool_result": state.get("recent_tool_result"),
            "recent_error": state.get("recent_error"),
            "budget": state.get("budget", {}),
            "selected_memory_ids": state.get("selected_memory_ids", []),
            "selected_skill_ids": state.get("selected_skill_ids", []),
            "thread_status": state.get("thread_status", "running"),
            "store_version": state.get("store_version", 0),
            "rng": state.get("rng", {}),
            "config": state.get("config", {}),
            "evaluation_state": state.get("evaluation_state", {}),
        }
        target = thread_dir / f"{checkpoint_name}.json"
        temp = target.with_suffix(".json.tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp, target)
        latest = thread_dir / "latest.json"
        latest_temp = latest.with_suffix(".json.tmp")
        with latest_temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(latest_temp, latest)
        return target
