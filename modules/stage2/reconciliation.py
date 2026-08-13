"""Reconcile quality-approved candidates with the cumulative Memory Store."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from modules.principle_schema import validate_principle_content


async def reconcile_candidate(client: Any, pipeline: Any, candidate: Dict[str, Any]) -> Dict[str, Any]:
    if not pipeline.memory.memory.principles:
        return {"operation": "add", "matches": []}
    if not pipeline.memory.is_indexed:
        pipeline.memory.build_indices()
    matches = pipeline.memory.retrieve_strict_semantic_matches(
        candidate.get("source_dsd", {}), facet_threshold=0.80, top_k=5
    )
    if not matches:
        return {"operation": "add", "matches": []}
    existing = [
        {
            "memory_id": item["item"].get("memory_id"),
            "content": item["item"].get("content", {}),
            "source_dsd": item["item"].get("source_dsd", {}),
        }
        for item in matches
    ]
    prompt = f"""Compare one quality-approved mathematical Principle with existing memories.
Choose exactly one operation:
- covered: an existing memory already fully contains the same trigger, exclusions and action.
- update: the candidate corrects or safely strengthens one existing memory.
- merge: both contain compatible complementary guidance and can become one coherent memory.
- add: the candidate is meaningfully distinct.

For covered return target_memory_id. For update/merge also return result_principle in the
exact canonical Principle schema. Never broaden scope merely to force a merge.
Output raw JSON only:
{{"operation":"add|covered|update|merge","target_memory_id":"", "result_principle":{{}}, "reason":"..."}}

Candidate: {json.dumps(candidate['content'], ensure_ascii=False)}
Existing: {json.dumps(existing, ensure_ascii=False)}
"""
    response = await client.auditor(prompt)
    parsed = response["parsed"]
    operation = str(parsed.get("operation", "add")).lower()
    ids = {str(item["memory_id"]) for item in existing}
    target = str(parsed.get("target_memory_id", ""))
    if operation not in {"add", "covered", "update", "merge"}:
        operation = "add"
    if operation != "add" and target not in ids:
        operation = "add"
        target = ""
    result_content: Dict[str, Any] = {}
    schema_errors: List[str] = []
    if operation in {"update", "merge"}:
        result_content, schema_errors = validate_principle_content(
            parsed.get("result_principle"), require_extended=True
        )
        if schema_errors:
            operation = "add"
            target = ""
            result_content = {}
    return {
        "operation": operation,
        "target_memory_id": target,
        "result_content": result_content,
        "reason": str(parsed.get("reason", "")),
        "schema_errors": schema_errors,
        "matches": existing,
        "raw_response": response["raw"],
    }
