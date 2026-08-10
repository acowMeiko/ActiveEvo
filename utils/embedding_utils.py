from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple


DEFAULT_EMBEDDING_MODEL = os.environ.get("MEMUA_EMBEDDING_MODEL", "BAAI/bge-m3")


def allow_hash_fallback() -> bool:
    return os.environ.get("MEMUA_ALLOW_HASH_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}


def resolve_embedding_candidates(model_name: str = DEFAULT_EMBEDDING_MODEL) -> List[str]:
    env_path = os.environ.get("MEMUA_EMBEDDING_MODEL_PATH", "").strip()
    candidates: List[str] = []
    if env_path:
        candidates.append(env_path)

    candidates.append(model_name)

    deduped: List[str] = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _candidate_exists(candidate: str) -> bool:
    return Path(candidate).exists()


def load_sentence_transformer_with_fallback(
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
    cache_folder: str | None = None,
    local_files_only: bool = True,
) -> Tuple[object, str]:
    from sentence_transformers import SentenceTransformer

    errors = []
    for candidate in resolve_embedding_candidates(model_name):
        kwargs = {"device": device}
        if cache_folder:
            kwargs["cache_folder"] = cache_folder

        # Local snapshot paths should always be treated as local files only.
        candidate_local_only = local_files_only or _candidate_exists(candidate)
        kwargs["local_files_only"] = candidate_local_only

        try:
            model = SentenceTransformer(candidate, **kwargs)
            return model, candidate
        except Exception as exc:
            err_str = str(exc).lower()
            errors.append(f"{candidate}: {exc}")
            # CUDA OOM → no point trying remaining candidates on same device
            if device != "cpu" and ("out of memory" in err_str or "oom" in err_str):
                break

    joined = "\n".join(errors)
    raise RuntimeError(
        "Failed to load a real embedding model. Tried these candidates:\n" + joined
    )
