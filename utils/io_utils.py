# utils/io_utils.py
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_json(path, encoding="utf-8") -> Any:
    """加载 JSON 文件"""
    path = Path(path)
    with open(path, "r", encoding=encoding) as f:
        data = json.load(f)
    logger.debug(f"Loaded {path.name} ({len(data) if isinstance(data, (list, dict)) else '?'} items)")
    return data


def save_json(data, path, encoding="utf-8", indent=2) -> None:
    """保存 JSON 文件（自动创建目录）"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
