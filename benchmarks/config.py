"""Benchmark 配置 — 集中管理所有测试集 + 模型端口分配。

新增 benchmark 只需在 BENCHMARKS 列表加一条即可。
"""
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent

# GPU 0-3 跑 LLaMA → ports 8000-8003
# GPU 4-7 跑 Qwen  → ports 8004-8007
MODELS: Dict[str, Dict] = {
    "llama": {
        "name": "llama",
        "gpus": [0, 1, 2, 3],
        "ports": [8000, 8001, 8002, 8003],
        "backbone": "llama",
    },
    "qwen": {
        "name": "qwen",
        "gpus": [4, 5, 6, 7],
        "ports": [8004, 8005, 8006, 8007],
        "backbone": "qwen",
    },
    # 8-GPU 部署 (同一模型占满 8 卡，轮询)
    "llama_8gpu": {
        "name": "llama",
        "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
        "ports": [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007],
        "backbone": "llama",
    },
    "qwen_8gpu": {
        "name": "qwen",
        "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
        "ports": [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007],
        "backbone": "qwen",
    },
}

# 6 个 benchmark 配置
BENCHMARKS: List[Dict] = [
    {
        "family": "deepscaler",
        "subset": "full",
        "name": "DeepScaler",
        "test_file": "data/test/deepscaler.json",
        "output_dir": "output/benchmarks/deepscaler",
    },
    {
        "family": "math500",
        "subset": "full",
        "name": "MATH500",
        "test_file": "data/test/benchmarks/math500_stage3.json",
        "output_dir": "output/benchmarks/math500",
    },
    {
        "family": "gsm8k",
        "subset": "full",
        "name": "GSM8K",
        "test_file": "data/test/benchmarks/gsm8k_stage3.json",
        "output_dir": "output/benchmarks/gsm8k",
    },
    {
        "family": "bbh",
        "subset": "logic",
        "name": "BBH Logic",
        "test_file": "data/test/benchmarks_split/bbh_logic_stage3.json",
        "output_dir": "output/benchmarks/bbh_logic",
    },
    {
        "family": "bbh",
        "subset": "math",
        "name": "BBH Math",
        "test_file": "data/test/benchmarks_split/bbh_math_stage3.json",
        "output_dir": "output/benchmarks/bbh_math",
    },
    {
        "family": "svamp",
        "subset": "official_test",
        "name": "SVAMP",
        "test_file": "data/test/official/svamp/processed/test.json",
        "output_dir": "output/benchmarks/svamp",
    },
    {
        "family": "general365",
        "subset": "official_test",
        "name": "General365",
        "test_file": "data/test/official/general365/processed/test.json",
        "output_dir": "output/benchmarks/general365",
    },
]
