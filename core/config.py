# core/config.py
from __future__ import annotations

import logging
import re
from datetime import datetime
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional, Tuple


# Local deployments may choose any unprivileged TCP port. The deployment
# scripts still use ``DEPLOY_PORT_BASE + GPU_ID`` as their default mapping.
ALLOWED_LOCAL_API_PORTS = tuple(range(1024, 65536))


def apply_local_api_port(url: str, port: int) -> str:
    """将本地 API URL 端口切换到指定值。"""
    if port not in ALLOWED_LOCAL_API_PORTS:
        raise ValueError(f"Unsupported port: {port}")
    if not url:
        return f"http://localhost:{port}/v1/chat/completions"
    return re.sub(r":\d+(?=/)", f":{port}", url)


@dataclass(frozen=True)
class BaseConfig:
    """基础配置类，包含项目根目录"""
    project_root: Path
    
    @classmethod
    def from_core_file(cls) -> "BaseConfig":
        """基于当前文件(core/config.py)定位项目根目录"""
        project_root = Path(__file__).resolve().parent.parent
        return cls(project_root=project_root)


@dataclass(frozen=True)
class LogConfig:
    """日志配置"""
    project_root: Path
    log_dir_name: str = "logs"
    log_file_name: str = "app.log"
    log_level: int = logging.DEBUG
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @property
    def log_dir(self) -> Path:
        return self.project_root / self.log_dir_name

    @property
    def log_file(self) -> Path:
        return self.log_dir / self.log_file_name

    def ensure_dirs(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self, *, force: bool = False) -> None:
        self.ensure_dirs()
        root = logging.getLogger()
        if force:
            for h in list(root.handlers):
                root.removeHandler(h)
        if root.handlers:
            return

        root.setLevel(logging.DEBUG)
        formatter = logging.Formatter(self.log_format)

        # 终端: INFO 级别 — 只显示关键进度
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)

        # 文件: DEBUG 级别 — 完整记录
        log_stem = Path(self.log_file_name).stem
        log_suffix = Path(self.log_file_name).suffix or ".log"
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_log_file = self.log_dir / f"{log_stem}_{run_tag}{log_suffix}"

        fh = logging.FileHandler(run_log_file, encoding="utf-8", mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        root.addHandler(sh)
        root.addHandler(fh)

        # 抑制三方库噪音 (httpx/httpcore/filelock/urllib3)
        for noisy in ("httpx", "httpcore", "filelock", "urllib3", "hf_xet"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        root.info(f"日志文件: {run_log_file}")


@dataclass(frozen=True)
class PathConfig:
    """路径配置"""
    project_root: Path
    data_dir_name: str = "data"
    memory_dir_name: str = "memory_data"
    test_dir_name: str = "test"
    output_dir_name: str = "output"
    checkpoint_dir_name: str = "checkpoints"
    models_dir_name: str = "models"
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / self.data_dir_name
    
    @property
    def memory_dir(self) -> Path:
        return self.data_dir / self.memory_dir_name
    
    @property
    def test_dir(self) -> Path:
        return self.data_dir / self.test_dir_name
    
    @property
    def output_dir(self) -> Path:
        output_override = os.environ.get("MEMUA_OUTPUT_DIR", "").strip()
        if output_override:
            return Path(output_override).expanduser().resolve()
        output_dir_name = os.environ.get("MEMUA_OUTPUT_DIR_NAME", "").strip() or self.output_dir_name
        return self.project_root / output_dir_name
    
    @property
    def checkpoint_dir(self) -> Path:
        return self.project_root / self.checkpoint_dir_name

    @property
    def models_dir(self) -> Path:
        """Role-based model mount point; model weights are never committed."""
        return self.project_root / self.models_dir_name

    @property
    def actor_model_dir(self) -> Path:
        return self.models_dir / "actor"

    @property
    def generator_model_dir(self) -> Path:
        return self.models_dir / "generator"

    @property
    def auditor_model_dir(self) -> Path:
        return self.models_dir / "auditor"

    @property
    def embedding_model_dir(self) -> Path:
        return self.models_dir / "embedding"
    
    @property
    def memory_file(self) -> Path:
        return self.memory_dir / "memory.json"
    
    @property
    def memory_checkpoint_file(self) -> Path:
        return self.checkpoint_dir / "memory_checkpoint.json"
    
    @property
    def memory_db_file(self) -> Path:
        return self.memory_dir / "memory.db" # SQLite 数据库

    @property
    def faiss_index_file(self) -> Path:
        return self.memory_dir / "memory.index" # FAISS 索引文件
        
    def ensure_dirs(self) -> None:
        """创建所有必需的目录"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for model_dir in (
            self.actor_model_dir,
            self.generator_model_dir,
            self.auditor_model_dir,
            self.embedding_model_dir,
        ):
            model_dir.mkdir(parents=True, exist_ok=True)


# ==================== 全局常量 ====================
# 六大固定类目 (唯一定义点，其他模块统一从此处导入)
FIXED_CATEGORIES = [
    "Numerical_Algebraic_Operations",
    "Functions_Equations",
    "Trigonometry_Transformations",
    "Geometry_Stereometry",
    "Linear_Algebra_Matrices",
    "Combinatorics_Logic",
]


@dataclass(frozen=True)
class HyperParamsConfig:
    """超参数配置

    ⚠️ 使用前请确认:
      1. weak_model_name 与 deploy.conf / active_deployment.env 中模型名一致
      2. weak_model_url 端口与 active_deployment.env 中实际端口一致
      3. 本地 vLLM 由 api_call/run.sh 启动，ActiveEvo 只负责调用其 API
    """

    # === 1. 弱模型 (本地 vLLM，由 api_call 模块部署) ===
    # model_name 必须与 deploy.conf 中 DEPLOY_MODEL 一致
    weak_model_name: str = os.environ.get("ACTIVEEVO_WEAK_MODEL", "llama")
    weak_model_url: str = os.environ.get(
        "ACTIVEEVO_WEAK_MODEL_URL", "http://localhost:8100/v1/chat/completions"
    )
    weak_model_key: str = os.environ.get("ACTIVEEVO_WEAK_MODEL_API_KEY", "EMPTY")
    weak_timeout: int = 180

    # 通用参数
    temperature: float = 0.7
    max_tokens: int = 7500
    top_p: float = 0.95
    
    # 推理相关
    batch_size: int = 32
    max_workers: int = 4
    tensor_parallel_size: int = 4
    
    # GPU 相关 (8 × A100 80G)
    cuda_visible_devices: str = "0,1,2,3,4,5,6,7"
    # FAISS
    facet_weights: tuple = (0.5, 0.3, 0.2)
    faiss_hard_threshold: float = 0.8 # 一票否决硬性及格线
    faiss_model_name: str = 'all-MiniLM-L6-v2' # 极速轻量级 Embedding 模型


@dataclass(frozen=True)
class PipelineConfig:
    """User-facing pipeline defaults.

    Deployment remains in api_call/. The pipeline only reads endpoint ports and
    controls Stage 1/2/3 experiment flow.
    """

    public_levels: Tuple[str, ...] = ("level1", "level2", "level3", "level4")
    default_ports: Tuple[int, ...] = (8100, 8101, 8102, 8103)
    default_model: str = "llama"
    default_backbone: str = "default"
    default_memory_dir: str = "memory/store"
    default_test_file: str = "data/test/deepscaler.json"
    default_temperature: float = 0.2
    default_pass_k: int = 3
    default_seeds: Tuple[int, ...] = (42, 43, 44)
    stage1_dataset_path: str = ""
    stage1_expected_dataset_sha256: str = "9d8b634043cb008076cbf60b7c8a01fddaeb2a83f9d3a8de564b05dd176eb225"
    stage1_expected_dataset_size: int = 6000
    stage1_k_trials: int = 10
    # Capability zones are mutually exclusive.  With ten rollouts:
    # Panic = 0-1 correct, Learning = 2-8 correct, Master = 9-10 correct.
    stage1_panic_max_correct: int = 1
    stage1_master_min_correct: int = 9
    stage1_experiment_seed: int = 42
    stage1_temperature: float = 0.7
    stage1_top_p: float = 0.95
    stage1_top_logprobs: int = 5
    stage1_entropy_histogram_bins: int = 20
    # Stage 1 uses the same two-stage Solver-v1.1 protocol as downstream
    # baseline/guided evaluation.  This is the Reasoner budget; the Finalizer
    # has its own small deterministic budget.
    stage1_max_tokens: int = 8192
    stage1_finalizer_max_tokens: int = 128
    stage1_finalizer_timeout: int = 60
    stage1_disable_early_stop: bool = True
    stage1_concurrency: int = 16
    stage1_task_batch_size: int = 1000
    stage2_concurrency: int = 8
    stage2_source_validation_concurrency: int = 2
    stage2_source_validation_seeds: Tuple[int, ...] = tuple(range(42, 52))
    stage2_source_validation_temperatures: Tuple[float, ...] = (0.7, 0.2)
    stage3_concurrency: int = 8
    stage3_threshold: float = 0.55
    stage3_candidate_top_k: int = 12
    stage3_gate_top_k: int = 6
    stage3_max_injected: int = 1
    embedding_device: str = "cpu"
    embedding_model: str = "BAAI/bge-m3"
    force_embedding: str = "1"
    allow_hash_fallback: str = "0"
    request_timeout: int = 600


@dataclass(frozen=True)
class Stage2Config:
    """Memory Factory defaults, independent from Stage-3 evaluation."""

    max_principles_per_question: int = 3
    reconciliation_top_k: int = 5
    generation_analysis_max_tokens: int = 8192
    generation_max_tokens: int = 4096
    audit_max_tokens: int = 8192
    audit_retry_max_tokens: int = 8192
    audit_formatter_max_tokens: int = 3072
    generator_timeout: int = 900
    auditor_timeout: int = 900
    generator_retries: int = 2
    auditor_retries_per_endpoint: int = 1
    generator_thinking_enabled: bool = True
    generator_synthesis_thinking_enabled: bool = False
    generator_seed: int = 42
    qwen_generation_analysis_max_tokens: int = 6144
    qwen_generation_max_tokens: int = 4096
    qwen_generation_retry_max_tokens: int = 8192
    qwen_generator_model: str = os.environ.get("ACTIVEEVO_QWEN_GENERATOR_MODEL", "qwen")
    qwen_generator_urls: tuple[str, ...] = tuple(
        value
        for value in re.split(
            r"[\s,]+",
            os.environ.get(
                "ACTIVEEVO_QWEN_GENERATOR_URLS",
                "http://localhost:8200/v1/chat/completions,http://localhost:8201/v1/chat/completions",
            ).strip(),
        )
        if value
    )
    qwen_generator_key_env: str = "ACTIVEEVO_QWEN_GENERATOR_API_KEY"
    qwen_generator_temperature: float = 1.0
    qwen_generator_top_p: float = 0.95
    qwen_generator_top_k: int = 20
    qwen_generator_min_p: float = 0.0
    qwen_generator_presence_penalty: float = 0.0
    qwen_generator_repetition_penalty: float = 1.0
    # ``medium`` is accepted by both vLLM's outer request schema and Qwen3.8's
    # model-level parser.  It keeps thinking enabled without the long xhigh
    # deliberation that exhausted the source-generation timeout.
    qwen_generator_reasoning_effort: str = "medium"
    auditor_thinking_enabled: bool = True
    auditor_reasoning_effort: str = "medium"
    validation_seed: int = 42
    generator_model: str = os.environ.get("ACTIVEEVO_GENERATOR_MODEL", "deepseek-v4-pro")
    generator_url: str = os.environ.get(
        "ACTIVEEVO_GENERATOR_URL", "https://api.deepseek.com/v1/chat/completions"
    )
    generator_key_env: str = "ACTIVEEVO_GENERATOR_API_KEY"
    auditor_provider: str = os.environ.get("ACTIVEEVO_AUDITOR_PROVIDER", "kimi")
    auditor_model: str = os.environ.get("ACTIVEEVO_AUDITOR_MODEL", "kimi-k2.6")
    auditor_url: str = os.environ.get(
        "ACTIVEEVO_AUDITOR_URL", "https://api.moonshot.cn/v1/chat/completions"
    )
    auditor_urls: tuple[str, ...] = tuple(
        value
        for value in re.split(
            r"[\s,]+",
            os.environ.get("ACTIVEEVO_AUDITOR_URLS", "").strip()
            or os.environ.get(
                "ACTIVEEVO_AUDITOR_URL",
                "https://api.moonshot.cn/v1/chat/completions",
            ).strip(),
        )
        if value
    )
    auditor_key_env: str = "ACTIVEEVO_AUDITOR_API_KEY"
    kimi_thinking_temperature: float = 1.0
    kimi_thinking_top_p: float = 0.95
    kimi_nonthinking_temperature: float = 0.6
    kimi_nonthinking_top_p: float = 0.95
    principle_test_seed: int = 42
    principle_test_sources_per_level: int = 10
    principle_test_reserve_per_level: int = 5
    principle_test_audit_repeats: int = 3
    principle_test_variant_attempts: int = 3
    principle_test_solver_seeds: tuple[int, ...] = tuple(range(42, 52))
    principle_test_primary_temperature: float = 0.2
    principle_test_robustness_temperature: float = 0.7
    role_compare_seeds: tuple[int, ...] = (42, 43, 44)
    role_compare_temperature: float = 0.2
    role_compare_initial_concurrency: int = 4
    role_compare_kimi_critic_max_tokens: int = 4096
    role_compare_audit_max_tokens: int = 8192
    role_compare_kimi_generation_retry_max_tokens: int = 8192

@dataclass(frozen=True)
class Config:
    """主配置类，整合所有子配置"""
    project_root: Path
    log: LogConfig = field(default_factory=lambda: None)
    paths: PathConfig = field(default_factory=lambda: None)
    hyperparams: HyperParamsConfig = field(default_factory=lambda: None)
    pipeline: PipelineConfig = field(default_factory=lambda: None)
    stage2: Stage2Config = field(default_factory=lambda: None)
    
    @classmethod
    def from_core_file(cls) -> "Config":
        """基于当前文件定位项目根目录"""
        project_root = Path(__file__).resolve().parent.parent
        
        log_config = LogConfig(project_root=project_root)
        path_config = PathConfig(project_root=project_root)
        hyperparam_config = HyperParamsConfig()
        pipeline_config = PipelineConfig()
        stage2_config = Stage2Config()
        
        return cls(
            project_root=project_root,
            log=log_config,
            paths=path_config,
            hyperparams=hyperparam_config,
            pipeline=pipeline_config,
            stage2=stage2_config,
        )
    
    def ensure_dirs(self) -> None:
        self.log.ensure_dirs()
        self.paths.ensure_dirs()
    
    def setup_logging(self, *, force: bool = False) -> None:
        self.log.setup_logging(force=force)


# ==================== api_call 模块参考配置 ====================
# 以下配置仅供文档参考，实际模型部署请使用 api_call/ 模块:
#   启动:  bash api_call/run.sh
#   停止:  bash api_call/stop.sh
#   配置:  编辑 deploy.conf
#
# 部署完成后，ActiveEvo 通过 HyperParamsConfig.weak_model_url 调用 API。

LOCAL_VLLM_CONFIG = {
    "model_path": os.environ.get("ACTIVEEVO_MODEL_PATH", "/path/to/local/model"),
    "served_model_name": "llama",
    "host": "0.0.0.0",
    "port": 8000,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.92,
    "max_model_len": 32768,
    "max_num_seqs": 128,
}


def get_local_vllm_command(
    model_path: str = LOCAL_VLLM_CONFIG["model_path"],
    port: int = LOCAL_VLLM_CONFIG["port"],
    served_model_name: str = LOCAL_VLLM_CONFIG["served_model_name"],
    tp: int = LOCAL_VLLM_CONFIG["tensor_parallel_size"],
    gpu_mem: float = LOCAL_VLLM_CONFIG["gpu_memory_utilization"],
    max_model_len: int = LOCAL_VLLM_CONFIG["max_model_len"],
) -> str:
    """生成本地单卡 vLLM 启动命令 (仅供临时测试)。

    生产环境请使用 api_call/run.sh 部署:
      bash api_call/run.sh
    """
    cmd = (
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {model_path} "
        f"--served-model-name {served_model_name} "
        f"--host 0.0.0.0 --port {port} "
        f"--tensor-parallel-size {tp} "
        f"--gpu-memory-utilization {gpu_mem} "
        f"--max-model-len {max_model_len} "
        f"--enforce-eager"
    )
    return cmd


def get_multi_gpu_vllm_command(
    model_path: str = LOCAL_VLLM_CONFIG["model_path"],
    ports: list = None,
    served_model_name: str = "qwen",
    tp: int = 1,
    gpu_mem: float = 0.92,
    max_model_len: int = 32768,
) -> str:
    """生成多卡 vLLM 启动命令 (仅供临时测试)。

    生产环境请使用 api_call/run.sh 部署多 Backbone:
      bash api_call/run.sh
    """
    if ports is None:
        ports = [8000, 8001, 8002, 8003]

    lines = []
    for gpu_id, port in enumerate(ports):
        lines.append(
            f"CUDA_VISIBLE_DEVICES={gpu_id} python -m vllm.entrypoints.openai.api_server "
            f"--model {model_path} "
            f"--served-model-name {served_model_name} "
            f"--host 0.0.0.0 --port {port} "
            f"--tensor-parallel-size {tp} "
            f"--gpu-memory-utilization {gpu_mem} "
            f"--max-model-len {max_model_len} &"
        )
    return "\n".join(lines)
