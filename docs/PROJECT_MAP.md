# 项目文件与流程地图

本文件只描述代码、配置和规范产物；运行时生成的每一个 JSON 不逐项枚举。

## 1. 核心流程

| 流程 | 主入口 | 主要实现 | 输入 | 输出 |
|---|---|---|---|---|
| 部署 | `deploy_model.sh` | `api_call/run.sh` | `deploy.conf` | vLLM endpoints、active manifest |
| Stage 1 | `pipeline.py --stage 1`、`build_learning_packets.py` | `modules/stage1_sampler.py`、`stage1_miner.py`、`execute_module.py` | `data/` | `learning_packet/`、`output/stage1/` |
| Stage 2 | `run_stage2_all_levels.sh` | `core/agent_pipeline.py`、`modules/stage2_repair.py`、`memory.py` | learning packets | `memory/` |
| Stage 3 | `run_all_tests.sh`、`evaluate_principle_guided_test.py` | `retrieve_principles.py`、gate、reranker、solver | test set + Memory | `output/`、`experiments/` |
| 论文证据 | `paper/evidence/scripts/extract_evidence.py` | summary extractor | `output/`、`memory/` | summaries、source manifest |

## 2. 根目录文件

| 文件 | 控制内容 | 状态 |
|---|---|---|
| `deploy.conf` | GPU、端口基数、模型路径、vLLM 参数和 seed | 唯一部署配置 |
| `config.env` | Stage 2/3 参数和嵌入模型 | 唯一流水线配置 |
| `deploy_model.sh` | `api_call/run.sh` 的稳定快捷入口 | 主入口 |
| `pipeline.py` | Stage 1–3 和 curriculum 总调度 | 主入口 |
| `run_stage1_dual.py` / `run_stage1_llama.py` | backbone 专用 Stage 1 运行器 | 专项入口 |
| `build_learning_packets.py` | 构建正负轨迹 learning packets | Stage 1 |
| `sample_unstable_questions.py` | 多次推理与能力分区 | Stage 1 |
| `hard_negative_mining.py` | 正确/错误轨迹困难负样本挖掘 | Stage 1 |
| `generate_principle.py` | 语义、diff 与 principle 生成；仍被评测代码导入 | Stage 2 共享 |
| `build_vector_db.py` / `parse_memory_semantics.py` | Memory records 与三维向量索引 | Stage 2 |
| `run_stage2_all_levels.sh` | 四个 level 的批量 Memory 构建 | Stage 2 主入口 |
| `retrieve_principles.py` | 在线三维召回与兼容接口 | Stage 3 |
| `evaluate_principle_guided_test.py` | direct/guided、Gate、paired 评测 | Stage 3 主入口 |
| `run_all_tests.sh` | 多测试集批处理 | Stage 3 主入口 |
| `embedding_utils.py` | bge-m3 路径选择和受控 fallback | 共享 |
| `output_paths.py` | Stage 1–3 规范路径 | 共享 |

## 3. 源码目录

### `core/`

| 文件 | 职责 |
|---|---|
| `agent_pipeline.py` | 三阶段总编排、增量轮次与反馈闭环 |
| `api_call.py` | OpenAI-compatible 请求、seed 和重试 |
| `endpoint_pool.py` | 多端点并发调度 |
| `config.py` | Python 侧配置对象与路径 |
| `field_names.py` | Memory/Principle 字段名规范 |
| `runtime_metrics.py` | token、时延与调用统计 |
| `thread_checkpoint.py` | goal/plan/step/observation 等 Working State Checkpoint |

### `modules/`

| 文件 | 职责 |
|---|---|
| `stage1_sampler.py` / `stage1_miner.py` | 分区采样与困难轨迹挖掘 |
| `stage2_repair.py` | Stage 2 schema 修复和补全 |
| `principle_schema.py` | principle 字段、验证与规范化 |
| `plan_module.py` | 题目三维结构抽取和计划 |
| `execute_module.py` | solver、答案生成和结果检查 |
| `memory.py` | Memory Store、反馈、快照和三维检索 |
| `memory_module.py` | Agent 对 Memory 的读写适配层 |
| `dsd_module.py` | domain/constraints/objective 三维结构模块 |
| `reranker_module.py` | 候选重排、Application Gate 和适配 |

### `benchmarks/`

`evaluator.py` 运行 direct/guided 对照；`matcher.py` 统一答案判定；`retrieval_compare.py` 对比检索；`run_suite.py` 编排 benchmark；`summary.py` 汇总结果。Shell 文件是对应的批处理入口。

## 4. 实验与分析

| 路径 | 内容 |
|---|---|
| `experiments/llama_deepscaler_l2_l3_seed42/` | 100 题初始配对实验和反事实回放 |
| `experiments/solver_v1_gpu67/` | Solver v1、重复惩罚与 prompt 冻结实验 |
| `experiments/stage3_gate_adaptation_gpu467/` | Raw/Adapted memory 对照 |
| `experiments/stage3_topk_gate_gpu467/` | Top-k、Application Gate 和 DeepSeek 标注 |
| `ablation/single_vector_principle/` | 单向量检索消融 |
| `ablation/no_learning_zone_collection/` | 去除 learning-zone collection 消融 |
| `error_analysis/` | principle、检索与题目级错误分析 |

目录名中的 GPU 编号是历史实验标识，不是当前部署配置。

## 5. 工具与测试

- `tools/` 根部：当前 Stage 3 配对、重放、Gate、标注和汇总工具。
- `tools/maintenance/`：数据修复、向量库重建、benchmark 准备和可审计清理。
- `tools/paper/`：DBLP/Crossref/BibTeX 辅助脚本。
- `tools/sync/`：A800 节点同步脚本。
- `tests/`：Gate、schema、matcher、reranker、solver 和指标单元测试。
- `scripts/legacy/`：生成既有论文结果时使用的旧端口/旧目录脚本，仅供追溯，不作为当前入口。

## 6. 生成数据的保留规则

| 路径 | 规则 |
|---|---|
| `learning_packet/` | Stage 1 规范结果，保留 |
| `memory/` | 当前 Store/Checkpoint/索引，保留；不存模型权重 |
| `output/` | 当前评测与消融结果，保留 |
| `paper/evidence/` | 只保留摘要、来源 manifest 和生成脚本 |
| `logs/`、`__pycache__/`、`*.bak`、`*.tmp` | 可再生，清理 |
| `*-before/` | 不放在活动目录；历史版本进入同级 archive |

历史归档及清理校验清单位于 `/home/czhu/document/ActiveEvo_ICASSP_2027/archive/code_cleanup_2026-08-09/`。
