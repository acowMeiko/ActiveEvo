# SUAgent (MemMetaV3) 代码流程文档

## 概述

SUAgent 是一个**经验驱动的自进化推理框架**，面向数学问题求解。系统通过 3 阶段课程学习流程，在 4 轮迭代中逐步构建一个可检索的经验记忆库（validated reasoning principles），并在测试时通过检索相关知识来引导模型推理。

- **模型**: Llama 3.1 8B / Qwen 2.5 14B
- **硬件**: 8× A800 80GB
- **推理引擎**: vLLM（OpenAI 兼容 API）
- **嵌入模型**: BAAI/bge-m3
- **向量索引**: FAISS (IndexFlatIP)

---

## 目录结构

```
SUAgent/
├── core/                       # 核心配置、LLM 调用、编排管线
│   ├── config.py               # 全局配置 (路径、超参、模型 URL)
│   ├── agent_pipeline.py       # 主编排器: 3 阶段 + 完整 curriculum 流程
│   ├── api_call.py             # 异步/同步 LLM API 调用 (含重试 + 超时)
│   ├── endpoint_pool.py        # 负载感知的多 GPU 请求路由
│   ├── field_names.py          # 语义字段常量 (core_skill 等)
│   └── runtime_metrics.py      # 运行时指标收集 (tokens, 延迟, 成功率)
│
├── modules/                    # 7 个核心可执行模块
│   ├── plan_module.py          # 规划模块: 语义提取、差异分析、原则生成/修复
│   ├── memory.py               # 记忆存储引擎: FAISS 索引、多策略检索、快照
│   ├── memory_module.py        # 记忆模块包装器: 从不同来源入库原则
│   ├── execute_module.py       # 执行模块: 直接/引导/基线推理 + 答案校验
│   ├── stage1_sampler.py       # Stage1 采样器: K 次试验 + 三区分类
│   ├── stage1_miner.py         # Stage1 挖掘器: 难负样本对挖掘 + 差异分析
│   ├── stage2_repair.py        # Stage2 递归修复: 邻居验证 → 诊断 → 修补
│   └── dsd_module.py           # DSD 注册表: 数据集级语义签名 + 邻居缓存
│
├── benchmarks/                 # Benchmark 评测系统
│   ├── config.py               # 6 个 benchmark + 模型端口映射
│   ├── evaluator.py            # 单 benchmark 评估器 (guided/baseline/direct/base)
│   ├── matcher.py              # 答案匹配器 (boxed LaTeX、数值、多选题)
│   ├── run_suite.py            # 批量运行器 (遍历模型 × benchmark)
│   ├── summary.py              # 结果汇总 (Markdown/JSON)
│   ├── run_all.sh              # 一键跑全部 benchmark (双模型 或 8-GPU 单模型)
│   └── run_one.sh              # 单独跑某个 benchmark
│
├── api_call/                   # vLLM 多卡部署系统
│   ├── run.sh                  # 启动 vLLM 实例 (每 GPU 一个, 支持 --daemon)
│   ├── stop.sh                 # 停止全部 vLLM 进程
│   └── test.sh                 # 端点健康检查
│
├── ablation/                   # 消融实验
│   ├── no_learning_zone_collection/  # 无学习区消融 (全量采样)
│   └── single_vector_principle/      # 单向量检索消融
│
├── template/
│   └── prompt_template.py      # 全部 LLM Prompt 模板
│
├── utils/                      # 工具函数 (IO、JSON、答案解析、汇总)
├── scripts/                    # 辅助脚本 (数据集准备、Stage3 评估、修复)
├── data/                       # 输入数据 (测试集、DPO 数据)
├── memory/                     # Stage2 产物 + 在线记忆库
├── output/                     # Stage1/2/3 输出 + benchmark 结果
├── learning_packet/            # Stage1 产物 (采样结果、学习包)
├── logs/                       # 运行日志
├── deploy.conf                 # vLLM 部署配置
└── output_paths.py             # 集中化路径解析
```

---

## 三阶段管线

### 总体数据流

```
data/deepscaler_merged_6000.json (6000 道数学题)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1: 学习区发现 (Learning Zone Discovery)        │
│                                                      │
│  Stage1Sampler: K=10 次独立采样                      │
│    → 三区分类: Master (SR=1) / Learning (0<SR<1)    │
│       / Panic (SR=0)                                 │
│    → Learning 区按 SR 分 4 级                         │
│                                                      │
│  Stage1Miner: 难负样本对挖掘                          │
│    → 发散点检测 (文本相似度 + 数学原子重叠)          │
│    → LLM 差异分析                                    │
│    → 打包为 learning_packets (按级别)                 │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: 记忆构建 (Memory Construction)              │
│                                                      │
│  DSDRegistry: 对所有源问题提取 3 维语义签名          │
│    → BAAI/bge-m3 编码 + 预计算邻居缓存               │
│                                                      │
│  Self-CPE: 从差异分析中提取防御性原则                 │
│    (introspection + trigger + action + scope)        │
│                                                      │
│  RecursiveRepair: 邻居验证 → 诊断 → 修补 → 再验证   │
│    (最多 3 轮递归)                                   │
│                                                      │
│  记忆调和: 新原则 vs 已有记忆                        │
│    → replace / merge / add                           │
│                                                      │
│  FAISS 索引构建 + 磁盘持久化 + 快照                   │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: 检索增强评测 (Retrieval-Augmented Eval)     │
│                                                      │
│  对每道测试题:                                       │
│    1. extract_semantics(question) → 3 维语义签名     │
│    2. memory.retrieve(semantics, top_k=5) → 原则     │
│    3. execute.answer_guided(question, principle)     │
│    4. check_answer(response, expected) → 正确/错误   │
│                                                      │
│  错误重试: 最多 2 次 (带上次尝试上下文)               │
│  渐进评测: 已做对的题跨轮次不重复测试                 │
└─────────────────────────────────────────────────────┘
```

### Curriculum 流程 (4 轮)

```
Round 1: 全量 Stage1 → Stage2 (Level 1 记忆) → Stage3 → Panic 救援
Round 2: Panic 重采样 → Stage2 (Level 2 记忆) → Stage3 → Panic 救援
Round 3: Panic 重采样 → Stage2 (Level 3 记忆) → Stage3 → Panic 救援
Round 4: Panic 重采样 → Stage2 (Level 4 记忆) → Stage3 (最终评测)

每轮记忆累加 (RU = Rebase-Update 策略):
  Round 1 记忆 ──→ Round 2 记忆 ──→ Round 3 记忆 ──→ Round 4 记忆
```

---

## 核心模块详解

### 1. PlanModule — 规划模块
`modules/plan_module.py`

| 方法 | 功能 |
|---|---|
| `extract_semantics(question)` | 提取 3 维语义签名: core_skill, context_constraints, task_objective |
| `extract_holistic_description(question)` | 生成单一整体检索键 (消融用) |
| `diff_analysis(question, chosen, rejected, semantics)` | 对比正确/错误推理路径，定位发散点 |
| `generate_principles(question, diff_list, semantics)` | 将错误模式抽象为可复用的推理原则 |
| `crucible_test(question, principle, expected)` | 验证原则: 用原则引导推理，检查正确性 |
| `judge_principle_consistency(...)` | LLM 判断原则是否重复/可合并/新增 |
| `self_cpe_extract(question, diff_list, source_dsd)` | 自对比原则提取 (Stage2 核心) |
| `repair_diagnose(principle, failed_validations)` | 诊断原则失败原因 (vague/over_general/misdirected) |
| `repair_patch(principle, diagnosis, failed_example)` | 基于诊断修补缺陷原则 |

### 2. Memory — 记忆存储引擎
`modules/memory.py`

**索引结构**:
```
4 个 FAISS IndexFlatIP:
  ├── core_skill.index        (DSD 3 维之一)
  ├── context_constraints.index
  ├── task_objective.index
  └── single_vector.index     (单向量消融)
```

**检索策略**:

| 方法 | 策略 |
|---|---|
| `retrieve()` | 推理对齐: 多面召回 + 质量加权重排 |
| `retrieve_hard_gated()` | 硬门控: 3 个面都必须过阈值 |
| `retrieve_reasoning_aligned()` | 结构分 (0.72) + 质量分 (0.28) 融合 |
| `retrieve_single_vector()` | 单一整体描述向量 → KNN |
| `retrieve_strict_semantic_matches()` | 近乎精确的 DSD 匹配 (用于调和) |

**记忆调和**: 快速启发式 (精确匹配、子串包含) → LLM 判断 → replace / merge / add

### 3. ExecuteModule — 执行模块
`modules/execute_module.py`

| 方法 | 功能 |
|---|---|
| `answer_direct(question)` | 直接推理 (无检索) |
| `answer_guided(question, principle_text)` | 经验引导推理 |
| `answer_baseline(question)` | 基线推理 (控制 Prompt) |
| `check_answer(generated, expected)` | `\boxed{}` 提取 + 标准化比较 |

`URLRouter`: 轮询负载均衡 + 冷却/失败跟踪

### 4. Stage1Sampler — 采样器
`modules/stage1_sampler.py`

- **K 次试验** (K=10): 每题独立采样 K 次
- **早停**: 连续 5 次正确 → Master
- **三区分类**: Master (SR=1.0) / Learning (0<SR<1) / Panic (SR=0)
- **四级分箱**: Level 1 (≥0.75) → Level 4 (<0.25)
- 支持 checkpoint/resume

### 5. Stage1Miner — 挖掘器
`modules/stage1_miner.py`

- 从 Learning 区样本中挖掘难负样本对 (正确 vs 错误轨迹)
- 多级步骤分割: 编号 → 双换行 → 句子边界 → 全文
- 数学原子提取: 数字、LaTeX 命令、运算符、函数名
- 发散点检测: `divergence_score()` → 文本相似度 + 数学原子重叠
- LLM 差异分析 → 打包为 learning_packet

### 6. RecursiveRepairProtocol — 递归修复
`modules/stage2_repair.py`

```
新原则 → 邻居任务验证 (3-5 个相似问题)
    ├── 通过率 ≥ 阈值 → 入库
    └── 通过率 < 阈值 → 诊断失败原因 → 修补 → 再验证
         (递归, 最多 3 层)
```

---

## Benchmark 评测系统

### 测试集 (6 个)

| Benchmark | 题目数 | 来源 |
|---|---|---|
| DeepScaler | 400 | 原始训练数据 |
| MATH500 | 500 | MATH 竞赛题 |
| GSM8K | 1,319 | 小学数学应用题 |
| BBH Logic | 2,437 | BIG-Bench Hard 逻辑子集 |
| BBH Math | 750 | BIG-Bench Hard 数学子集 |
| SVAMP | 1,000 | 结构变体数学题 |
| General365 | 720 | 通用 365 题 |

### 评测模式

| 模式 | 说明 |
|---|---|
| `base` | 单次直接推理, 无检索, 无重试 |
| `direct` | 直接推理, 无检索, 有重试 |
| `baseline` | 基线 Prompt 引导, 无检索 |
| `guided` | 记忆检索 + 经验引导 (完整流程) |

### 模型端口配置 (`benchmarks/config.py`)

| 模型 | GPU | 端口 |
|---|---|---|
| `llama` | 0-3 | 8000-8003 |
| `qwen` | 4-7 | 8004-8007 |
| `llama_8gpu` | 0-7 | 8100-8107 |
| `qwen_8gpu` | 0-7 | 8100-8107 |

### 快速命令

```bash
# 部署 (8 卡单模型, 给嵌入模型留显存)
GPU_MEMORY_UTILIZATION=0.75 MAX_NUM_SEQS=64 \
  bash api_call/run.sh deploy.conf llama 0 1 2 3 4 5 6 7 --daemon

# 跑全部 benchmark (8 卡轮询)
bash benchmarks/run_all.sh --8gpu llama

# 单独跑某个 benchmark
bash benchmarks/run_one.sh math500 llama_8gpu

# 停止部署
bash api_call/stop.sh
```

---

## 部署系统 (`api_call/`)

| 脚本 | 功能 |
|---|---|
| `run.sh` | 启动 vLLM 实例, 支持 `--daemon` 后台模式 |
| `stop.sh` | 停止全部 vLLM 进程 (PID 文件 + 端口释放 + Ray 清理) |
| `test.sh` | 健康检查 (发送测试请求到所有端口) |

**vLLM 参数** (`deploy.conf`):
```
GPU_MEMORY_UTILIZATION=0.92   # 显存利用率 (8-GPU 模式降为 0.75)
MAX_NUM_SEQS=128              # 单卡最大并发序列
MAX_MODEL_LEN=32768           # 最大上下文长度
DTYPE=bfloat16
--enforce-eager --enable-prefix-caching --trust-remote-code
```

---

## 嵌入与检索系统

### 嵌入模型加载流程

```
MEMUA_EMBEDDING_DEVICE 环境变量:
  "auto" (默认) → 先试 cuda, 失败则 CPU
  "cuda"         → 仅 GPU
  "cpu"          → 仅 CPU

模型搜索顺序:
  1. /home/models/models--BAAI--bge-m3/snapshots/...
  2. memory/metadata/hf_cache/...
  3. HuggingFace Hub (BAAI/bge-m3)
```

### 检索流程 (推理对齐模式)

```
1. 查询语义 → 3 维字段独立 FAISS 召回 (每维 top-K)
2. 候选池合并, 过滤: 至少 2 个面超过阈值 (threshold=0.55)
3. 结构分: facet_min × facet_mean × facet_balance × support_ratio
4. 质量分: overall_pass_ratio × neighbor_pass_ratio × source_pass
5. 最终分 = 0.72 × 结构分 + 0.28 × 质量分
6. 按 (final_score, structure_score, relevance_score) 排序
```

### 记忆去重阈值

```
DUPLICATE_THRESHOLD = 0.9   → replace_existing
UPDATE_THRESHOLD    = 0.75  → merge_with_existing
< 0.75                      → add_as_new
```

---

## 消融实验

### No Learning Zone
`ablation/no_learning_zone_collection/`
- `collect_all_samples=True`: 全量采样 (Master + Learning + Panic) 都进入下游
- 测试学习区三分类是否必要

### Single Vector
`ablation/single_vector_principle/`
- 用单一整体嵌入替代 3 维 DSD 分解
- `retrieval_method=single`: LLM 生成 holistic description → 嵌入 → KNN

---

## 关键配置环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MEMUA_EMBEDDING_DEVICE` | `auto` | 嵌入模型设备 (auto/cuda/cpu) |
| `MEMUA_EMBEDDING_MODEL` | `BAAI/bge-m3` | 嵌入模型名 |
| `MEMUA_EMBEDDING_MODEL_PATH` | (空) | 本地嵌入模型路径 |
| `MEMUA_FORCE_EMBEDDING` | `1` | 强制使用真实嵌入 (不允许 hash 降级) |
| `GPU_MEMORY_UTILIZATION` | `0.92` | vLLM 显存利用率 |
| `MAX_NUM_SEQS` | `128` | vLLM 单卡最大序列数 |
| `BENCH_CONCURRENCY_8GPU` | `128` | 8-GPU benchmark 并发数 |
