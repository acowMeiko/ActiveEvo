 # MemUA Current Architecture

这份文档只描述当前仓库里真实在运行的实现，不再沿用旧版 `MemMetaV3` 或设计草稿里的表述。

核心入口：

- [pipeline.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/pipeline.py:1)
- [core/agent_pipeline.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/core/agent_pipeline.py:233)
- [core/config.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/core/config.py:1)
- [output_paths.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/output_paths.py:1)

## 1. 总体结构

当前 MemUA 是一个三阶段、可按 `dpo_level1-4` 迭代运行的外部记忆式推理系统：

1. `Stage 1`：从训练集里找出不稳定但可解的问题，并封装成 `learning_packets`
2. `Stage 2`：对 `learning_packets` 做 DSD 建模、principle 生成、source/neighbor 验证、memory reconciliation、入库和索引构建
3. `Stage 3`：对测试题做语义抽取、memory 检索、guided answering 和 progressive evaluation

当前主调度器是 [AgentPipeline](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/core/agent_pipeline.py:233)，其中：

- `plan`：规划与抽象
- `memory`：记忆存储与检索
- `execute`：采样、答题、答案校验

## 2. 目录与数据流

常用目录：

- 训练数据：`data/input/V3/`
- 测试数据：`data/test/`
- 运行输出：`output/`
- 日志：`logs/`
- 记忆目录：默认 `memory/`

主数据流：

```text
data/input/V3/dpo_level_X.json
  -> Stage 1 sampling_results.json
  -> Stage 1 hard_negative_pairs.json
  -> Stage 1 learning_packets.json
  -> Stage 2 question_dsd_registry.json
  -> Stage 2 semantic_memory_bank.json
  -> Stage 2 02_vector_db/
  -> Stage 3 principle_guided_test_results.json
  -> Stage 3 principle_guided_test_summary.json
```

统一输出路径在 [output_paths.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/output_paths.py:1)。

## 3. Stage 1 结构

入口：

- [AgentPipeline.stage1_learn](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/core/agent_pipeline.py:256)
- [sample_unstable_questions.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/sample_unstable_questions.py:1)
- [modules/execute_module.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/modules/execute_module.py:1)

### 3.1 输入

- `data/input/V3/dpo_level_1.json` 到 `dpo_level_4.json`
- 本地 vLLM 服务，通常走 `8000-8003`

### 3.2 做的事情

1. 多次采样同一题
2. 将题划分为 `master / learning / panic`
3. 从 `learning` 样本里挖 hard negative
4. 为正负轨迹提取三维语义
5. 对正负轨迹做 diff
6. 封装成 `learning_packets.json`

### 3.3 关键技术

- `asyncio` 并发采样
- 端口轮询与失败熔断
- 文本清洗、截断检测、答案抽取
- LLM 语义抽取与 diff 分析

## 4. Stage 2 结构

入口：

- [AgentPipeline.stage2_memorize](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/core/agent_pipeline.py:435)
- [modules/dsd_module.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/modules/dsd_module.py:1)
- [modules/memory_module.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/modules/memory_module.py:1)
- [modules/memory.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/modules/memory.py:1)
- [modules/plan_module.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/modules/plan_module.py:1)

### 4.1 DSD registry

每个 packet 会被编码成三维语义：

- `core_skill`
- `context_constraints`
- `task_objective`

DSD registry 会做两件事：

1. 保存当前轮所有问题的三维语义记录
2. 预计算语义邻居，供后续 neighbor validation 使用

### 4.2 Principle 选择链路

当前 Stage 2 的真实链路是：

1. 读取 `learning_packets`
2. 构建 DSD registry
3. 对每个 packet 最多生成 `3` 轮 principle candidate
4. 先做 source question 验证
5. 再做 neighbor gate 验证
6. 通过 gate 的 principle 先进入 `pending_memory_candidates`
7. 再做严格 memory reconciliation
8. 最终写入 memory，构建索引并保存快照

### 4.3 新的 memory reconciliation

当前代码已经不是“过 gate 就直接入库”。

在 [core/agent_pipeline.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/core/agent_pipeline.py:700) 附近，当前逻辑是：

1. 用 `retrieve_strict_semantic_matches()` 去已有 memory 里找近乎同题的旧记录
2. 要求三维语义都超过 `0.95`
3. 若找到候选，用 LLM 判断：
   - `replace_existing`
   - `merge_with_existing`
   - `add_as_new`

这一步的作用是减少 memory 里“同题同义 principle”越来越多的问题。

## 5. Memory 结构与 FAISS

底层记忆管理在 [modules/memory.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/modules/memory.py:1)。

### 5.1 Memory 存什么

每条 active principle 至少包含：

- `memory_id`
- `question`
- `core_skill`
- `context_constraints`
- `task_objective`
- `principle`
- `source_round`
- `source_dataset`
- `status`

### 5.2 Memory 如何保存

默认 memory 目录布局是：

```text
memory/
  principles/principles.json
  indices/core_skill.index
  indices/context_constraints.index
  indices/task_objective.index
  metadata/memory_data.json
  metadata/stats.json
  snapshots/round_1/
  snapshots/round_2/
  ...
```

对应实现：

- 持久化：`load()` / `save()` / `snapshot()`
- 索引保存：`save_indices()`
- 索引恢复：`_load_indices()`

### 5.3 FAISS 在当前代码里怎么用

当前不是单索引，而是三套独立的 `IndexFlatIP`：

- `core_skill.index`
- `context_constraints.index`
- `task_objective.index`

流程：

1. 用 `SentenceTransformer` 对三维文本分别编码
2. 对向量做 L2 normalize
3. 以 cosine 等价的内积形式写入 `faiss.IndexFlatIP`
4. 将三个索引分别保存到 `indices/`
5. 同时把 `memory_data.json` 存下来，兼容在线检索器读取

相关实现：

- [build_indices](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/modules/memory.py:433)
- [save_indices](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/modules/memory.py:454)
- [_load_indices](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/modules/memory.py:503)
- [_build_faiss_indices](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/modules/memory.py:615)

### 5.4 检索规则

当前检索不是“加权重排 importance”的版本，而是：

1. 三维都要过 `threshold`
2. 通过后按 `(core + context + target) / 3` 排序
3. 返回 `top_k`

对应实现是 [Memory.retrieve](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/modules/memory.py:318)。

## 6. Stage 3 结构

入口：

- [AgentPipeline.stage3_evaluate](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/core/agent_pipeline.py:811)
- [evaluate_principle_guided_test.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/evaluate_principle_guided_test.py:1)
- [retrieve_principles.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/retrieve_principles.py:1)

### 6.1 做的事情

1. 对测试题提取三维语义
2. 从 memory 做三维门控检索
3. 组装 guided prompt
4. 最多做 3 次尝试
5. 统计 `first_attempt_accuracy` 和 `accuracy`
6. 更新 progressive `progress_state.json`

### 6.2 Progressive evaluation

当前代码会记录 `output/stage3/curriculum/progress_state.json`。

这意味着：

- 第一次评估会测全量
- 后续轮次会继承已经答对的题
- summary 中的最终 `accuracy` 是累计意义上的

## 7. 模型与嵌入技术

### 7.1 弱模型

弱模型通过本地 vLLM OpenAI-compatible API 调用，默认 URL 来自 [core/config.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/core/config.py:137)。

### 7.2 强模型

`PlanModule` 保留了外部强模型配置位，但当前主线大多仍以本地弱模型为主。

### 7.3 嵌入模型

当前默认嵌入模型是：

- `BAAI/bge-m3`

加载链路在：

- [embedding_utils.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/embedding_utils.py:1)
- [retrieve_principles.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/retrieve_principles.py:1)
- [modules/memory.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/modules/memory.py:629)

当前支持：

- 本地 cache 优先
- 在线下载回退
- 严格模式下禁止 hash fallback
- 可通过 `MEMUA_EMBEDDING_DEVICE=cpu` 强制放到 CPU

## 8. 当前最重要的几个实现事实

下面这些是当前代码最容易被旧文档误导的地方：

1. Stage 2 现在有严格 reconciliation，不是简单 dedup
2. 检索排序目前仍是三维平均分，不是 `importance_score` 主导
3. Stage 3 是 progressive evaluation，不是每轮完全独立横评
4. FAISS 保存的是三套分维索引，不是单个混合索引
5. memory 构建和在线检索共用同一套 `bge-m3` 语义空间
    load(memory_dir)                    # 加载已有记忆
    save(memory_dir)                    # 保存记忆
    snapshot(round_id)                  # 按轮次快照

    # 记忆写入 (自动去重)
    add(principle_item) → memory_id     # 添加新经验
    detect_similar(item, threshold)     # 检测相似项
    decide_action(new, similar) → str   # KEEP_NEW / UPDATE / DEPRECATE
    ingest_principle(item) → (id, action)  # 高级接口: 检测→决策→执行

    # 检索
    retrieve(query, top_k=5, threshold=0.55)  # 三维门控检索
    get_new_principles(since_round)     # 获取新增经验

    # 索引
    build_indices()                     # 构建 FAISS 三维索引
    save_indices()                      # 保存索引到磁盘
```

**外部记忆目录结构**:
```
memory/
├── principles/
│   └── principles.json         # 所有结构化经验 (含 round_id, status)
├── indices/
│   ├── core_skill.index        # FAISS IndexFlatIP
│   ├── context_constraints.index
│   └── task_objective.index
├── metadata/
│   ├── memory_data.json        # embeddings + 元数据
│   └── stats.json
└── snapshots/
    ├── round_1/
    ├── round_2/
    ├── round_3/
    └── round_4/
```

**去重策略**:
- sim > 0.9 → DEPRECATE (近似重复，不添加)
- sim > 0.75 → UPDATE_EXISTING (合并更新)
- else → KEEP_NEW (添加新经验)

**关键文件**:
- `modules/memory.py` — Memory 类
- `core/stage2_pipeline.py` — Stage 2 入口 (Step 21 入库, Step 22 建库)

---

## Stage 3: 联想驱动的检索与增量记忆演化

**核心目的**：基于测试问题进行三维语义匹配，从 Memory 库检索相关经验，引导推理。

**检索策略**:
- top_k = 5 (可配置)
- 三维门控: core_skill, context_constraints, task_objective 都 >= threshold (0.55)
- 所有通过门控的经验都注入 prompt
- 引导式语气: "Consider", "verify their relevance"

**多经验注入格式**:
```
[Experience 1 (relevance: 0.92)]:
When dealing with recursive sequences, verify boundary conditions...

[Experience 2 (relevance: 0.85)]:
For optimization under constraints, check feasibility first...
```

**Prompt 设计原则**:
1. 经验作为"软指导"而非"强制执行" — 使用 Consider/Verify 语气
2. 保持独立推理能力 — "Solve step by step from first principles"
3. 经验冲突时信任自身计算 — "trust your own calculation but double-check"

**关键文件**:
- `evaluate_principle_guided_test.py` — 评估入口 (支持 --memory-dir, --top-k)
- `retrieve_principles.py` — OnlinePrincipleRetriever (向量检索)
- `template/prompt_template.py` — GUIDED_ANSWER_PROMPT (引导式模板)

---

## 四轮迭代编排

`run_full_pipeline.sh` 编排完整循环:

```bash
for LEVEL in 1 2 3 4; do
    DATASET="dpo_level${LEVEL}"

    # Stage 1: 学习区挖掘
    python sample_unstable_questions.py --dataset $DATASET ...
    python hard_negative_mining.py --dataset $DATASET
    python generate_principle.py --dataset $DATASET ...

    # Stage 2: 记忆构建 (增量)
    python main.py -s 21 -d $DATASET --memory-dir memory --round-id $LEVEL
    python main.py -s 22 -d $DATASET --memory-dir memory --round-id $LEVEL

    # Stage 3: 检索增强评估
    python evaluate_principle_guided_test.py \
        --mode guided --memory-dir memory --top-k 5 ...
done
```

**增量记忆演化**:
```
Round 1: memory = [P1..P50]
Round 2: memory = [P1..P50] + [P51..P80]   (去重后)
Round 3: memory = [P1..P80] + [P81..P110]
Round 4: memory = [P1..P110] + [P111..P135]
```

---

## 文件改动清单

### 新增文件
| 文件 | 说明 |
|------|------|
| `modules/memory.py` | Memory 类 — 统一外部记忆管理器 |
| `core/field_names.py` | 三维语义字段名映射 (向后兼容) |
| `modules/principle_quality.py` | Principle Utility Score 评估器 |

### 重要修改
| 文件 | 改动 |
|------|------|
| `core/stage2_pipeline.py` | 重写为 Memory 类入口 |
| `evaluate_principle_guided_test.py` | 多经验检索 + Memory 适配器 + --top-k/--memory-dir |
| `run_full_pipeline.sh` | 四轮迭代循环编排 |
| `template/prompt_template.py` | 引导式语气 + 泛化字段名 |
| `generate_principle.py` | PrincipleGenerator (去 DPO 命名) |
| `output_paths.py` | stage1_principle_file (去 DPO 命名) |
| `retrieve_principles.py` | top_k 默认 5 |
| `main.py` | 新增 --memory-dir, --round-id; step 21/22 走 Memory 类 |
| `build_vector_db.py` | build_record_from_principle_record (去 DPO 命名) |

### 命名清理
- `PrincipleDPGenerator` → `PrincipleGenerator` (兼容别名保留)
- `build_dpo_item()` → `build_principle_record()`
- `stage1_dpo_file()` → `stage1_principle_file()` (deprecated wrapper 保留)
- `dpo_level1-4` 作为课程等级名保留不变
