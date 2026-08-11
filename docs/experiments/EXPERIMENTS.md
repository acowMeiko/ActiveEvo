# MemMetaV3 实验设计文档

## 概览

四组对比实验，回答框架的核心价值问题。所有实验共享同一测试集和推理模型，仅改变记忆构建/检索策略。

---

## 实验一：Framework Ablation（框架消融）

> **回答**：整个框架值不值得做？

| 条件 | 描述 |
|------|------|
| **base** | 纯零-shot，无任何检索 |
| **base+raw** | 用原题文本做单向量 dense 检索，top-k 经验注入 |
| **base+structure** | 三维结构化检索（无门控），top-k 经验注入 |
| **full** | 三维检索 + 三维门控 + 多经验引导 prompt（完整方法） |

**指标**：
- Accuracy（最终准确率）
- Pass@1（首次尝试准确率）
- Misleading Rate（用了检索但做错的比例 / 总检索使用数）

**运行**：
```bash
bash experiments/exp1_ablation/run.sh
```

**预期结论**：full > structure > raw > base，说明每个模块都有正贡献。

---

## 实验二：Experience Screening（经验筛选）

> **回答**：learning zone 的经验是否更有价值？不是所有样本都值得学。

| 条件 | Memory 来源 |
|------|-------------|
| **full_data** | 所有采样题（含 comfort + panic zone）都走经验生成 |
| **learning_zone** | 只用 early_stop（不稳定）题的经验 |
| **comfort_included** | learning_zone + all_correct 题 |
| **panic_included** | learning_zone + all_wrong 题 |

**指标**：
- Accuracy（最终任务性能）
- Memory Size（各条件下的记忆库规模）

**运行**：
```bash
bash experiments/exp2_screening/run.sh
```

**预期结论**：learning_zone ≥ full_data，说明 comfort/panic zone 的经验若加入反而稀释质量或引入噪声。

---

## 实验三：Retrieval Ablation（检索消融 — 理解 × 联想组合拳）

> **回答**：结构化理解和联想门控各带来多少提升？

| 条件 | 描述 |
|------|------|
| **raw_dense** | 原题单向量 dense 检索（无结构化拆解） |
| **structure** | 三维结构化检索（无门控） |
| **structure_gate** | 三维检索 + 三维硬门控（完整方法） |

**指标**：
- Accuracy（最终准确率）
- Usable Ratio（检索后可用经验占总检索的比例）
- Average Retrieval Score（平均检索相关度）
- Misleading Rate（误导率）

**运行**：
```bash
bash experiments/exp3_retrieval/run.sh
```

**预期结论**：
- raw → structure：结构化拆解提升检索精度（理解能力）
- structure → structure_gate：门控过滤噪声，usable ratio 显著提升（联想能力）

---

## 实验四：Multi-round Evolution（多轮演化）

> **回答**：agent 是否在多轮中不断演化，而非一次性记住？

| 条件 | 描述 |
|------|------|
| **frozen** | 只用 Round 1 的 memory snapshot，在所有轮评估（记忆不增长） |
| **incremental** | 用 Round N 的 snapshot 评估（记忆随轮次增长） |

**输出**：
- 轮次-准确率曲线（frozen vs incremental）
- 记忆规模增长表
- Delta（incremental - frozen）

**运行**：
```bash
bash experiments/exp4_evolution/run.sh
```

**预期结论**：incremental 随轮次提升且与 frozen 差距增大 → 新轮次学到的经验确实有用，不是单靠第一轮记忆。

---

## 文件结构

```
experiments/
├── exp1_ablation/
│   ├── __init__.py
│   ├── run.py          # 4 条件统一评估脚本
│   └── run.sh          # Shell 启动器
├── exp2_screening/
│   ├── __init__.py
│   ├── run.py          # 经验筛选 + 构建 filtered Memory + 评估
│   └── run.sh
├── exp3_retrieval/
│   ├── __init__.py
│   ├── run.py          # 检索消融 + 可用比例统计
│   └── run.sh
└── exp4_evolution/
    ├── __init__.py
    ├── run.py          # Frozen vs Incremental 多轮对比
    └── run.sh
```

## 通用环境变量

所有 `run.sh` 支持以下环境变量覆盖默认值：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | MetaEVO | 推理模型名 |
| `PORTS` | 8000 8001 8002 8003 | vLLM 端口列表 |
| `TIMEOUT` | 180 | LLM 超时 (秒) |
| `CONCURRENCY` | 100 | 并发评估数 |
| `TOP_K` | 5 | 最大检索经验数 |
| `THRESHOLD` | 0.55 | 三维门控阈值 |
| `MEMORY_DIR` | memory | Memory 根目录 |
| `TEST_INPUT` | data/test/math_500_test.jsonl | 测试集路径 |

示例：
```bash
PORTS="8000 8001" CONCURRENCY=50 bash experiments/exp1_ablation/run.sh
```
