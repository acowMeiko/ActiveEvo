# ActiveEvo / SUAgent

这是论文实验代码的活动目录。主流程围绕三类持久化产物组织：Stage 1 生成学习样本，Stage 2 构建 Memory Store，Stage 3 完成检索、门控和配对评测。

## 最短使用路径

```bash
# 1. 在 deploy.conf 中设置模型、GPU 和端口基数
bash deploy_model.sh deploy.conf --daemon

# 2. 检查实际部署
bash api_call/test.sh deploy.conf llama

# 3. 构建 Memory 或运行测试
bash run_stage2_all_levels.sh
bash run_all_tests.sh

# 4. 停止本次实际部署
bash api_call/stop.sh deploy.conf
```

当前默认端口规则是 `DEPLOY_PORT_BASE + GPU_ID`。`deploy.conf` 中基数为 8100，因此 GPU 0–3 对应 8100–8103。启动成功后，实际模型、GPU、端口和 PID 写入 `api_call/logs/active_deployment.env`；测试与停止脚本优先读取该文件。

## 日常手动运行命令

以下命令均从项目根目录执行。GPU、模型路径和端口基数只需要在
`deploy.conf` 中维护；当前配置使用 Llama、GPU 0–3 和端口 8100–8103。

### 1. 部署、检查与停止模型

```bash
# 后台部署；脚本会等待所有端口通过健康检查
bash deploy_model.sh deploy.conf --daemon

# 检查实际运行清单中的全部模型端点
bash api_call/test.sh deploy.conf llama

# 停止本次部署记录中的全部实例
bash api_call/stop.sh deploy.conf
```

### 2. Stage 1：生成 Learning Packets

```bash
python pipeline.py \
  --stage 1 \
  --dataset dpo_level1 \
  --model llama \
  --concurrency 64 \
  --diff-concurrency 64 \
  --ports 8100 8101 8102 8103
```

Stage 1 输出位于 `learning_packet/` 和 `output/stage1/`。如果
`deploy.conf` 中更换了 GPU，请把端口改成 `8100 + GPU_ID`，或者查看
`api_call/logs/active_deployment.env` 中的 `ACTIVE_PORTS`。

### 3. Stage 2：重新生成并验证 v3 Memory

```bash
# 按 config.env 中的 LEVELS 顺序运行全部 level
bash run_stage2_all_levels.sh
```

需要让 Stage 2 在后台运行时：

```bash
mkdir -p logs/stage2
nohup bash run_stage2_all_levels.sh > logs/stage2/stage2_launcher.log 2>&1 &
echo $!
```

只运行 10 条 Smoke Test：

```bash
python pipeline.py \
  --stage 2 \
  --dataset dpo_level2 \
  --stage2-limit 10 \
  --stage2-run-tag smoke10 \
  --memory-dir memory/v3_smoke \
  --model llama \
  --stage2-concurrency 8 \
  --ports 8100 8101 8102 8103
```

正式 Store 位于 `memory/v3/`；旧 `memory/principles/` 不会被覆盖。

### 4. Stage 3：运行检索增强评测

```bash
# 对所有已配置测试集运行 Stage 3
bash run_all_tests.sh
```

只运行一个数据集时：

```bash
python pipeline.py \
  --stage 3 \
  --dataset dpo_level2 \
  --test-input data/test/deepscaler.json \
  --memory-dir memory/v3 \
  --model llama \
  --eval-concurrency 32 \
  --ports 8100 8101 8102 8103
```

### 5. 离线检查 Stage 2 代码

```bash
python tests/run_stage2_v3_contracts.py
DRY_RUN=1 bash run_all_tests.sh
```

## 目录职责

| 目录 | 职责 | 是否为生成物 |
|---|---|---|
| `api_call/` | vLLM 启动、健康检查、测试、停止和运行清单 | 日志/PID 是 |
| `core/` | Agent 总编排、API 池、配置、运行指标和 Checkpoint | 否 |
| `modules/` | Stage 1/2/3 的采样、规划、Memory、门控和 reranker 实现 | 否 |
| `benchmarks/` | direct/guided 基线评测、答案匹配和汇总 | 结果是 |
| `data/`、`test_sets/` | 训练题、测试题和官方数据转换结果 | 部分是 |
| `learning_packet/` | Stage 1 的规范输出 | 是，保留 |
| `memory/` | Stage 2 分析产物；长期 Store 位于 `memory/v3/`，旧库只读保留 | 是，保留 |
| `output/` | Stage 3、benchmark 和消融的规范结果 | 是，保留 |
| `experiments/` | 当前 ICASSP 实验协议、固定子集和分析报告 | 是，保留 |
| `ablation/` | single-vector、no-learning-zone 等论文消融 | 是，保留 |
| `error_analysis/` | 样本级错误、principle 与检索分析 | 是，保留 |
| `paper/` | LaTeX、图、规划和去重后的论文证据摘要 | PDF/证据是 |
| `scripts/` | 稳定批处理脚本 | 否 |
| `tools/` | 分析、标注、维护、引文和同步工具 | 否 |
| `tests/` | 单元测试和 smoke test | 否 |
| `docs/` | 架构、参数、复现说明和项目地图 | 否 |

完整的“文件 → 流程 → 输入/输出”说明见 [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md)，操作顺序见 [WORKFLOW.md](WORKFLOW.md)。

## 数据流

```text
data
  └─ Stage 1: sampling / hard-negative mining
       └─ learning_packet
            └─ Stage 2: principle generation / paired causal validation / versioning
                 └─ memory/v3
                      └─ Stage 3: retrieve / application gate / solver
                           ├─ output
                           ├─ experiments
                           └─ paper/evidence summaries
```

`memory/metadata/hf_cache` 不再保存模型副本。嵌入模型统一来自 `/home/share/models/modelscope/BAAI/bge-m3`。

## 历史归档

2026-08-09 清理前的旧 Memory、旧 Output、研究过程记录和论文原始证据位于：

```text
/home/czhu/document/ActiveEvo_ICASSP_2027/archive/code_cleanup_2026-08-09/
```

其中 `cleanup_manifest.csv` 记录原路径、归档路径、大小和 SHA-256；活动目录不再保存 `*-before`、重复 evidence 原始 JSON 或运行日志。
