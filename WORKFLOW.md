# ActiveEvo 可复现流程

## 0. 配置边界

- `deploy.conf`：模型路径、GPU、端口基数、vLLM 参数和服务端 seed。
- `config.env`：Stage 2/3 并发、阈值、top-k、token 上限和共享嵌入模型。
- `api_call/logs/active_deployment.env`：最近一次成功部署的实际模型、GPU、端口和 PID，由脚本生成，不手工修改。

## 1. 部署与检查

推荐直接使用部署目录中的固定入口：

```bash
bash api_call/run.sh deploy.conf
bash api_call/test.sh deploy.conf llama
```

如果需要一个短的兼容部署入口，可以使用：

```bash
bash scripts/deploy/deploy_model.sh deploy.conf llama 2 5 7 --daemon
```

## 2. Stage 1：学习区与轨迹对

主入口：

```bash
python pipeline.py --stage 1 --dataset dpo_level1 --ports 8100 8101 8102 8103
```

辅助入口：

```bash
python scripts/stage1/build_learning_packets.py --help
python scripts/stage1/run_stage1_llama.py --help
python scripts/stage1/run_stage1_dual.py --help
```

规范输出写入 `learning_packet/`；统一 backbone 采样结果仍写入 `output/stage1/`。

## 3. Stage 2：Memory 构建

全 Level 运行：

```bash
bash scripts/stage2/run_all_levels.sh
```

单 Level 或小样本：

```bash
python pipeline.py --stage 2 --dataset dpo_level2 --stage2-limit 10 \
  --stage2-run-tag smoke10 --memory-dir memory/v3_smoke --ports 8100 8101 8102 8103

python pipeline.py --stage 2 --dataset dpo_level2 --stage2-limit 30 \
  --stage2-run-tag pilot30 --memory-dir memory/v3_pilot --ports 8100 8101 8102 8103
```

主要输出：

```text
memory/dpo_level*/00_dsd_registry/
memory/dpo_level*/01_candidates/
memory/dpo_level*/04_gate_results/
memory/dpo_level*/05_dev_paired_validation/
memory/dpo_level*/07_holdout_paired_validation/
memory/dpo_level*/08_memory_decisions/
memory/dpo_level*/09_store_commit/
memory/v3/{principles,indices,metadata,snapshots,checkpoints,staging}/
```

只有 Holdout 至少一个 `gain`、零 `loss` 且净收益为正的 Principle 才会成为 `active`。

## 4. Stage 3：检索增强与门控评测

全测试入口：

```bash
bash scripts/tests/run_all_tests.sh
```

单数据集入口：

```bash
python pipeline.py --stage 3 --dataset dpo_level1 \
  --test-input data/test/deepscaler.json \
  --memory-dir memory/v3 --ports 8100 8101 8102 8103
```

当前论文专项实验位于 `experiments/`，每个子目录包含自己的 README、固定输入、运行脚本和结果摘要。

## 5. 停止

```bash
bash api_call/stop.sh deploy.conf
```

如需显式指定本轮卡号：

```bash
bash api_call/stop.sh deploy.conf llama 0 1 2 3
```

## 6. 维护与清理

- 数据修复与重建：`tools/maintenance/`
- DeepSeek 标注与配对分析：`tools/`
- benchmark 和官方测试集处理：`benchmarks/`、`scripts/prepare_official_datasets.py`
- 项目结构说明：`docs/PROJECT_STRUCTURE.md`
- 可复现记录说明：`docs/REPRODUCIBILITY.md`
