# ActiveEvo 可复现流程

## 0. 配置边界

- `deploy.conf`：模型路径、GPU、端口基数、vLLM 参数和服务端 seed。
- `config.env`：Stage 2/3 并发、阈值、top-k、token 上限和共享嵌入模型。
- `api_call/logs/active_deployment.env`：最近一次成功部署的实际模型、GPU、端口和 PID，由脚本生成，不手工修改。

## 1. 部署与检查

```bash
bash deploy_model.sh deploy.conf --daemon
bash api_call/test.sh deploy.conf llama
```

默认配置为 GPU 0–3，端口 8100–8103。命令行可以临时覆盖 GPU；实际拓扑仍会写入运行清单。

```bash
bash deploy_model.sh deploy.conf llama 2 5 7 --daemon
```

## 2. Stage 1：学习区与轨迹对

主入口按任务选择：

```bash
python pipeline.py --stage 1 --dataset dpo_level1
python build_learning_packets.py --help
python run_stage1_llama.py --help
```

规范输出写入 `learning_packet/`；统一 backbone 采样结果仍写入 `output/stage1/`。

## 3. Stage 2：Memory 构建

```bash
bash run_stage2_all_levels.sh
```

输入为 `learning_packet/dpo_level*/04_learning_packets/learning_packets.json`，输出为：

```text
memory/dpo_level*/00_dsd_registry/               # 题目结构注册表与严格三维邻居
memory/dpo_level*/01_candidates/                 # Principle 候选与四视图
memory/dpo_level*/04_gate_results/                # 批量 Application Gate
memory/dpo_level*/05_dev_paired_validation/       # Dev 配对验证
memory/dpo_level*/07_holdout_paired_validation/   # 未参与修复的 Holdout 验证
memory/dpo_level*/08_memory_decisions/            # active/provisional/rejected
memory/dpo_level*/09_store_commit/                # 版本提交记录
memory/v3/{principles,indices,metadata,snapshots,checkpoints,staging}/
```

只有 Holdout 至少一个 `gain`、零 `loss` 且净收益为正的 Principle 才会成为
`active`。旧 `memory/principles/` 只读导入为 provisional，不会被覆盖。

先运行小规模流程时使用隔离的 Store 和输出标签：

```bash
python pipeline.py --stage 2 --dataset dpo_level2 --stage2-limit 10 \
  --stage2-run-tag smoke10 --memory-dir memory/v3_smoke --ports 8100 8101 8102 8103

python pipeline.py --stage 2 --dataset dpo_level2 --stage2-limit 30 \
  --stage2-run-tag pilot30 --memory-dir memory/v3_pilot --ports 8100 8101 8102 8103
```

## 4. Stage 3：检索增强与门控评测

```bash
bash run_all_tests.sh
```

当前论文专项实验位于 `experiments/`，每个子目录包含自己的 README、固定输入、运行脚本和结果摘要。不要从目录名推断当前 GPU；运行脚本读取统一部署配置。

## 5. 论文证据

```bash
python paper/evidence/scripts/extract_evidence.py
```

该脚本直接读取 `output/` 和 `memory/`，生成小型 summary 与 `source_manifest.json`，不再复制数百 MB 的原始 JSON。

## 6. 停止

```bash
bash api_call/stop.sh deploy.conf
```

如需显式指定本轮卡号：

```bash
bash api_call/stop.sh deploy.conf llama 0 1 2 3
```

## 7. 维护与清理

- 数据修复与重建：`tools/maintenance/`
- DeepSeek 标注与配对分析：`tools/`
- 引文工具：`tools/paper/`
- 跨机器同步：`tools/sync/`
- 项目清理器默认先归档历史数据：`python tools/maintenance/cleanup_project.py --help`
