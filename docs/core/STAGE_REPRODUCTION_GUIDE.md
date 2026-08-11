# MemUA Stage Reproduction Guide

这份文档给的是当前代码下可直接执行的复现顺序，按阶段组织，不再混入旧流程。

主要入口：

- [pipeline.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/pipeline.py:1)
- [api_call/run.sh](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/api_call/run.sh)
- [api_call/stop.sh](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/api_call/stop.sh)
- [run_stage1_round_robin.sh](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/run_stage1_round_robin.sh)
- [run_stage3_round_robin.sh](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/run_stage3_round_robin.sh)

## 1. 复现前准备

### 1.1 环境

默认使用：

```bash
/home/czhu/miniconda3/envs/hcz/bin/python
```

### 1.2 数据

训练集主入口：

- `data/input/V3/dpo_level_1.json`
- `data/input/V3/dpo_level_2.json`
- `data/input/V3/dpo_level_3.json`
- `data/input/V3/dpo_level_4.json`

常用测试集：

- `data/test/deepscaler.json`
- `data/test/benchmarks_full/gsm8k_stage3.json`
- `data/test/benchmarks_full/math500_stage3.json`
- `data/test/benchmarks_split/bbh_logic_stage3.json`
- `data/test/benchmarks_split/bbh_math_stage3.json`
- `data/test/official/svamp/processed/test.json`
- `data/test/official/general365/processed/test.json`

## 2. 启动模型服务

### 2.1 启动 Qwen-14B

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
bash api_call/run.sh qwen14b
```

### 2.2 启动 Llama-8B

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
bash api_call/run.sh llama8b
```

### 2.3 停服务

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
bash api_call/stop.sh
```

## 3. 单独跑 Stage 1

### 3.1 跑某一轮

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
/home/czhu/miniconda3/envs/hcz/bin/python pipeline.py \
  --stage 1 \
  -d dpo_level2 \
  --ports 8100 8101 8102 8103
```

### 3.2 常用输出

- `output/stage1/dpo_level2/01_sampling/sampling_results.json`
- `output/stage1/dpo_level2/02_mining/hard_negative_pairs.json`
- `output/stage1/dpo_level2/04_learning_packets/learning_packets.json`

## 4. 单独跑 Stage 2

### 4.1 标准跑法

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
/home/czhu/miniconda3/envs/hcz/bin/python pipeline.py \
  --stage 2 \
  -d dpo_level2 \
  --ports 8100 8101 8102 8103
```

### 4.2 当主模型占满 GPU 时

如果 `bge-m3` 无法在 GPU 上加载，可以显式把 embedding 放到 CPU：

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
MEMUA_EMBEDDING_DEVICE=cpu \
/home/czhu/miniconda3/envs/hcz/bin/python pipeline.py \
  --stage 2 \
  -d dpo_level2 \
  --ports 8100 8101 8102 8103
```

### 4.3 常用输出

- `output/stage2/dpo_level2/00_dsd_registry/question_dsd_registry.json`
- `output/stage2/dpo_level2/01_memory_bank/semantic_memory_bank.json`
- `output/stage2/dpo_level2/01_memory_bank/stage2_summary.json`
- `output/stage2/dpo_level2/02_vector_db/`

## 5. 单独跑 Stage 3

### 5.1 默认 deepscaler

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
/home/czhu/miniconda3/envs/hcz/bin/python pipeline.py \
  --stage 3 \
  -d dpo_level2 \
  --test-input data/test/deepscaler.json \
  --ports 8100 8101 8102 8103
```

### 5.2 自定义测试集

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
/home/czhu/miniconda3/envs/hcz/bin/python pipeline.py \
  --stage 3 \
  -d dpo_level2 \
  --test-input data/test/official/svamp/processed/test.json \
  --ports 8100 8101 8102 8103 \
  --threshold 0.55 \
  --top-k 5
```

### 5.3 常用输出

- `output/stage3/dpo_level2/01_guided_eval/principle_guided_test_results.json`
- `output/stage3/dpo_level2/01_guided_eval/principle_guided_test_summary.json`
- `output/stage3/curriculum/progress_state.json`

## 6. 跑完整 curriculum

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
/home/czhu/miniconda3/envs/hcz/bin/python pipeline.py \
  --levels 1 2 3 4 \
  --ports 8100 8101 8102 8103
```

或者用现成脚本：

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
bash run_full_pipeline.sh
```

## 7. 当前常用脚本怎么用

### 7.1 Stage 1 多端口

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
bash run_stage1_round_robin.sh
```

### 7.2 Stage 3 多端口

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
bash run_stage3_round_robin.sh
```

### 7.3 Level2 后台跑 Stage2 + Stage3

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
bash run_level2_stage23_nohup.sh
```

### 7.4 用重构后的 level2 memory 跑全测试集

```bash
cd /home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code
bash run_stage3_all_level2_after_refactor.sh
```

## 8. 两条最常见排障

### 8.1 Stage 2 卡在 embedding 模型

症状：

- `Retrieval embedding model failed to load`
- `bge-m3` 在 GPU 上 OOM

处理：

```bash
MEMUA_EMBEDDING_DEVICE=cpu
```

### 8.2 Stage 3 结果不对齐

要先确认：

1. `output/stage2/<dataset>/01_memory_bank/semantic_memory_bank.json` 已经生成
2. `output/stage2/<dataset>/02_vector_db/` 存在
3. `output/stage3/curriculum/progress_state.json` 是否需要先清空

如果你想做“独立重测”，要先删掉旧的 `progress_state.json`，否则会继承之前已经答对的题。
