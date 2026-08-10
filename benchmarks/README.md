# Benchmarks 评测模块

独立的 benchmark 评估工具，不依赖 AgentPipeline 主流程。

## 结构

```
benchmarks/
  config.py      # 模型 + benchmark 配置
  matcher.py     # 多格式答案匹配器
  evaluator.py   # 单次评估器 (Plan + Memory + Execute)
  run_suite.py   # 批量运行器 (所有 model × benchmark)
  deploy.sh      # 双模型 vLLM 部署
  summary.py     # 结果汇总 + BBH 合并准确率
```

## 评测范围

| Benchmark | 数据集 | 题量 | 答案格式 |
|-----------|--------|------|----------|
| DeepScaler | data/test/deepscaler.json | 400 | LaTeX + boxed |
| MATH500 | data/test/benchmarks/math500_stage3.json | 500 | LaTeX + boxed |
| GSM8K | data/test/benchmarks/gsm8k_stage3.json | 1319 | 数字 |
| BBH Logic | data/test/benchmarks_split/bbh_logic_stage3.json | 2437 | True/False |
| BBH Math | data/test/benchmarks_split/bbh_math_stage3.json | 750 | 数字 |
| SVAMP | data/test/official/svamp/processed/test.json | 1000 | 数字 (容差) |
| General365 | data/test/official/general365/processed/test.json | 720 | 单选/多选/文本/数字 |

## 模型端口

| 模型 | GPU | 端口 |
|------|-----|------|
| LLaMA | 0-3 | 8000-8003 |
| Qwen | 4-7 | 8004-8007 |

## 使用方法

### 1. 部署模型

```bash
# 双模型全部部署
bash benchmarks/deploy.sh deploy.conf

# 仅部署一个模型
bash benchmarks/deploy.sh deploy.conf llama
bash benchmarks/deploy.sh deploy.conf qwen

# 查看状态
bash benchmarks/deploy.sh deploy.conf --status

# 停止
bash benchmarks/deploy.sh deploy.conf --stop
```

### 2. 运行评测

```bash
# 全部 benchmark + 全部模型 (guided 模式)
python benchmarks/run_suite.py --models llama qwen --mode guided --output output/benchmarks

# 仅 LLaMA, baseline 模式 (无经验引导)
python benchmarks/run_suite.py --models llama --mode baseline --output output/benchmarks

# 单个 benchmark 单次评估
python benchmarks/evaluator.py \
  --test-file data/test/benchmarks/math500_stage3.json \
  --memory-dir memory/llama \
  --ports 8000 8001 8002 8003 \
  --model llama \
  --mode guided \
  --output output/benchmarks/llama/math500
```

### 3. 查看结果

```bash
# 终端表格
python benchmarks/summary.py --results-dir output/benchmarks

# Markdown 表格
python benchmarks/summary.py --results-dir output/benchmarks --format markdown

# JSON 格式
python benchmarks/summary.py --results-dir output/benchmarks --format json
```

## 模式说明

| 模式 | 说明 |
|------|------|
| `guided` | 从 Memory 检索经验 principle, 引导模型答题 |
| `baseline` | 使用论文对照组 prompt, 不带经验引导 |
| `direct` | 直接答题, 不带经验引导和 baseline prompt |

## 输出

每个 benchmark 的 `output_dir` 下生成:

- `eval_results.json` — 逐题详细结果
- `eval_summary.json` — 汇总统计

BBH 合并准确率在 combined_summary.json 中单独列出。
