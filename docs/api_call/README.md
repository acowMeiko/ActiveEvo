# api_call/ — vLLM 模型部署指南

本目录负责在 **Linux 服务器** 上用 vLLM 部署本地模型，提供 OpenAI 兼容的 API 接口，供 MemMeta 流水线调用。

---

## 📁 文件说明

| 文件 | 作用 |
|------|------|
| `run.sh` | **启动脚本**（推荐）—— 调用 `vllm.entrypoints.openai.api_server` 官方服务 |
| `stop.sh` | **停止脚本** —— 杀死所有 vLLM 相关进程并释放端口 |
| `test.sh` | **测试脚本** —— curl 发送一条请求验证服务是否正常 |
| `serve_vllm.py` | 手写 FastAPI 服务（备用，功能不如官方入口全） |
| `requirements.txt` | 运行环境的完整依赖列表 |
| `vllm.log` / `vllm.out` | 运行日志 |

---

## 🚀 快速开始 (3 步)

### Step 1: 修改 `run.sh` 中的配置

打开 `run.sh`，修改顶部的变量：

```bash
# ===== 必须修改的 =====
MODEL_DIR='/home/models/Qwen2.5-14B-Instruct'  # ← 改为你的模型路径
CUDA_VISIBLE_DEVICES="0,1,2,3"                  # ← 4 张 A6000
PORT=8008                                        # ← 你想用的端口

# ===== 4x A6000 推荐参数 =====
TP_SIZE=4
GPU_MEMORY_UTILIZATION=0.92
MAX_NUM_SEQS=64

# ===== 可选：LoRA 适配器 =====
# 不加载 LoRA：注释掉或置空
ADAPTOR_PATH=''

# 加载 LoRA：格式为 "名称=路径"（名称在 API 调用时用作 model 参数）
ADAPTOR_PATH='dpo_level1=/home/LLaMA-Factory/saves/qwen_2_5/lora/dpo-level1'
```

### Step 2: 启动服务

```bash
cd api_call
chmod +x run.sh stop.sh test.sh
nohup bash run.sh > vllm.out 2>&1 &
```

等待模型加载完成（通常 1-3 分钟，可以用 `tail -f vllm.out` 查看进度）。  
当看到 `Uvicorn running on http://0.0.0.0:8008` 时，说明启动成功。

### Step 3: 验证服务

```bash
# 方法 1: 用自带的 test.sh
bash test.sh

# 方法 2: 手动 curl
curl http://localhost:8008/v1/models

# 方法 3: 发一条对话请求
curl http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/models/Qwen2.5-14B-Instruct",
    "messages": [{"role": "user", "content": "1+1=?"}],
    "temperature": 0.1
  }'
```

---

## ⚠️ 关键注意事项

### 1. model 参数必须与 MODEL_DIR 完全一致

vLLM 启动后会将 `--model` 参数的值注册为 model name。  
在 API 请求中，`"model"` 字段必须**完全一致**，否则返回 `Model not found` 错误。

```
run.sh 中:    MODEL_DIR='/home/models/Qwen2.5-14B-Instruct'
config.py 中: weak_model_name = "/home/models/Qwen2.5-14B-Instruct"  ← 必须一样!
test.sh 中:   "model": "/home/models/Qwen2.5-14B-Instruct"           ← 必须一样!
```

你可以随时用 `curl http://localhost:8008/v1/models` 查看已注册的 model name。

### 2. 加载 LoRA 时的 model 参数

如果加载了 LoRA 适配器 `dpo_level1=/path/to/lora`，API 中使用该适配器时：

```json
{"model": "dpo_level1", ...}   // ← 使用 LoRA 名称
{"model": "/home/models/Qwen2.5-14B-Instruct", ...}  // ← 使用基座模型
```

### 3. 端口一致性

`run.sh` 中的 `PORT` 必须与 `core/config.py` 中的 `weak_model_url` 端口一致：

```
run.sh:     PORT=8008
config.py:  weak_model_url = "http://localhost:8008/v1/chat/completions"
```

如果不一致，可在 `main.py` 中用 `--weak-url` 覆盖：
```bash
python main.py -s 3 --weak-url http://localhost:8008/v1/chat/completions
```

### 4. GPU 内存

| 模型 | GPU 需求 (bfloat16) | 建议配置 |
|------|---------------------|----------|
| Qwen2.5-7B-Instruct | ~15GB | 1× A100 40GB |
| Qwen2.5-14B-Instruct | ~30GB | 2× A100 40GB 或 1× A100 80GB |
| Qwen2.5-14B + LoRA | ~35GB | 2× A100 80GB (TP=2) |
| Qwen2.5-14B + LoRA 高并发 | ~60GB | 4× A100 80GB (TP=4) |
| Qwen2.5-14B + LoRA 高并发 | ~60GB | 4× A6000 48GB 或 4× A100 80GB (TP=4) |

---

## 🔗 与 MemMeta Pipeline 的对接

部署完成后，确保 `core/config.py` 中的配置正确：

```python
# core/config.py → HyperParamsConfig

# 弱模型 = 你部署的本地 vLLM 模型
weak_model_name = "/home/models/Qwen2.5-14B-Instruct"  # 与 run.sh MODEL_DIR 一致
weak_model_url  = "http://localhost:8008/v1/chat/completions"  # 与 run.sh PORT 一致
weak_model_key  = "EMPTY"  # vLLM 不需要 key

# 强模型 = 外部商用 API (Step 1 必须)
strong_model_name = "deepseek-reasoner"
strong_model_url  = "https://api.deepseek.com/v1/chat/completions"
strong_model_key  = "<your-api-key>"   # 填入你的 key
```

然后运行流水线：

```bash
# Step 1: 生成 DPO 数据 (需要强+弱模型)
python main.py -s 1 -i data/dpo_data/dpo_level1.json --strong-key sk-xxx

# Step 2: 从 DPO 数据更新记忆 (仅需弱模型)
python main.py -s 2

# Step 3: 基于记忆推理 (仅需弱模型)
python main.py -s 3 -i data/test/test_filter.json
```

---

## 🛑 停止服务

```bash
bash stop.sh
```

该脚本会：
1. 杀死所有 vLLM 相关进程
2. 杀死 Ray 子进程
3. 释放端口 8008
4. 清理 GPU 显存

---

## 🔧 常见问题

### Q: `Model not found` 错误
→ 检查 `config.py` 中的 `weak_model_name` 是否与 `run.sh` 中的 `MODEL_DIR` **完全一致**

### Q: `Connection refused` 错误
→ vLLM 还没启动完成，或端口不对。用 `curl http://localhost:8008/v1/models` 确认

### Q: `CUDA out of memory`
→ 减少 `MAX_NUM_SEQS`（如 64），或增大 `TP_SIZE` 使用更多 GPU

### Q: LoRA 加载失败
→ 确认 LoRA 路径存在，且与基座模型兼容（同系列 Qwen2.5）

### Q: `serve_vllm.py` 和 `run.sh` 用哪个？
→ **推荐用 `run.sh`**（官方入口，支持 chat_template、LoRA、流式输出等）。  
`serve_vllm.py` 是早期手写版本，不支持 chat_template，仅供参考。
