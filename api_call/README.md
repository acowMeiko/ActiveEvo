# api_call/ — vLLM 模型部署模块

本目录是**固定、可复用**的模型部署模块。默认模型、GPU和运行参数统一写入 `deploy.conf`，命令行参数只用于临时覆盖。

端口规则：**port = DEPLOY_PORT_BASE + GPU_ID**。默认基数为 8100（GPU 0 → 8100，GPU 7 → 8107）。

---

## 文件说明

| 文件 | 作用 |
|------|------|
| `run.sh` | **启动脚本** — 指定模型和 GPU 列表，每卡启动一个 vLLM |
| `stop.sh` | **停止脚本** — 终止 vLLM 进程并释放 GPU |
| `test.sh` | **测试脚本** — 验证 API 端点 |
| `requirements.txt` | vllm / openai / fastapi / uvicorn |

---

## 快速开始

### 1. 编辑配置

```bash
vim deploy.conf   # 项目根目录
```

只需确认模型路径和 vLLM 参数：

```bash
LLAMA_PATH=/home/models/Llama-3_1-8b-Instruct
QWEN_PATH=/home/models/Qwen2.5-14B-Instruct
MAX_NUM_SEQS=128
GPU_MEMORY_UTILIZATION=0.92
DEPLOY_MODEL=llama
DEPLOY_GPU_IDS="0 1 2 3"
DEPLOY_PORT_BASE=8100
DEPLOY_SERVER_SEED=42
```

### 2. 启动

```bash
# 使用 deploy.conf 中的默认模型和GPU
bash api_call/run.sh deploy.conf --daemon

# llama 部署在 GPU 0-3 (端口 8100-8103)
bash api_call/run.sh deploy.conf llama 0 1 2 3 --daemon

# qwen 部署在 GPU 4-7 (端口 8104-8107)
bash api_call/run.sh deploy.conf qwen 4 5 6 7 --daemon

# llama 全 8 卡
bash api_call/run.sh deploy.conf llama 0 1 2 3 4 5 6 7 --daemon

# 后台运行
# 环境变量临时覆盖，实验脚本也会读取相同覆盖
MEMUA_GPU_IDS="2 5 7" bash api_call/run.sh deploy.conf --daemon
```

### 3. 测试

```bash
bash api_call/test.sh deploy.conf llama       # 使用本次实际部署的全部端口
bash api_call/test.sh deploy.conf llama 8103  # 指定端口
```

启动成功后会生成 `logs/active_deployment.env`。未显式给出端口时，测试脚本优先读取本次实际部署的端口。

### 4. 停止

```bash
bash api_call/stop.sh deploy.conf                 # 停止本次实际部署的实例
bash api_call/stop.sh deploy.conf llama 4 6       # 只停止指定GPU
bash api_call/stop.sh deploy.conf --clean         # 同时清理对应日志
bash api_call/stop.sh deploy.conf --all            # 停止所有受PID文件管理的实例
```

未显式给出 GPU 时，停止脚本同样优先读取 `logs/active_deployment.env`，因此命令行覆盖的卡号不会丢失。

---

## API 端点

启动后，每个 Backbone 实例提供：

```
http://{HOST}:{PORT}/v1/chat/completions   # 对话接口 (POST)
http://{HOST}:{PORT}/v1/models              # 模型列表 (GET)
```

**model 参数** 使用配置中的 `MODEL_*_SERVED_NAME`：

```bash
curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "What is 1+1?"}],
    "temperature": 0.1,
    "max_tokens": 512
  }'
```

---

## SUAgent 对接

SUAgent 在 `core/config.py` 中配置 API 地址，与部署配置中的 `MODEL_A_SERVED_NAME` 和 `MODEL_A_PORTS` 保持一致：

```python
weak_model_name = "qwen"
weak_model_url  = "http://localhost:8100/v1/chat/completions"
weak_model_key  = "EMPTY"
```

运行时通过 `--ports` 指定多卡：

```bash
python -m core.agent_pipeline --ports 8100 8101 8102 8103 --model qwen
```

---

## 多 Backbone 场景

| Backbone | Model Name | 端口 | GPU |
|----------|------------|------|-----|
| A (Qwen) | `qwen` | 8100-8103 | 0,1,2,3 |
| B (Llama)| `llama` | 8104-8105,8107 | 4,5,7 |

```bash
# Qwen
python run_stage1_dual.py --backbone qwen --ports 8100 8101 8102 8103

# Llama
python run_stage1_llama.py --backbone llama --ports 8104 8105 8107
```

---

## 常见问题

| 问题 | 解决 |
|------|------|
| `Connection refused` | 模型加载中（1-3 分钟），`tail -f api_call/logs/*.log` |
| `Model not found` | model 参数需与配置中 `MODEL_*_SERVED_NAME` 一致 |
| `CUDA out of memory` | 降低 `GPU_MEMORY_UTILIZATION` 或 `MAX_NUM_SEQS` |
| LoRA/Adapter | 在配置的 `VLLM_EXTRA_ARGS` 中添加 `--enable-lora --lora-modules name=/path` |
