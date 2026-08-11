# MemUA Parameter Reference

这份文档只记录当前代码中实际生效的参数来源、默认值和设置含义。

主要来源：

- [pipeline.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/pipeline.py:1)
- [core/config.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/core/config.py:1)
- [core/agent_pipeline.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/core/agent_pipeline.py:1)
- [sample_unstable_questions.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/sample_unstable_questions.py:1)
- [modules/memory.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/modules/memory.py:1)
- [retrieve_principles.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/retrieve_principles.py:1)

## 1. CLI 参数

来自 [pipeline.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/pipeline.py:67)。

| 参数 | 默认值 | 用途 | 为什么这样设 |
|---|---:|---|---|
| `--levels` | `1 2 3 4` | 按 level 跑完整 curriculum | 当前数据就是按 4 个 level 组织 |
| `--stage` | `None` | 只跑单个 stage | 方便单独调试 Stage 1/2/3 |
| `--dataset/-d` | `None` | 指定数据集，如 `dpo_level2` | 单 stage 时必须知道要跑哪一轮 |
| `--local` | `False` | 单卡本地模式 | 本地调试只需要 `8000` |
| `--ports` | `None` | 指定本地 vLLM 端口列表 | 支持多卡轮询 |
| `--model/-m` | `None` | 覆盖模型名 | 兼容不同 served model name |
| `--test-input` | `None` | 覆盖测试集路径 | 默认会回到 `data/test/deepscaler.json` |
| `--memory-dir` | `memory` | memory 根目录 | 让训练/测试能切到不同记忆目录 |
| `--concurrency` | `10` | Stage 1 并发 | 默认值偏稳，避免一次把服务压死 |
| `--diff-concurrency` | `10` | Stage 1 diff 并发 | 与 Stage 1 主并发分开控制 |
| `--batch-size` | `5` | Stage 1 每批生成条数 | 配合多次采样和 early stop |
| `--max-trials` | `15` | Stage 1 每题最大采样次数 | 足够判断题是否稳定，同时不至于过慢 |
| `--skip-sampling` | `False` | 复用已有 sampling | 方便从中间产物续跑 |
| `--skip-mining` | `False` | 复用已有 mining | 同上 |
| `--eval-concurrency` | `10` | Stage 3 并发 | 默认保守，避免检索和推理同时打满 |
| `--stage2-concurrency` | `8` | Stage 2 验证并发 | Stage 2 单项更重，所以默认比 Stage 3 更保守 |
| `--threshold` | `0.55` | Stage 3 三维门控阈值 | 这是当前经验检索的默认阈值 |
| `--top-k` | `5` | Stage 3 返回经验上限 | 避免 prompt 太长，也保留足够多候选 |
| `--timeout` | `600` | pipeline 级超时 | 覆盖 Stage 2/3 的慢请求场景 |
| `MEMUA_MAX_ENDPOINT_ATTEMPTS` | `2` | 单个逻辑请求最多尝试多少个不同 endpoint | 每个 endpoint 只调用一次；正常并发仍使用全部 GPU |

## 2. Config 默认参数

来自 [core/config.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/core/config.py:137)。

### 2.1 弱模型 / 本地服务

| 参数 | 默认值 | 为什么这样设 |
|---|---|---|
| `weak_model_name` | `base` | 和 vLLM 的 `--served-model-name` 对齐，方便本地切模型但不改调用端 |
| `weak_model_url` | `http://localhost:8000/v1/chat/completions` | 单端口时可直接运行，多端口时会被轮询替换 |
| `weak_model_key` | `EMPTY` | 本地 vLLM 不要求真实 key |
| `weak_timeout` | `180` | 比普通 API 请求长，覆盖较慢推理 |

### 2.2 强模型

| 参数 | 默认值 | 为什么这样设 |
|---|---|---|
| `strong_model_name` | `deepseek-reasoner` | 为外部更强规划模型预留 |
| `strong_model_url` | `https://api.deepseek.com/v1/chat/completions` | 当前默认外部接口 |
| `strong_model_key` | 配置文件中的字符串 | 这是代码里当前真实默认值，但不建议再把 key 写死进仓库 |
| `strong_max_tokens` | `6144` | 强模型调用更贵，默认比弱模型更保守 |
| `strong_timeout` | `240` | 给外部接口更长等待时间 |

### 2.3 通用生成参数

| 参数 | 默认值 | 为什么这样设 |
|---|---:|---|
| `temperature` | `0.7` | Stage 1 需要不稳定性，温度不能太低 |
| `max_tokens` | `7500` | 当前流程经常要求长推理与长 diff |
| `top_p` | `0.95` | 保持采样多样性 |
| `batch_size` | `32` | 主要是全局默认，不直接等于 Stage 1 批大小 |
| `max_workers` | `4` | 本地并发保守值 |
| `tensor_parallel_size` | `4` | 对齐常见 4 卡部署 |
| `cuda_visible_devices` | `0,1,2,3` | 当前本机主跑卡默认 |

### 2.4 检索相关

| 参数 | 默认值 | 为什么这样设 |
|---|---:|---|
| `facet_weights` | `(0.5, 0.3, 0.2)` | 配置里保留了分维权重位，但当前主检索排序并未实际使用 |
| `faiss_hard_threshold` | `0.8` | 配置里保留的旧硬阈值位 |
| `faiss_model_name` | `all-MiniLM-L6-v2` | 配置项还在，但当前主 memory / retrieval 默认嵌入模型已是 `BAAI/bge-m3` |

## 3. Stage 1 参数

### 3.1 代码参数

| 参数 | 默认值 | 来源 | 解释 |
|---|---:|---|---|
| `concurrency` | `10` | `pipeline.py` | 并发处理题目 |
| `diff_concurrency` | `10` | `pipeline.py` | 并发做 diff / semantics |
| `batch_size` | `5` | `pipeline.py` / `sample_unstable_questions.py` | 每轮生成数 |
| `max_trials` | `15` | `pipeline.py` / `sample_unstable_questions.py` | 判断不稳定性的最大尝试次数 |

### 3.2 环境变量

| 环境变量 | 默认值 | 解释 | 为什么这样设 |
|---|---:|---|---|
| `MEMUA_STAGE1_MAX_RETRIES` | `2` | 单请求失败/截断后重试次数 | 避免坏端口或截断直接污染样本 |
| `MEMUA_STAGE1_REQUEST_TIMEOUT` | `300` | Stage 1 请求超时 | Stage 1 有时会遇到很长输出 |
| `MEMUA_STAGE1_MAX_TOKENS` | `8192` | Stage 1 单次最大生成长度 | Stage 1 轨迹需要尽量完整，避免截断误判 |

## 4. Stage 2 参数

### 4.1 主参数

| 参数 | 默认值 | 来源 | 为什么这样设 |
|---|---:|---|---|
| `validation_concurrency` | `8` | `AgentPipeline.stage2_memorize` | Stage 2 每条记录都很重，默认保守 |
| `max_principle_rounds` | `3` | `core/agent_pipeline.py` | 给 candidate 重新生成留出空间，但防止无限循环 |
| `STRICT_MEMORY_RECONCILE_THRESHOLD` | `0.95` | `core/agent_pipeline.py` | 只有语义几乎一致才考虑替换或合并 |

### 4.2 Gate 参数

| 参数 | 默认值 | 来源 | 为什么这样设 |
|---|---:|---|---|
| `MEMUA_STAGE2_NEIGHBOR_PASS_RATIO_THRESHOLD` | `0.6` | 环境变量 | 至少通过 60% 语义邻居，才能说明 principle 不是只对源题有效 |
| source validation `max_attempts` | `3` | `core/agent_pipeline.py` | 兼顾 first-shot 和 retry 稳定性 |
| neighbor validation `max_attempts` | `3` | `core/agent_pipeline.py` | 与 source 口径一致 |

### 4.3 DSD / 嵌入参数

| 参数 | 默认值 | 来源 | 解释 |
|---|---:|---|---|
| `embedding_model_name` | `BAAI/bge-m3` | `retrieve_principles.py` / `embedding_utils.py` | 当前主语义空间 |
| DSD batch size | `256` | `DSDRegistry` | 批量编码 registry |
| neighbor cache `top_k` | `16` | `DSDRegistry` | 预计算局部语义邻居 |

## 5. Stage 3 参数

| 参数 | 默认值 | 来源 | 为什么这样设 |
|---|---:|---|---|
| `threshold` | `0.55` | `pipeline.py` / `memory.retrieve()` | 当前三维门控检索默认阈值 |
| `top_k` | `5` | `pipeline.py` / `memory.retrieve()` | 避免经验块过多拖慢回答 |
| `concurrency` | `10` | `pipeline.py` | 默认稳妥并发 |
| `retry_on_wrong` | `2` | `AgentPipeline.stage3_evaluate` | 总尝试数上限为 `3` |
| `max_attempts` | `3` | `_summarize()` / `stage3_evaluate()` | 把一次首答 + 两次纠错固定下来 |

## 6. Memory 与检索参数

来自 [modules/memory.py](/home/czhu/document/ActiveEvo_ICASSP_2027/incoming/code/ActiveEvo_code/modules/memory.py:1)。

| 参数 | 默认值 | 解释 | 为什么这样设 |
|---|---:|---|---|
| `DUPLICATE_THRESHOLD` | `0.9` | 相似度超过则视为近重复 | 避免 memory 被几乎相同的 principle 塞满 |
| `UPDATE_THRESHOLD` | `0.75` | 相似度超过则更新已有项 | 给相关 principle 留合并空间 |
| `retrieve threshold` | `0.55` | 三维都要过线 | 保持 hard gate 风格 |
| `strict facet threshold` | `0.95` | Stage 2 reconciliation 专用 | 只对几乎同题项做替换/合并 |

## 7. 嵌入加载相关环境变量

| 环境变量 | 默认值 | 作用 | 为什么这样设 |
|---|---:|---|---|
| `MEMUA_FORCE_EMBEDDING` | `1` | 强制必须加载真实嵌入模型 | 默认严模式，避免静默退化成 hash |
| `MEMUA_ALLOW_HASH_FALLBACK` | `0` | 是否允许 hash fallback | 默认关闭，防止实验结果失真 |
| `MEMUA_EMBEDDING_DEVICE` | `auto` | 指定嵌入跑在 `cpu/cuda` | 方便在主模型占满 GPU 时把 embedding 挪到 CPU |
| `MEMUA_EMBEDDING_MODEL_PATH` | 空 | 手动指定 embedding 模型路径 | 方便离线 cache 或服务器迁移 |

## 8. 当前几个容易误解的参数事实

1. `facet_weights` 目前没有真正进入主检索排序
2. 当前主嵌入模型是 `bge-m3`，不是 `all-MiniLM-L6-v2`
3. Stage 3 的最终 `accuracy` 是 progressive 口径，不等于每轮独立全量重测
4. `retry_on_wrong=2` 对应的总尝试次数是 `3`，不是 `2`
