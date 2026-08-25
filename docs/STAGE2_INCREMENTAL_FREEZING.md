# Stage 2 增量生成与逐轮冻结

本机制固定以下长期规则：

```text
40题校准并检查质量
→ 冻结40题的输入、Principle、审查结果和协议
→ 扩展到100题时复用这40题，只生成新增60题
→ 检查100题累计结果并冻结
→ 扩展到600题时复用前100题，只生成新增500题
→ 最后扩展到全部合格Learning Bundle
```

“冻结”不是把结果留在对话中，而是写入项目内可校验、不可静默覆盖的文件。成功生成、没有可靠Principle、审查拒绝和终止失败都会形成题目终态，因此恢复运行不会为了获得更好结果而重新抽样。

## 1. 固定样本集合

目标集合使用同一个稳定排序，保证集合严格嵌套：

| 目标 | Level 1 | Level 2 | Level 3 | Level 4 | 总数 |
|---|---:|---:|---:|---:|---:|
| 40 | 10 | 10 | 10 | 10 | 40 |
| 100 | 25 | 25 | 25 | 25 | 100 |
| 600 | 100 | 200 | 200 | 100 | 600 |
| all | 全部 | 全部 | 全部 | 全部 | 全部 |

稳定键为 `sha256(question_id + "stage2_principle_test_42")`。历史8道开发题若仍在合格池中，会固定在对应Level的开头。

## 2. 三类持久化文件

```text
output/stage2/candidate_registry/
├── questions/<question_id>/<generation_protocol_hash>/generation.json
├── questions/<question_id>/<candidate_protocol_hash>/audits/...
└── principles/<candidate_id>/versions.json

output/stage2/incremental/
├── target_0040/
├── target_0100/
├── target_0600/
└── target_all_<N>/

state/stage2/
├── current_protocol.lock.json
├── protocols/<protocol_hash>.json
├── releases/round_0040/
├── releases/round_0100/
├── releases/round_0600/
└── STAGE2_EXPERIMENT_STATE.md
```

- Candidate Registry按“题目 + 协议哈希”保存可复用模型结果。
- Target Workspace保存某个累计目标的清单、候选库快照和质量检查模板。
- Release保存用户已经确认的不可变轮次。

只有文件存在不等于可复用。代码会同时校验Learning Bundle、Prompt、模型、Schema、审查设置和输入内容哈希。任一项变化都会判为缺失，不能静默混用。

## 3. 40题首次冻结

40+40校准实验完成并且报告满足验收线后：

```bash
python pipeline.py stage2 \
  --mode incremental \
  --incremental-phase freeze \
  --target-size 40 \
  --incremental-target-root output/stage2/principle_test_80_v1
```

该命令把胜出的生成、审查和介入协议写入 `current_protocol.lock.json`，并导入40道来源题的已有结果。报告未批准扩大实验时，命令会拒绝冻结。

## 4. 扩展到100题

先只查看会复用多少题、需要新增多少题：

```bash
python pipeline.py stage2 --mode incremental --incremental-phase prepare --target-size 100
```

生成新增60题，并对新增候选运行已经冻结的审查协议：

```bash
python pipeline.py stage2 --mode incremental --incremental-phase run \
  --target-size 100 --model-concurrency 2
```

运行结束会产生：

```text
output/stage2/incremental/target_0100/quality_decision.template.json
```

检查这一轮的Principle和汇总结果后，复制该模板，填写：

```json
{
  "approved": true,
  "target_count": 100,
  "manifest_hash": "保持模板原值",
  "candidate_bank_hash": "保持模板原值",
  "protocol_hash": "保持模板原值",
  "reviewer": "检查人",
  "reviewed_at": "时间",
  "rationale": "为什么允许进入下一规模",
  "notes": []
}
```

再冻结累计100题：

```bash
python pipeline.py stage2 --mode incremental --incremental-phase freeze \
  --target-size 100 \
  --quality-decision output/stage2/incremental/target_0100/quality_decision.json
```

质量文件必须与本轮题数、Manifest、Candidate Bank和协议哈希完全匹配。仅写一个通用的 `approved=true` 不会通过。

## 5. 继续扩展

600题与全量使用同一组命令，只替换目标：

```bash
python pipeline.py stage2 --mode incremental --incremental-phase prepare --target-size 600
python pipeline.py stage2 --mode incremental --incremental-phase run --target-size 600 --model-concurrency 2
# 检查并填写 target_0600/quality_decision.json 后：
python pipeline.py stage2 --mode incremental --incremental-phase freeze \
  --target-size 600 \
  --quality-decision output/stage2/incremental/target_0600/quality_decision.json

python pipeline.py stage2 --mode incremental --incremental-phase prepare --target-size all
python pipeline.py stage2 --mode incremental --incremental-phase run --target-size all --model-concurrency 2
```

查看长期状态：

```bash
python pipeline.py stage2 --mode incremental --incremental-phase status
```

## 6. Principle修改规则

- 已冻结内容不得原地修改。
- 对同一候选的修复会生成新的 `version_id`，并用 `parent_version_id` 指向旧版本。
- 旧版本保留，后续Release选择新版本；拒绝结果也保留为审计证据。
- 协议或Prompt改变后，必须显式建立新协议迁移，不能把新旧候选放入同一冻结轮次。
- `release.json`最后写入，是一轮成功提交的标志；中途退出可安全恢复。

