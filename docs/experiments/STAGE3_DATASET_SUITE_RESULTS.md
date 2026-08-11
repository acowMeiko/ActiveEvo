# Stage 3 Dataset Suite Results

This page summarizes `MemUA` Stage 3 guided evaluation with the level-2 memory on the shared test suite.

## Core Results

| Dataset | Split | Test File | Total | Correct | Accuracy | Accuracy@3 | First Attempt Acc. |
|---|---|---|---:|---:|---:|---:|---:|
| `math500` | `full` | `data/test/benchmarks/math500_stage3.json` | 500 | 363 | 0.726 | 0.726 | 0.676 |
| `deepscaler` | `full` | `data/test/deepscaler.json` | 400 | 181 | 0.453 | 0.453 | 0.365 |
| `bbh` | `logic` | `data/test/benchmarks_split/bbh_logic_stage3.json` | 2437 | 2098 | 0.861 | 0.861 | 0.768 |
| `bbh` | `math` | `data/test/benchmarks_split/bbh_math_stage3.json` | 750 | 653 | 0.871 | 0.871 | 0.816 |
| `svamp` | `official_test` | `data/test/official/svamp/processed/test.json` | 1000 | 949 | 0.949 | 0.949 | 0.941 |
| `general365` | `official_test` | `data/test/official/general365/processed/test.json` | 720 | 111 | 0.154 | 0.154 | 0.108 |

## Retrieval And Retry Statistics

| Dataset | Split | Retrieval Used | Retrieval Ratio | Retry Used | Retry Ratio | Rescued by Retry |
|---|---|---:|---:|---:|---:|---:|
| `math500` | `full` | 375 | 0.750 | 162 | 0.324 | 25 |
| `deepscaler` | `full` | 257 | 0.642 | 254 | 0.635 | 35 |
| `bbh` | `logic` | 230 | 0.094 | 566 | 0.232 | 227 |
| `bbh` | `math` | 529 | 0.705 | 138 | 0.184 | 41 |
| `svamp` | `official_test` | 154 | 0.154 | 59 | 0.059 | 8 |
| `general365` | `official_test` | 232 | 0.322 | 642 | 0.892 | 33 |

## Shared Setup

| Field | Value |
|---|---|
| Run Date | `2026-04-21` |
| Memory Source | `memory/principles/principles.json` |
| Ports | `8000,8001,8002,8003` |
| Answer Max Tokens | `1024` |
| Status | `done` for all rows above |
