# Single Test Set Runners

Run one test set at a time from the project root:

```bash
bash test_sets/run_deepscaler.sh
bash test_sets/run_math500.sh
bash test_sets/run_gsm8k.sh
bash test_sets/run_bbh_logic.sh
bash test_sets/run_bbh_math.sh
bash test_sets/run_svamp.sh
bash test_sets/run_general365.sh
```

Sandbox simulation:

```bash
DRY_RUN=1 bash test_sets/run_math500.sh
```

All scripts share `../config.env`.
