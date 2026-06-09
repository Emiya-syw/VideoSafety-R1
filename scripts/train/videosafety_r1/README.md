# Four-Stage Training Scripts

These scripts implement the exact checkpoint sequence used by the repository.
All paths and distributed settings are configured in `common.sh` through
environment variables.

## Stages

| Script | Input checkpoint | Output checkpoint | Updated parameters |
| --- | --- | --- | --- |
| `stage1_alarm_ar.sh` | `BASE_MODEL` | `STAGE1_DIR` | LLM + alarm tokens |
| `stage2_alarm_atc.sh` | `STAGE1_DIR` | `STAGE2_DIR` | LLM + alarm tokens + ATC heads |
| `stage3_cot.sh` | `STAGE2_DIR` | `STAGE3_DIR` | LLM |
| `stage4_grpo.sh` | `STAGE3_DIR` | `STAGE4_DIR` | LLM |

Alarm embeddings remain active in Stages 3 and 4. They are not optimized in
those stages.

## Minimal Launch

```bash
BASE_MODEL=/path/to/VideoLLaMA3-2B \
DATA_ROOT=/path/to/data_root \
NPROC_PER_NODE=8 \
bash scripts/train/videosafety_r1/run_all.sh
```

## Common Overrides

```bash
export OUTPUT_ROOT=/path/to/output
export AT_SFT_DATA=/path/to/at_sft_10k.jsonl
export COT_DATA=/path/to/cot_15k.jsonl
export GRPO_DATA=/path/to/rl_25k.jsonl
export GLOBAL_BATCH_SIZE=128
export LOCAL_BATCH_SIZE=1
```

Stage 2:

```bash
export LAMBDA_VISUAL=0.1
export LAMBDA_TEXTUAL=0.1
bash scripts/train/videosafety_r1/stage2_alarm_atc.sh
```

Stage 4:

```bash
export NUM_GENERATIONS=6
export GRPO_EPSILON=0.2
export KL_BETA=0.04
export ALPHA_MIN=0.1
export ALPHA_MAX=0.5
bash scripts/train/videosafety_r1/stage4_grpo.sh
```

## Continue From a Completed Stage

Do not rerun earlier stages. Point the next script to the completed checkpoint:

```bash
STAGE2_DIR=/path/to/stage2_alarm_atc \
bash scripts/train/videosafety_r1/stage3_cot.sh
```

```bash
STAGE3_DIR=/path/to/stage3_cot \
bash scripts/train/videosafety_r1/stage4_grpo.sh
```

Stages 1-3 automatically resume from `checkpoint-*` under their output
directory. Stage 4 accepts
`--resume_from_checkpoint /path/to/checkpoint-N` in its launch command.

## Output Naming

The default output tree is:

```text
work_dirs/videosafety_r1/
  stage1_alarm_ar/
  stage2_alarm_atc/
  stage3_cot/
  stage4_grpo/
```

Set `PROJECT_NAME` or `OUTPUT_ROOT` to isolate different experiments.
