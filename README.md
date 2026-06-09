# VideoSafety-R1 Training

Training-only implementation of **VideoSafety-R1** based on VideoLLaMA3.
This repository contains the model changes, data utilities, DeepSpeed
configurations, and scripts required to reproduce the four-stage safety
post-training recipe.

> This repository does not include evaluation servers, demos, pretrained
> checkpoints, or datasets.

## Method

The training pipeline contains four consecutive stages:

| Stage | Objective | Dataset | Trainable modules | Learning rate |
| --- | --- | --- | --- | --- |
| 1 | Alarm-token autoregressive initialization | VST-SFT-6k + VCG-plus-2k + LLaVA-SFT-2k | LLM, alarm tokens | `1e-6`, `1e-5` |
| 2 | AT-SFT with visual/textual ATC losses | Same 10k mixture | LLM, alarm tokens, classifiers | `1e-6`, `1e-5`, `1e-5` |
| 3 | Safety CoT cold start | VST-CoT-15k | LLM | `1e-6` |
| 4 | Safety-guided GRPO | VST-RL-25k | LLM | `1e-6` |

Two dedicated placeholders are registered dynamically:

```text
<|visual_alarm|>
<|textual_alarm|>
```

Their tokenizer IDs are never hard-coded. Before the LLM forward pass, their
ordinary token embeddings are replaced with modality-specific trainable alarm
embeddings.

All stages use the same binary label convention:

```text
0 = safe
1 = harmful
```

## Repository Layout

```text
videollama3/
  model/
    alarm_tokens.py              # alarm-token registration
    processor.py                 # token insertion and ATC label packing
    videollama3_qwen2.py         # alarm embeddings, classifiers, AT-SFT loss
  trainer/grpo_trainer.py        # multimodal GRPO implementation
  train.py                       # stages 1-3
  train_grpo_vidlm3.py           # stage 4 and reward functions
scripts/
  data/                          # dataset construction/migration tools
  train/videosafety_r1/          # four-stage launch scripts
  zero3.json                     # stages 1-3 DeepSpeed configuration
local_scripts/
  zero3_offload.json             # stage 4 DeepSpeed configuration
```

## Requirements

Recommended environment:

- Linux
- Python `3.10`
- CUDA `11.8` or a compatible PyTorch/CUDA combination
- NVIDIA GPUs with BF16 support
- FFmpeg available from the command line

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Stage 4 defaults to Flash Attention 2:

```bash
pip install flash-attn --no-build-isolation
```

If Flash Attention is unavailable, change
`--attn_implementation flash_attention_2` in
`scripts/train/videosafety_r1/stage4_grpo.sh` to an implementation supported by
your environment.

## Base Checkpoint

`BASE_MODEL` must point to a VideoLLaMA3 Qwen2 checkpoint containing the LLM,
vision encoder configuration, and multimodal projector metadata.

Example:

```bash
export BASE_MODEL=/path/to/VideoLLaMA3-2B
```

The scripts use `DAMO-NLP-SG/SigLIP-NaViT` as the default vision encoder. Set
`VISION_ENCODER` when using another local or Hugging Face checkpoint:

```bash
export VISION_ENCODER=/path/to/SigLIP-NaViT
```

## Data Preparation

### AT-SFT and CoT format

Stages 1-3 consume JSON or JSONL records in the VideoLLaMA3 conversation
format. Video paths are relative to `DATA_ROOT`.

```json
{
  "video": ["videos/example.mp4"],
  "conversations": [
    {
      "from": "human",
      "value": "<video>\nDescribe the safety risk in this video."
    },
    {
      "from": "gpt",
      "value": "The video contains dangerous behavior."
    }
  ],
  "safety_labels": {
    "visual": 1,
    "textual": 0
  }
}
```

For CoT cold-start data, the assistant target should contain the structured
reasoning and labels:

```text
<think>...</think>
<answer>...</answer>
<vidType>1</vidType>
<textType>0</textType>
```

Explicit `safety_labels` are strongly recommended for AT-SFT. Legacy `$v$t`
markers and regex inference remain only for backward compatibility.

### Build the 10k AT-SFT mixture

The helper samples 6k VST-SFT, 2k VCG-plus, and 2k LLaVA-SFT records with a
deterministic seed:

```bash
python3 scripts/data/build_at_sft_mix.py \
  --vst /path/to/vst_sft_6k.jsonl \
  --vcg /path/to/vcg_plus.jsonl \
  --llava /path/to/llava_sft_665k.jsonl \
  --output data_root/vst_sft_10k.jsonl
```

VCG-plus and LLaVA records without labels are assigned
`visual=0, textual=0`.

### Convert legacy labels

If an existing dataset uses `0=harmful, 1=safe`, convert it before training:

```bash
python3 scripts/data/normalize_safety_labels.py \
  /path/to/legacy.jsonl \
  data_root/normalized.jsonl
```

This converts explicit fields, `$v$t` markers, and
`<vidType>/<textType>` values.

### GRPO format

Stage 4 expects JSON or JSONL records with the following fields:

```json
{
  "problem_id": "sample-0001",
  "data_type": "video",
  "path": "/absolute/path/to/example.mp4",
  "problem": "Analyze the video and provide a safe, useful response.",
  "problem_type": "free-form-and-choice",
  "solution": "<think>...</think><answer>...</answer><vidType>1</vidType><textType>0</textType>"
}
```

Supported `problem_type` values:

- `free-form-and-choice` for the safety reward
- `free-form`
- `multiple choice` with an additional `options` list
- `numerical`
- `OCR`
- `regression`

`path` must be accessible from every training node. Both `video` and `image`
media types are supported.

## Expected Data Layout

The default script paths are:

```text
data_root/
  vst_sft_10k.jsonl
  vst_cot_15k.jsonl
  vst_rl_25k.jsonl
  videos/
```

You can use different locations through environment variables:

```bash
export DATA_ROOT=/datasets/videosafety_r1
export AT_SFT_DATA=/datasets/videosafety_r1/at_sft_10k.jsonl
export COT_DATA=/datasets/videosafety_r1/cot_15k.jsonl
export GRPO_DATA=/datasets/videosafety_r1/rl_25k.jsonl
```

## Run the Full Recipe

Single node with eight GPUs:

```bash
export BASE_MODEL=/checkpoints/VideoLLaMA3-2B
export DATA_ROOT=/datasets/videosafety_r1
export NPROC_PER_NODE=8

bash scripts/train/videosafety_r1/run_all.sh
```

Outputs are written by default to:

```text
work_dirs/videosafety_r1/
  stage1_alarm_ar/
  stage2_alarm_atc/
  stage3_cot/
  stage4_grpo/
```

The checkpoint chain is:

```text
BASE_MODEL
  -> stage1_alarm_ar
  -> stage2_alarm_atc
  -> stage3_cot
  -> stage4_grpo
```

## Run Individual Stages

```bash
bash scripts/train/videosafety_r1/stage1_alarm_ar.sh
bash scripts/train/videosafety_r1/stage2_alarm_atc.sh
bash scripts/train/videosafety_r1/stage3_cot.sh
bash scripts/train/videosafety_r1/stage4_grpo.sh
```

To start from an existing checkpoint, override the preceding output variable:

```bash
STAGE2_DIR=/checkpoints/stage2_alarm_atc \
bash scripts/train/videosafety_r1/stage3_cot.sh
```

```bash
STAGE3_DIR=/checkpoints/stage3_cot \
bash scripts/train/videosafety_r1/stage4_grpo.sh
```

Stages 1-3 automatically resume when their output directory already contains
`checkpoint-*`. To restart one of these stages from scratch, use a new output
directory or remove/move the existing checkpoints.

For Stage 4, append the following argument to the command in
`stage4_grpo.sh`:

```bash
--resume_from_checkpoint /path/to/checkpoint-N
```

## Multi-Node Training

Set the same paths and networking values on every node:

```bash
export WORLD_SIZE=2
export NPROC_PER_NODE=8
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=16667
export DATA_ROOT=/shared/datasets/videosafety_r1
export OUTPUT_ROOT=/shared/outputs/videosafety_r1
```

On node 0:

```bash
RANK=0 bash scripts/train/videosafety_r1/stage1_alarm_ar.sh
```

On node 1:

```bash
RANK=1 bash scripts/train/videosafety_r1/stage1_alarm_ar.sh
```

## Configuration

Common environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `BASE_MODEL` | `weights/VideoLLaMA3-2B` | Initial VideoLLaMA3 checkpoint |
| `VISION_ENCODER` | `DAMO-NLP-SG/SigLIP-NaViT` | Vision encoder |
| `DATA_ROOT` | `data_root` | Dataset/media root |
| `OUTPUT_ROOT` | `work_dirs/videosafety_r1` | Training output root |
| `WORLD_SIZE` | `1` | Number of nodes |
| `NPROC_PER_NODE` | `8` | GPUs per node |
| `GLOBAL_BATCH_SIZE` | `128` | Stages 1-3 global batch size |
| `LOCAL_BATCH_SIZE` | `1` | Stages 1-3 per-GPU batch size |
| `LAMBDA_VISUAL` | `0.1` | Visual ATC loss weight |
| `LAMBDA_TEXTUAL` | `0.1` | Textual ATC loss weight |
| `NUM_GENERATIONS` | `6` | GRPO completions per prompt |
| `GRPO_EPSILON` | `0.2` | Policy-ratio clipping range |
| `KL_BETA` | `0.04` | Reference-model KL coefficient |
| `ALPHA_MIN` | `0.1` | ROUGE weight when both labels are correct |
| `ALPHA_MAX` | `0.5` | ROUGE weight when either label is wrong |
| `GAMMA_VISUAL` | `1.0` | Visual classification reward weight |
| `GAMMA_TEXTUAL` | `1.0` | Text classification reward weight |

`GLOBAL_BATCH_SIZE` must be divisible by:

```text
WORLD_SIZE * NPROC_PER_NODE * LOCAL_BATCH_SIZE
```

## Training Notes

- Stages 1 and 2 train alarm embeddings at `1e-5`.
- Stage 2 enables both visual and textual ATC heads.
- Stage 3 keeps alarm tokens active but disables ATC and updates only the LLM.
- Stage 4 keeps alarm embeddings active and frozen while updating the LLM.
- GRPO policy and reference log-probabilities are computed with the complete
  multimodal inputs, including video features and alarm embeddings.
- The GRPO reward is the sum of format reward and the DRA safety/task reward.
- Length and temporal auxiliary rewards are disabled in the provided recipe.

## Troubleshooting

**Dataset not found**

The launch scripts validate the annotation files before starting. Check
`AT_SFT_DATA`, `COT_DATA`, and `GRPO_DATA`.

**Out of memory**

Reduce `LOCAL_BATCH_SIZE`, `NUM_GENERATIONS`, `max_frames`, or sequence lengths.
For Stage 4, CPU parameter/optimizer offload is already enabled.

**Flash Attention import error**

Install a Flash Attention build compatible with the installed PyTorch/CUDA
versions, or change the attention implementation in the Stage 4 script.

**Wrong safety labels**

Verify that every stage uses `0=safe, 1=harmful`. Run
`normalize_safety_labels.py` for legacy annotations.

**Multi-node media loading failure**

GRPO media paths must resolve identically on every node. Prefer absolute paths
on shared storage.

## License

See [LICENSE](LICENSE). This project is derived from VideoLLaMA3; retain and
follow the licenses and usage terms of the original model and all datasets.

## Citation

Add the VideoSafety-R1 BibTeX entry here when the paper metadata is public.
Please also cite VideoLLaMA3 and the datasets used by your training run.
