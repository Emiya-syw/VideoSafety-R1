#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

WORLD_SIZE="${WORLD_SIZE:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-16667}"
RANK="${RANK:-0}"

PROJECT_NAME="${PROJECT_NAME:-videosafety_r1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/work_dirs/${PROJECT_NAME}}"
BASE_MODEL="${BASE_MODEL:-${ROOT_DIR}/weights/VideoLLaMA3-2B}"
VISION_ENCODER="${VISION_ENCODER:-DAMO-NLP-SG/SigLIP-NaViT}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data_root}"

AT_SFT_DATA="${AT_SFT_DATA:-${DATA_ROOT}/vst_sft_10k.jsonl}"
COT_DATA="${COT_DATA:-${DATA_ROOT}/vst_cot_15k.jsonl}"
GRPO_DATA="${GRPO_DATA:-${DATA_ROOT}/vst_rl_25k.jsonl}"

STAGE1_DIR="${STAGE1_DIR:-${OUTPUT_ROOT}/stage1_alarm_ar}"
STAGE2_DIR="${STAGE2_DIR:-${OUTPUT_ROOT}/stage2_alarm_atc}"
STAGE3_DIR="${STAGE3_DIR:-${OUTPUT_ROOT}/stage3_cot}"
STAGE4_DIR="${STAGE4_DIR:-${OUTPUT_ROOT}/stage4_grpo}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
DENOMINATOR=$((WORLD_SIZE * NPROC_PER_NODE * LOCAL_BATCH_SIZE))
if (( GLOBAL_BATCH_SIZE % DENOMINATOR != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by WORLD_SIZE*NPROC_PER_NODE*LOCAL_BATCH_SIZE" >&2
    exit 1
fi
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / DENOMINATOR))

# A single array keeps torchrun arguments identical across all four stages and
# supports both single-node and multi-node launches through environment values.
torchrun_cmd=(
    torchrun
    --nnodes "${WORLD_SIZE}"
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    --node_rank "${RANK}"
)

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required dataset not found: $1" >&2
        exit 1
    fi
}
