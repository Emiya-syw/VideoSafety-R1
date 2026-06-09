#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"
require_file "${GRPO_DATA}"

GRPO_LOCAL_BATCH_SIZE="${GRPO_LOCAL_BATCH_SIZE:-1}"
GRPO_GRAD_ACCUM_STEPS="${GRPO_GRAD_ACCUM_STEPS:-1}"

# Stage 4: sample grouped completions and optimize the LLM with format, DRA,
# modality-classification, ROUGE, clipping, and KL signals.
"${torchrun_cmd[@]}" videollama3/train_grpo_vidlm3.py \
    --deepspeed local_scripts/zero3_offload.json \
    --model_type videollama3_qwen2 \
    --model_name_or_path "${STAGE3_DIR}" \
    --dataset_name "${GRPO_DATA}" \
    --output_dir "${STAGE4_DIR}" \
    --vision_encoder "${VISION_ENCODER}" \
    --mm_projector_type mlp2x_gelu \
    --image_merge_size 8 \
    --video_merge_size 8 \
    --fps 1 \
    --max_frames 90 \
    --model_max_length 10000 \
    --mm_max_length 5000 \
    --max_prompt_length 10000 \
    --max_completion_length 768 \
    --use_token_compression True \
    --learnable_tokens True \
    --llm_lr 1e-6 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size "${GRPO_LOCAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRPO_GRAD_ACCUM_STEPS}" \
    --num_generations "${NUM_GENERATIONS:-6}" \
    --num_train_epochs 1 \
    --epsilon "${GRPO_EPSILON:-0.2}" \
    --beta "${KL_BETA:-0.04}" \
    --alpha_min "${ALPHA_MIN:-0.1}" \
    --alpha_max "${ALPHA_MAX:-0.5}" \
    --gamma_visual "${GAMMA_VISUAL:-1.0}" \
    --gamma_textual "${GAMMA_TEXTUAL:-1.0}" \
    --temporal False \
    --len_control False \
    --torch_dtype bfloat16 \
    --bf16 True \
    --tf32 False \
    --gradient_checkpointing True \
    --attn_implementation flash_attention_2 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 5 \
    --logging_steps 1 \
    --save_steps 200 \
    --save_only_model False \
    --run_name stage4_grpo
