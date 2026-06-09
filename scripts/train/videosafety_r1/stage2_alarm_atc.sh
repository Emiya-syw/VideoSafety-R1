#!/usr/bin/env bash

source "$(dirname "$0")/common.sh"
require_file "${AT_SFT_DATA}"

# Stage 2: continue AR training and add visual/textual binary ATC objectives.
"${torchrun_cmd[@]}" videollama3/train.py \
    --deepspeed scripts/zero3.json \
    --model_type videollama3_qwen2 \
    --model_path "${STAGE1_DIR}" \
    --vision_encoder "${VISION_ENCODER}" \
    --mm_projector_type mlp2x_gelu \
    --data_path "${AT_SFT_DATA}" \
    --data_folder "${DATA_ROOT}" \
    --image_merge_size 2 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames 90 \
    --model_max_length 8192 \
    --mm_max_length 5120 \
    --use_token_compression True \
    --learnable_tokens True \
    --multi_task True \
    --token_lr 1e-5 \
    --cls_lr 1e-5 \
    --llm_lr 1e-6 \
    --lambda_1 "${LAMBDA_VISUAL:-0.1}" \
    --lambda_2 "${LAMBDA_TEXTUAL:-0.1}" \
    --bf16 True \
    --tf32 True \
    --output_dir "${STAGE2_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${LOCAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --run_name stage2_alarm_atc
