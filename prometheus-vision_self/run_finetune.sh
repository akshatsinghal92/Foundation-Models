#!/bin/bash

# Set the image folder to a path that makes "../mmmu" resolve correctly.
# Images are in /home/839temp/omkar/prometheus-vision/mmmu
# JSON paths are "../mmmu/..."
# So we need base path P such that P/../mmmu = .../mmmu
# P = .../prometheus-vision/llava works because llava/../mmmu = mmmu
IMAGE_FOLDER="/home/839temp/omkar/prometheus-vision/llava"
export PATH="/home/839temp/miniconda3/envs/prometheus-vision/bin:$PATH"

/home/839temp/miniconda3/envs/prometheus-vision/bin/deepspeed --include "localhost:0" llava/train/train_mem.py \
    --deepspeed ./zero2.json \
    --model_name_or_path prometheus-eval/prometheus-vision-7b-v1.0 \
    --version plain \
    --data_path ./new_train_llava_judge_data.json \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/new-ad-5e-finetuned-prometheus-vlm-7b \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
