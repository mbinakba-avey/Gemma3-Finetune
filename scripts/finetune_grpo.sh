#!/bin/bash

# MODEL_NAME="google/gemma-3-4b-it"
# MODEL_NAME="google/gemma-3-4b-pt"
MODEL_NAME="./gemma-3-base"

export PYTHONPATH=src:$PYTHONPATH
export WANDB_API_KEY="7eadd40652b0651b0f12dc86ea4d5fde56db2e2a"
export WANDB_PROJECT="gemma3-grpo-experiment-1-final"
export REWARD_JSON_OUTPUT_PATH="reward_data/reward_data.json"

export CUDA_VISIBLE_DEVICES=3
deepspeed src/train/train_grpo.py \
    --loss_type "grpo" \
    --use_vllm True \
    --optim adamw_bnb_8bit \
    --max_completion_length 128 \
    --max_prompt_length 256 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path data/train_set_specific_grpo.json \
    --image_folder /path/to/your/image/folder \
    --disable_flash_attn2 True \
    --lora_enable False \
    --freeze_projector False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --bf16 True \
    --output_dir output/test \
    --num_train_epochs 2 \
    --num_generations 16 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 50 \
    --sync_ref_model True \
    --ref_model_sync_steps 400 \
    --temperature 1.0 \
    --top_p 1.0 \
    --save_total_limit 5 \
    --dataloader_num_workers 16

# If the save steps is 50, real save steps is 500