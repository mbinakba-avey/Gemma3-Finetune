#!/bin/bash

MODEL_NAME="checkpoint-555"

# Set CUDA_HOME if not already set
if [ -z "$CUDA_HOME" ]; then
    # Try common CUDA installation paths (check for versioned directories first)
    for cuda_path in /usr/local/cuda-12.4 /usr/local/cuda-12.2 /usr/local/cuda-12.1 /usr/local/cuda-12 /usr/local/cuda-11.8 /usr/local/cuda-11 /usr/local/cuda; do
        if [ -d "$cuda_path" ] && [ -f "$cuda_path/bin/nvcc" ]; then
            export CUDA_HOME="$cuda_path"
            break
        fi
    done
    
    # If still not set, try to find nvcc in PATH
    if [ -z "$CUDA_HOME" ]; then
        NVCC_PATH=$(which nvcc 2>/dev/null)
        if [ -n "$NVCC_PATH" ] && [ -f "$NVCC_PATH" ]; then
            export CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
        fi
    fi
    
    # Try conda environment CUDA if available
    if [ -z "$CUDA_HOME" ] && [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX" ]; then
        for cuda_path in "$CONDA_PREFIX/pkgs/cuda-toolkit" "$CONDA_PREFIX"; do
            if [ -d "$cuda_path" ] && [ -f "$cuda_path/bin/nvcc" ]; then
                export CUDA_HOME="$cuda_path"
                break
            fi
        done
    fi
    
    # Try to find CUDA through Python environment (pip-installed CUDA packages)
    if [ -z "$CUDA_HOME" ] && [ -n "$VIRTUAL_ENV" ]; then
        # Check if nvidia-cuda-runtime is installed and try to infer CUDA_HOME
        CUDA_RUNTIME_PATH=$(find "$VIRTUAL_ENV" -path "*/nvidia/cuda_runtime*" -type d 2>/dev/null | head -1)
        if [ -n "$CUDA_RUNTIME_PATH" ]; then
            # Navigate up to find potential CUDA installation
            POTENTIAL_CUDA=$(dirname $(dirname $(dirname "$CUDA_RUNTIME_PATH")))
            if [ -d "$POTENTIAL_CUDA" ] && [ -f "$POTENTIAL_CUDA/bin/nvcc" ]; then
                export CUDA_HOME="$POTENTIAL_CUDA"
            fi
        fi
    fi
fi

# Validate CUDA_HOME if set
if [ -n "$CUDA_HOME" ]; then
    if [ ! -d "$CUDA_HOME" ] || [ ! -f "$CUDA_HOME/bin/nvcc" ]; then
        echo "Warning: CUDA_HOME is set to '$CUDA_HOME' but nvcc not found. Unsetting CUDA_HOME."
        unset CUDA_HOME
    else
        echo "Using CUDA_HOME: $CUDA_HOME"
        export PATH="$CUDA_HOME/bin:$PATH"
    fi
fi

# If CUDA_HOME is still not set, provide helpful error message
if [ -z "$CUDA_HOME" ]; then
    echo "Error: CUDA_HOME is not set and CUDA toolkit not found."
    echo ""
    echo "DeepSpeed requires CUDA toolkit (including nvcc compiler) to compile CUDA operations."
    echo ""
    echo "To fix this, you can:"
    echo "1. Install CUDA toolkit (recommended for Ubuntu):"
    echo "   sudo apt-get update"
    echo "   sudo apt-get install -y cuda-toolkit-12-4"
    echo "   export CUDA_HOME=/usr/local/cuda-12.4"
    echo ""
    echo "2. Or set CUDA_HOME manually if CUDA toolkit is installed elsewhere:"
    echo "   export CUDA_HOME=/path/to/cuda"
    echo ""
    echo "3. Or use conda to install CUDA toolkit:"
    echo "   conda install -c nvidia cuda-toolkit=12.4"
    echo ""
    exit 1
fi

export PYTHONPATH=src:$PYTHONPATH
export WANDB_API_KEY="7eadd40652b0651b0f12dc86ea4d5fde56db2e2a"
export WANDB_PROJECT="gemma3-poc-12"


deepspeed src/train/train_grpo.py \
    --optim adamw_bnb_8bit \
    --max_completion_length 256 \
    --max_prompt_length 512 \
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
    --num_train_epochs 1 \
    --num_generations 8 \
    --per_device_train_batch_size 64 \
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
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 10 \
    --dataloader_num_workers 4