#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_mamba_gsm8k.sh
#
# One-liner launcher for train_mamba_gsm8k.py
#  * Works with single- or multi-GPU via `accelerate launch`
#  * Respects your global `accelerate config`
#  * Exposes the most common knobs as env vars / flags
# ---------------------------------------------------------------------------
set -euo pipefail

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

SCRIPT_DIR=/home/user33/kashurin/deep-reasoning
cd $SCRIPT_DIR
RUNS_DIR=/home/user33/kashurin/runs

NP=2
export CUDA_VISIBLE_DEVICES="0,1"

MODEL="state-spaces/mamba-130m-hf"
PROJECT="mamba-130m-hf"
USE_COT=true
EPOCHS=15
LR=3e-4
WARMUP=500
GRAD_ACC=8
TRAIN_BS=8
EVAL_BS=4
OUTDIR=${RUNS_DIR}/${PROJECT}/BS${TRAIN_BS}_GA${GRAD_ACC}_LR${LR}-cot/
ACCEL_CONFIG="${SCRIPT_DIR}/accel_configs/accelerate_fp32_stage2.yaml"

echo Launching training with accelerate

accelerate launch --num_processes $NP --config_file $ACCEL_CONFIG \
        run_finetuning_reasoning_mamba.py \
        --model_name $MODEL \
        --dataset_name "booydar/gsm8k" \
        --output_dir $OUTDIR \
        --max_cot_steps 8 \
        --use_cot \
        --per_device_train_batch_size $TRAIN_BS \
        --per_device_eval_batch_size $EVAL_BS \
        --gradient_accumulation_steps $GRAD_ACC \
        --num_train_epochs $EPOCHS \
        --metric_for_best_model "eval_loss" \
        --greater_is_better False \
        --save_total_limit 1 \
        --weight_decay 0.001 \
        --learning_rate ${LR} \
        --lr_scheduler_type constant \
        --warmup_steps $WARMUP \
        --data_n_workers 4 \
        --logging_steps 5 --eval_steps 50 --save_steps 50 \
        --eval_strategy steps \
        --eval_accumulation_steps 8 \
        --show_valid_examples 0 \
        --early_stopping_patience 15 \
        --seed 43 \
        --max_grad_norm 1.0 \
        --mask_non_completion \
        --report_to wandb \
        --log_level info


