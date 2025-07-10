#!/usr/bin/env bash
set -e

SCRIPT_DIR=/home/user30/kashurin/deep-reasoning
RUNS_DIR=/home/user30/kashurin/runs

cd $SCRIPT_DIR

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_amt.language_modeling:AssociativeMemoryCell
RECURRENT_WRAPPER=modeling_amt.language_modeling:AssociativeRecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM

export CUDA_VISIBLE_DEVICES="2,3"
NP=2
TASK_NAME=gsm8k
N_EPOCHS=20
GRADIENT_ACC_STEP=8   # should be 8
BS=32
INPUT_SEQ_LEN=512
MAX_COT_STEPS=8
MODEL_ID=gpt2
MODEL_NAME=GPT2
SCHEDULER=constant

ACCEL_CONFIG="${SCRIPT_DIR}/accel_configs/accelerate_fp32_stage2.yaml"
MAIN_SCRIPT="${SCRIPT_DIR}/run_finetuning_reasoning-v2.py"

for N in 1; do
    for LR in 3e-04; do
        echo "RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N"
        echo "RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N $ITERS $D_MEM"

        accelerate launch --num_processes $NP --config_file $ACCEL_CONFIG $MAIN_SCRIPT \
        --task_name $TASK_NAME \
        --dataset_name "booydar/gsm8k" \
        --output_dir ${RUNS_DIR}/${TASK_NAME}/TR_${MODEL_NAME}/L${INPUT_SEQ_LEN}_BS${BS}_LR${LR}-cot \
        --from_pretrained $MODEL_ID \
        --model_type $MODEL_TYPE \
        --model_cls $BACKBONE_CLS \
        --sample_size $INPUT_SEQ_LEN \
        --max_cot_steps $MAX_COT_STEPS \
        --use_cot \
        --per_device_train_batch_size $BS \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps $GRADIENT_ACC_STEP \
        --num_train_epochs $N_EPOCHS \
        --layers_attr base_model.base_model.layers \
        --metric_for_best_model "eval_loss" \
        --greater_is_better False \
        --save_total_limit 1 \
        --optimizer AdamW --weight_decay 0.001 \
        --learning_rate ${LR} \
        --lr_scheduler_type $SCHEDULER \
        --warmup_steps 500 \
        --data_n_workers 4 \
        --logging_steps 5 --eval_steps 50 --save_steps 50 \
        --eval_strategy steps \
        --eval_accumulation_steps 8 \
        --show_valid_examples 0 \
        --early_stopping_patience 15 \
        --seed $((N+42)) \
        --max_grad_norm 1.0 \
        --mask_non_completion \
        --report_to wandb \
        --log_level info
    done
done

echo "done"