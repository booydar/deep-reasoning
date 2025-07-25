#!/usr/bin/env bash
# set -e

SCRIPT_DIR=/home/jovyan/bulatov/rmt/reasoning/deep-reasoning
RUNS_DIR=/home/jovyan/bulatov/rmt/runs

cd $SCRIPT_DIR

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_amt.language_modeling:AssociativeMemoryCell
RECURRENT_WRAPPER=modeling_amt.experimental:AssociativeRecurrentWrapperNoSegmentation
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_NAME=gsm8k
ITERS=50000
TBS=256
INPUT_SIZE=1024

for N in 1; do
    MODEL_NAME=gpt2
    SEGMENT_ORDERING=regular
    MAX_N_SEGMENTS=1
    MEMORY_SIZE=32
    D_MEM=32
    BS=8
    SCHEDULER=constant
    INPUT_SEQ_LEN=$((INPUT_SIZE))

    for LR in 5e-04; do
        GRADIENT_ACC_STEP=$((TBS/(BS*NP)))
        ACCEL_CONFIG="${SCRIPT_DIR}/accel_configs/accelerate_bf16.yaml"
        MAIN_SCRIPT="${SCRIPT_DIR}/run_finetuning_reasoning_rmt-v2.py"

        echo "RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N D_MEM"
        echo "RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N $ITERS $D_MEM"

        accelerate launch --num_processes $NP --main_process_port 29502 --config_file $ACCEL_CONFIG $MAIN_SCRIPT \
        --task_name $TASK_NAME \
        --dataset_name "booydar/gsm8k" \
        --output_dir ${RUNS_DIR}/${TASK_NAME}/${MODEL_NAME}-armt/${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_${INPUT_SEQ_LEN}_LR${LR}-cot_from_gsm \
        --backbone_cpt /home/jovyan/bulatov/rmt/runs/gsm8k/gpt2/basex1024-cot-v2/run1/checkpoint-24000/pytorch_model.bin \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --sample_size $INPUT_SEQ_LEN \
        --segment_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --d_mem $D_MEM \
        --max_n_segments $MAX_N_SEGMENTS \
        --per_device_train_batch_size $BS --gradient_accumulation_steps $GRADIENT_ACC_STEP \
        --max_steps $ITERS \
        --use_cot \
        --greater_is_better False \
        --save_total_limit 1 \
        --metric_for_best_model "eval_loss" \
        --k1 -1 --k2 -1 \
        --optimizer AdamW --weight_decay 0.001 \
        --learning_rate ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps $((ITERS/10)) \
        --min_lr 5e-05 \
        --data_n_workers 2 \
        --logging_steps 100 --eval_steps 500 --save_steps 1000 \
        --show_valid_examples 0 \
        --early_stopping_patience 75 \
        --seed $((N+42)) \
        --max_grad_norm 1.0 \
        --mask_non_completion \
        --report_to tensorboard
    done
done

echo "done"
