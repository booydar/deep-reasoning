#!/usr/bin/env bash
# set -e

SCRIPT_DIR=/workspace-SR006.nfs2/bulatov/rmt/reasoning/deep-reasoning
RUNS_DIR=/workspace-SR006.nfs2/bulatov/rmt/runs

cd $SCRIPT_DIR

NP=1
CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.experimental:RecurrentWrapperNoSegmentation
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_NAME=multiplication_4x4_reversed
ITERS=30000
TBS=256
INPUT_SIZE=1024

for N in 1; do
    MODEL_NAME=gpt2
    SEGMENT_ORDERING=regular
    MAX_N_SEGMENTS=1
    MEMORY_SIZE=16
    BS=256
    SCHEDULER=constant
    INPUT_SEQ_LEN=$((INPUT_SIZE))

    for LR in 5e-04; do
        GRADIENT_ACC_STEP=$((TBS/(BS*NP)))
        ACCEL_CONFIG="${SCRIPT_DIR}/accel_configs/accelerate_bf16.yaml"
        MAIN_SCRIPT="${SCRIPT_DIR}/run_finetuning_reasoning_rmt-v2.py"

        echo "RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N"
        echo "RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N $ITERS $D_MEM"

        accelerate launch --num_processes $NP --config_file $ACCEL_CONFIG $MAIN_SCRIPT \
        --task_name $TASK_NAME \
        --dataset_name "booydar/multiplication_4x4_reversed" \
        --output_dir ${RUNS_DIR}/${TASK_NAME}/${MODEL_NAME}/${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_${INPUT_SEQ_LEN}_LR${LR}-cot-v2 \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --sample_size $INPUT_SEQ_LEN \
        --segment_size $INPUT_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --use_cot \
        --per_device_train_batch_size $BS --gradient_accumulation_steps $GRADIENT_ACC_STEP \
        --max_steps $ITERS \
        --greater_is_better False \
        --save_total_limit 1 \
        --metric_for_best_model "eval_accuracy_ans" \
        --k1 -1 --k2 -1 \
        --optimizer AdamW --weight_decay 0.001 \
        --learning_rate ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps 5000 \
        --data_n_workers 2 \
        --logging_steps 100 --eval_steps 500 --save_steps 1000 \
        --show_valid_examples 0 \
        --early_stopping_patience 400 \
        --seed $((N+42)) \
        --max_grad_norm 1.0 \
        --mask_non_completion \
        --report_to tensorboard
    done
done

echo "done"
