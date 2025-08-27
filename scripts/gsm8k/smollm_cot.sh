#!/usr/bin/env bash
SCRIPT_DIR=/home/jovyan/bulatov/rmt/reasoning/deep-reasoning
RUNS_DIR=/home/jovyan/bulatov/rmt/runs

cd $SCRIPT_DIR

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_amt.language_modeling:AssociativeMemoryCell
RECURRENT_WRAPPER=modeling_amt.language_modeling:AssociativeRecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_NAME=gsm8k
ITERS=50000
TBS=256
INPUT_SIZE=1024

for N in 1; do
    MODEL_NAME=SmolLM2-135M
    FROM_PRETRAINED=HuggingFaceTB/SmolLM2-135M
    SEGMENT_ORDERING=regular
    MAX_N_SEGMENTS=1
    BS=32
    SCHEDULER=constant
    INPUT_SEQ_LEN=$((INPUT_SIZE))

    for LR in 1e-05 5e-05 1e-04; do
        GRADIENT_ACC_STEP=$((TBS/(BS*NP)))
        ACCEL_CONFIG="${SCRIPT_DIR}/accel_configs/accelerate_bf16.yaml"
        MAIN_SCRIPT="${SCRIPT_DIR}/run_finetuning_reasoning-v2.py"

        echo "RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N"
        echo "RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N $ITERS $D_MEM"

        accelerate launch --num_processes $NP --main_process_port 29506 --config_file $ACCEL_CONFIG $MAIN_SCRIPT \
        --task_name $TASK_NAME \
        --dataset_name "booydar/gsm8k" \
        --output_dir ${RUNS_DIR}/${TASK_NAME}/${MODEL_NAME}/SEGM_${MAX_N_SEGMENTS}x${INPUT_SIZE}_${INPUT_SEQ_LEN}_LR${LR}-cot \
        --from_pretrained $FROM_PRETRAINED \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --sample_size $INPUT_SEQ_LEN \
        --segment_size $INPUT_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --use_cot \
        --per_device_train_batch_size $BS --gradient_accumulation_steps $GRADIENT_ACC_STEP \
        --max_steps $ITERS \
        --layers_attr base_model.base_model.layers \
        --metric_for_best_model "eval_loss" \
        --greater_is_better False \
        --save_total_limit 1 \
        --k1 -1 --k2 -1 \
        --optimizer AdamW --weight_decay 0.001 \
        --learning_rate ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps 3000 \
        --min_lr 5e-05 \
        --data_n_workers 2 \
        --logging_steps 50 --eval_steps 250 --save_steps 500 \
        --show_valid_examples 0 \
        --early_stopping_patience 25 \
        --seed $((N+42)) \
        --max_grad_norm 1.0 \
        --mask_non_completion \
        --report_to tensorboard
    done
done

echo "done"