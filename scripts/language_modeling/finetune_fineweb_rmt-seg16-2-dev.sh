#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetune_babilong_baseline.sh
set -e

# SCRIPT_DIR=/workspace-SR006.nfs2/bulatov/rmt/reasoning/deep-reasoning
# RUNS_DIR=/workspace-SR006.nfs2/bulatov/rmt/runs

SCRIPT_DIR=/home/jovyan/bulatov/rmt/reasoning/deep-reasoning
RUNS_DIR=/home/jovyan/bulatov/rmt/runs

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
DATASET_NAME=fineweb-edu
METRIC=exact_match

MODEL_NAME=SmolLM2-135M
FROM_PRETRAINED=HuggingFaceTB/SmolLM2-135M

TASK_DATASET=HuggingFaceFW/fineweb-edu

ITERS=15000
TBS=1024
BS=256

for LR in 3e-04
do

for SEGMENT_SIZE in 16 
do

for MAX_N_SEGMENTS in 4
do

for MEMORY_SIZE in 32
do

SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens
GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
SCHEDULER=constant

for N in 1
do

K2=-1   # BPTT unroll length

ACCEL_CONFIG="${SCRIPT_DIR}/accel_configs/accelerate_bf16.yaml"
MAIN_SCRIPT="${SCRIPT_DIR}/run_finetuning_lm_rmt.py"

echo RUNNING: DATASET_NAME $DATASET_NAME MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME  LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

# python run_finetuning_lm_rmt.py \
accelerate launch --num_processes $NP --config_file $ACCEL_CONFIG --main_process_port 29032 $MAIN_SCRIPT \
        --task_name $TASK_DATASET \
        --output_dir ${RUNS_DIR}/${DATASET_NAME}/$MODEL_NAME/LR${LR}_${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}-bs1024-nfs/run_$N \
        --model_cpt /home/jovyan/bulatov/rmt/runs/fineweb-edu/SmolLM2-135M/LR3e-04_constant_adamw_wd1e-03_2x16_mem32_bs1024_bptt--1-bs1024-nfs/run_1/checkpoint-15000/pytorch_model.bin \
        --from_pretrained $FROM_PRETRAINED \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --segment_size $SEGMENT_SIZE \
        --sample_size $SAMPLE_SIZE \
        --val_sample_size $SAMPLE_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --no_loss_from_first_segment \
        --per_device_train_batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --max_steps $ITERS \
        --metric_for_best_model "eval_loss" \
        --greater_is_better false \
        --save_total_limit 1 \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --learning_rate ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps 3000 \
        --data_n_workers 2 \
        --logging_steps 25 --eval_steps 100 \
        --show_valid_examples 5 \
        --seed $(($N+42)) \
        
done
done
done
done
done
done
done
done
echo "done"
