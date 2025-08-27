#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetune_babilong_baseline.sh
set -e

SCRIPT_DIR=/home/jovyan/bulatov/rmt/reasoning/deep-reasoning
RUNS_DIR=/home/jovyan/bulatov/rmt/runs

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
DATASET_NAME=wikitext
METRIC=exact_match

MODEL_NAME=gpt2  # backbone model
MODEL_PATH=gpt2

TASK_DATASET=Salesforce/wikitext

ITERS=50000
TBS=256
BS=16

for LR in 1e-04
do

for SEGMENT_SIZE in 512 # size of one segment in tokens
do

for MAX_N_SEGMENTS in 2
do

for MEMORY_SIZE in 16
do

SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens
GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
SCHEDULER=linear

for N in 1
do

K2=-1   # BPTT unroll length

ACCEL_CONFIG="${SCRIPT_DIR}/accel_configs/accelerate_bf16.yaml"
MAIN_SCRIPT="${SCRIPT_DIR}/run_finetuning_lm_rmt.py"

echo RUNNING: DATASET_NAME $DATASET_NAME MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME  LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

# python run_finetuning_lm_rmt.py \
accelerate launch --num_processes $NP --config_file $ACCEL_CONFIG --main_process_port 29002 $MAIN_SCRIPT \
        --task_name $TASK_DATASET \
        --output_dir ${RUNS_DIR}/${DATASET_NAME}/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}_nfs/run_$N \
        --from_pretrained $MODEL_PATH \
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
        --min_sample_len 16000 \
        --per_device_train_batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --max_steps $ITERS \
        --metric_for_best_model "eval_loss" \
        --greater_is_better false \
        --save_total_limit 1 \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --learning_rate ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps $(($ITERS/10)) \
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
