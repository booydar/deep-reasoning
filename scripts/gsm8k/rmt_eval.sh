#!/usr/bin/env bash
set -e

SCRIPT_DIR=/home/user33/kashurin/deep-reasoning
RUNS_DIR=/home/user33/kashurin/runs

cd $SCRIPT_DIR

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.experimental:RecurrentWrapperNoSegmentationGenerate
BACKBONE_CLS=transformers:AutoModelForCausalLM

export CUDA_VISIBLE_DEVICES="0"
NP=1
TASK_NAME=gsm8k
CHECKPOINT=/home/user33/kashurin/RMT_SmolLM2-135M/cot_fixed_pad/checkpoint-2700/pytorch_model.bin
# CHECKPOINT=/home/user33/kashurin/RMT_SmolLM2-135M/cot/checkpoint-3300/pytorch_model.bin
# CHECKPOINT=/home/user33/kashurin/RMT_SmolLM2-135M/cot_mode_single_reasoning-corrected/checkpoint-3350/pytorch_model.bin
# CHECKPOINT=/home/user33/kashurin/RMT_SmolLM2-135M/cot_mode_overlap_reasoning/checkpoint-2100/pytorch_model.bin
MODEL_ID=HuggingFaceTB/SmolLM2-135M
MODEL_NAME=SmolLM2-135M
INPUT_SEQ_LEN=64
MEMORY_SIZE=16
MAX_N_SEGMENTS=10
MAX_NEW_TOKENS=100

BATCH_SIZE=1
MAX_COT_STEPS=$((MAX_N_SEGMENTS-2))
SEED=42

EVAL_SCRIPT="${SCRIPT_DIR}/run_evaluate_reasoning_rmt-v2.py"

echo "Evaluating model $CHECKPOINT on $TASK_NAME with max_cot_steps $MAX_COT_STEPS"

python $EVAL_SCRIPT \
    --model_cpt $CHECKPOINT \
    --from_pretrained $MODEL_ID \
    --memory_cell_cls $MEMORY_CELL \
    --recurrent_wrapper_cls $RECURRENT_WRAPPER \
    --dataset_name "booydar/gsm8k" \
    --task_name $TASK_NAME \
    --num_mem_tokens $MEMORY_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --max_n_segments $MAX_N_SEGMENTS \
    --max_cot_steps $MAX_COT_STEPS \
    --batch_size $BATCH_SIZE \
    --device cuda \
    --seed $SEED

echo "Evaluation done" 