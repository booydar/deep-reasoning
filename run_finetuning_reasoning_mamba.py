#!/usr/bin/env python
"""
Fine‑tune **Mamba‑130M** (`state-spaces/mamba-130m-hf`) on the GSM8K reasoning dataset
(`booydar/gsm8k`) using Supervised Fine‑Tuning (TRL `SFTTrainer`).

🔑 **Accelerate‑ready & DeepSpeed‑friendly**
-------------------------------------------
* Launch with **`accelerate launch`** for seamless multi‑GPU / multi‑node jobs.
* Optional **`--deepspeed_config`** enables ZeRO‑1/2/3 + offloading.
* Console, TensorBoard *and* Weights & Biases logging built‑in.

Usage
-----
```bash
pip install "torch>=2" transformers datasets accelerate trl sentencepiece wandb

# 1️⃣ Configure accelerate (one‑time)
accelerate config  # or accelerate config default

# 2️⃣ Launch (single or multi‑GPU), full CoT, bf16, wandb enabled
accelerate launch train_mamba_gsm8k.py \
    --use_cot \
    --bf16 \
    --output_dir ./mamba_gsm8k

# With a DeepSpeed ZeRO‑3 json file
accelerate launch train_mamba_gsm8k.py \
    --deepspeed_config ds_zero3.json \
    --use_cot --bf16 --output_dir ./mamba_gsm8k_ds

# Disable wandb entirely
accelerate launch train_mamba_gsm8k.py --no_wandb
```

Environment variables recognised by 🤗/wandb still apply, e.g.:
* `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_WATCH`.
* `TOKENIZERS_PARALLELISM`, `CUDA_VISIBLE_DEVICES`.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback
)

from dataclasses import dataclass, field

from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator, DistributedDataParallelKwargs

# -----------------------------------------------------------------------------
# LOGGING SET‑UP
# -----------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_mamba_gsm8k")


# -----------------------------------------------------------------------------
# CLI ARGUMENTS
# -----------------------------------------------------------------------------

parser = HfArgumentParser(SFTConfig)

parser.add_argument('--dataset_dir', type=str, default=None, help="path to local dataset dir")
parser.add_argument('--dataset_name', type=str, default=None, help="name of HF dataset")
parser.add_argument('--append_concat_token', action='store_true', default=False,
                    help="Append concat token during packing")
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')
parser.add_argument('--mask_non_completion', action='store_true', default=False,
                    help='Mask everything except completion in dataset')

# reasoning args
parser.add_argument('--use_cot', action='store_true', help='use chain of thought examples')
parser.add_argument('--max_cot_steps', type=int, default=None, help='maximum number of cot steps')

# model args
parser.add_argument('--model_name', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Full experiment checkpoint, used to resume training in SFTTrainer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')
parser.add_argument('--early_stopping_patience', type=int, default=-1,
                    help='Early stopping tolerance')
parser.add_argument('--min_lr', type=float, default=0,
                    help='Minimum learning rate for the scheduler')




# -----------------------------------------------------------------------------
# COLLATE FUNCTION (unchanged core logic)
# -----------------------------------------------------------------------------

def make_collate_fn(tokenizer, args, pad, think, ans, bos, eos):
    def collate_fn(batch: List[Dict[str, str]]):
        input_ids, labels, attention_mask = [], [], []

        for sample in batch:
            task, lab, cot = sample["task"], sample["labels"], sample["cot"]
            task_tokens = tokenizer.encode(task, add_special_tokens=False)
            labels_tokens = tokenizer.encode(str(lab), add_special_tokens=False)
            cot_tokens = tokenizer.encode(cot, add_special_tokens=False) if cot else []

            if args.use_cot and cot_tokens:
                full_input = task_tokens + think + cot_tokens + ans + labels_tokens + eos
            else:
                full_input = task_tokens + ans + labels_tokens + eos

            inp_tensor = torch.tensor(full_input, dtype=torch.long)
            input_ids.append(inp_tensor)

            # Mask prompt (& optional CoT) in labels
            lab_tensor = torch.tensor(full_input, dtype=torch.long)
            lab_tensor[:len(task_tokens)] = -100
            # if args.use_cot and cot_tokens:
            #     cot_end = len(task_tokens) + len(think) + len(cot_tokens)
            #     lab_tensor[len(task_tokens):cot_end] = -100
            labels.append(lab_tensor)
            attention_mask.append(torch.ones_like(inp_tensor))

        input_ids = pad_sequence(input_ids, padding_value=pad, batch_first=True)
        attention_mask = pad_sequence(attention_mask, padding_value=0, batch_first=True)
        labels = pad_sequence(labels, padding_value=-100, batch_first=True)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    args = parser.parse_args()
    
    logger.info(f"Parsed script args: {args}")

    set_seed(args.seed)

    # -----------------------------------------
    # Accelerator initialise early (barrier‑sync logging)
    # -----------------------------------------
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )

    accelerator.print(
        f"Accelerator initialised — num_processes={accelerator.num_processes}, device={accelerator.device}"
    )

    # -----------------------------------------
    # wandb set‑up (only on main process)
    # -----------------------------------------
    # run_name = args.wandb_run_name or Path(args.output_dir).name
    # if args.no_wandb:
    #     os.environ["WANDB_MODE"] = "disabled"
    #     report_to = ["tensorboard"]
    #     accelerator.print("wandb disabled via --no_wandb")
    # else:
    #     import wandb  # local import avoids dependency if disabled

    #     if accelerator.is_main_process:
    #         wandb.init(name=run_name)
    #     report_to = ["tensorboard", "wandb"]
    

    # -----------------------------------------
    # Tokenizer & model (main process downloads; Accelerator handles broadcast)
    # -----------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    accelerator.print("Tokenizer ready.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    accelerator.print("Model loaded.")

    # -----------------------------------------
    # Data
    # -----------------------------------------
    pad = tokenizer.pad_token_id
    think = tokenizer.encode("<think>")
    ans = tokenizer.encode("<answer>")
    bos = [tokenizer.bos_token_id]
    eos = [tokenizer.eos_token_id]

    accelerator.print("Loading dataset …")
    train_ds = load_dataset(args.dataset_name, split="train")
    eval_ds = load_dataset(args.dataset_name, split="valid")
    accelerator.print(f"Dataset sizes — train: {len(train_ds)}, eval: {len(eval_ds)}")

    if args.max_cot_steps is not None:
        train_ds = train_ds.filter(lambda x: x['cot_len'] <= args.max_cot_steps)
        eval_ds = eval_ds.filter(lambda x: x['cot_len'] <= args.max_cot_steps)
        logger.info(f"Filtered ds sizes: {len(train_ds), len(eval_ds)}")

    collate_fn = make_collate_fn(tokenizer, args, pad, think, ans, bos, eos)

    # -----------------------------------------
    # TrainingArguments (Accelerate‑compatible)
    # -----------------------------------------

    def compute_accuracy(eval_pred):
        preds = eval_pred.predictions.argmax(axis=-1)[:, :-1]
        labels = eval_pred.label_ids[:, 1:]

        labels_masks = labels > 0
        preds_full = [p[m] for p, m in zip(preds, labels_masks)]
        labels_full = [lab[m] for lab, m in zip(labels, labels_masks)]

        all_preds_cot, all_labels_cot = [], []
        all_preds_ans, all_labels_ans = [], []
        for i, (lab_tokens, pred_tokens) in enumerate(zip(labels_full, preds_full)):
            lab_tokens = lab_tokens.tolist()
            pred_tokens = pred_tokens.tolist()
            
            ans_start_index_l = max(i for i, x in enumerate(lab_tokens) if x == ans[0])
            ans_end_index_l = min(i for i, x in enumerate(lab_tokens) if x == eos[0])

            if ans[0] in pred_tokens:
                ans_start_index_p = max(i for i, x in enumerate(pred_tokens) if x == ans[0])
            else:
                ans_start_index_p = ans_start_index_l

            if eos[0] in pred_tokens:
                ans_end_index_p = min(i for i, x in enumerate(pred_tokens) if x == eos[0])
            else:
                ans_end_index_p = ans_end_index_l

            pred_cot_tokens = pred_tokens[:ans_start_index_p]
            lab_cot_tokens = lab_tokens[:ans_start_index_l]

            all_preds_cot.append(pred_cot_tokens)
            all_labels_cot.append(lab_cot_tokens)

            pred_ans_tokens = pred_tokens[ans_start_index_p+1:ans_end_index_p]
            lab_ans_tokens = lab_tokens[ans_start_index_l+1:ans_end_index_l]

            all_preds_ans.append(pred_ans_tokens)
            all_labels_ans.append(lab_ans_tokens)
        
        cot_correct = [p == l for p, l in zip(all_preds_cot, all_labels_cot)]
        ans_correct = [p == l for p, l in zip(all_preds_ans, all_labels_ans)]

        return {'accuracy_cot': np.mean(cot_correct), 'accuracy_ans': np.mean(ans_correct)}


    # Training args
    training_args_dict = {key: value for key, value in vars(args).items() if hasattr(SFTConfig('.'), key)}

    training_args_dict['remove_unused_columns'] = False
    training_args_dict['save_safetensors'] = False
    training_args_dict['label_names'] = ['labels']
    
    training_args_dict['log_level'] = 'debug'
    training_args_dict['load_best_model_at_end'] = args.early_stopping_patience != -1

    training_args_dict['dataset_kwargs'] = {"skip_prepare_dataset": True}


    training_args = SFTConfig(**training_args_dict)


    def lr_lambda(current_step):
        if current_step < training_args.warmup_steps:
            return current_step / training_args.warmup_steps
        if training_args.lr_scheduler_type == "linear":
            decay_factor = (training_args.max_steps - current_step) / (training_args.max_steps - training_args.warmup_steps)
            return max(training_args.min_lr / training_args.learning_rate, decay_factor)
        elif training_args.lr_scheduler_type == "constant":
            return 1.0
        else:
            raise ValueError("Unsupported lr_scheduler_type")

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda)

    # -----------------------------------------
    # SFTTrainer
    # -----------------------------------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_accuracy,
    )
    accelerator.print("Trainer initialised. Starting training …")

    if args.early_stopping_patience != -1:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )
        trainer.add_callback(early_stopping)

    trainer.train()
    accelerator.print("Training complete.")

    

    # -----------------------------------------
    # Save artefacts (main process only)
    # -----------------------------------------
    if accelerator.is_main_process:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        accelerator.print(f"Model & tokenizer saved to {args.output_dir}")

    # -----------------------------------------
    # Close wandb
    # -----------------------------------------
    # if not args.no_wandb and accelerator.is_main_process:
    #     import wandb

    #     wandb.finish()
    #     accelerator.print("wandb run finished & closed.")


if __name__ == "__main__":
    main()
