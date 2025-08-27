import logging
import os
from pathlib import Path
from datetime import timedelta
import torch
import datasets
from transformers import EarlyStoppingCallback, set_seed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch.nn.utils.rnn import pad_sequence

import accelerate
from peft import get_peft_model, LoraConfig, TaskType

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402

from lm_experiments_tools.utils import get_cls_by_name

from utils.reasoning import make_segment, split_cot

from trl import GRPOTrainer, GRPOConfig, DPOTrainer

from modeling_rmt.huggingface import RMTConfig, RMTForReasoning

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')

if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

parser = HfArgumentParser(GRPOConfig)
parser.add_argument('--task_name', type=str, help="Task name, wikitext:arxiv ...")
parser.add_argument('--dataset_dir', type=str, default=None, help="path to local dataset dir")
parser.add_argument('--dataset_name', type=str, default=None, help="name of HF dataset")
parser.add_argument('--task_ratios', type=str, help="Task rations, separated by : , 0.1:0.8: ..., sup up to 1")
parser.add_argument('--append_concat_token', action='store_true', default=False,
                    help="Append concat token during packing")
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
parser.add_argument('--sample_size', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')
parser.add_argument('--sliding_window', action='store_true', help='use slinding window attentinon mask, '
                    'eval on last segment only', default=False)
parser.add_argument('--attend_to_previous_input', action='store_true', help='attend to the previous segment',
                    default=False)
parser.add_argument('--use_length_filtering', action='store_true', help='filter samples longer than train len',
                    default=False)
parser.add_argument('--reduce_eval', type=float, default=None, help='part of eval to use')
parser.add_argument('--no_packing', action='store_true', help='disable packing, add padding', default=False)
parser.add_argument('--padding_side', type=str, help='set padding side', default=False)
parser.add_argument('--truncate_before', type=int, default=0, help='truncate input before defined ids, for debug only')
parser.add_argument('--truncate_only_train', action='store_true',
                    help='set truncate input before defined ids only for train for debug only', default=False)

# reasoning args
parser.add_argument('--use_cot', action='store_true', help='use chain of thought examples')
parser.add_argument('--answer_loss_weight', type=float, default=1, help='weight of answer in model loss')
parser.add_argument('--max_cot_steps', type=int, default=None, help='maximum number of cot steps')

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--memory_cell_cls', type=str, default=None, help='cell class for RMT')
parser.add_argument('--recurrent_wrapper_cls', type=str, default=None, help='recurrent wrapper class for RMT')
parser.add_argument('--model_cpt', type=str, default=None, help='pretrained model checkpoint path')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Full experiment checkpoint, used to resume training in SFTTrainer')

# Aydar # RMT args
parser.add_argument('--segment_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--vary_n_segments', action='store_true', default=False,
                    help='Randomly choose segment number from 1 to max_n_segments')
parser.add_argument('--loss_from_last_seg_only', action='store_true', default=False,
                    help='take loss from last segment only')
parser.add_argument('--no_loss_from_first_segment', action='store_true', default=False,
                    help='turn off loss from first segment')
parser.add_argument('--sum_loss', action='store_true', default=False,
                    help='with this flag task loss from all segments is summed')
parser.add_argument('--bptt_depth', type=int, default=-1, help='max num of previous segments in gradient computation.')
parser.add_argument('--segment_ordering', type=str, help='segment order', default='regular',
                    choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
parser.add_argument('--memory_forward_func', type=str, help='path to memory forward funсtion script', default=None)
parser.add_argument('--memory_layers', type=str, help='memory-augmented layer inds or "all" for all layers',
                    default=None)
parser.add_argument('--share_memory_layers', action='store_true', help='share weights of memory layers', default=False)
parser.add_argument('--reconstruction_loss_coef', type=float, default=None,
                    help='reconstuction loss ratio in total loss')
parser.add_argument('--retain_graph', action='store_true', help='Retain computation graph during backward pass',
                    default=False)
parser.add_argument('--use_truncated_backward', action='store_true', default=False,
                    help='whether to use RMT truncated bptt method in backward')
parser.add_argument('--k1', type=int, default=-1,
                    help='(not implemented) If not -1, gradient update is done each k1 segments')
parser.add_argument('--k2', type=int, default=-1, help='number of last segments used by backward')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')
parser.add_argument('--load_optimizer', type=int, default=1, help='load optimizer')
parser.add_argument('--tune_only_memory', action='store_true', default=False,
                    help='Stop training all model weights except memory layer memory')
parser.add_argument('--tune_only_armt', action='store_true', default=False,
                    help='Stop training all model weights except ARMT params')
parser.add_argument('--mask_non_completion', action='store_true', default=False,
                    help='Mask everything except completion in dataset')
# ARMT parameters
parser.add_argument('--d_mem', type=int, default=None, help='number of rows in associative matrix')
parser.add_argument('--layers_attr', type=str, default=None, help='attribute of model, which contains layers')
parser.add_argument('--no_correction', action='store_true', default=False,
                    help='ARMT shmidhuber correction for rewriting')
parser.add_argument('--wrap_pos', action='store_true', default=False,
                    help='Wrap positional encoding for memory tokens (default: False)')

# tokenizer
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')
parser.add_argument('--tokenizer_for_chat_template', type=str, default=None,
                    help='path or name of pre-trained HF Tokenizer, from which CT are used')

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

# LoRA args
parser.add_argument('--use_lora', action='store_true', default=False, help='')
parser.add_argument('--lora_attn_dim', type=int, default=8, help='')
parser.add_argument('--lora_attn_alpha', type=int, default=32, help='')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='')
parser.add_argument('--add_lora_to_armt', action='store_true', default=False, help='')

# Parallel Adapter args
parser.add_argument('--use_adapter', action='store_true', default=False, help='')
parser.add_argument('--adapter_bottleneck_dim', type=int, default=512, help='')
parser.add_argument('--adapter_dropout', type=float, default=0.1, help='')
parser.add_argument('--adapter_scale', type=float, default=4.0, help='')


class RMT_GRPOTrainer(GRPOTrainer):
    """
    A custom GRPOTrainer that is adapted to work with the segment-based input
    required by the RMTForReasoning model.
    """
    def __init__(
        self,
        model: Union["PreTrainedModel", "torch.nn.Module"] = None,
        args: Optional[GRPOConfig] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional["datasets.Dataset"] = None,
        eval_dataset: Optional[Union["datasets.Dataset", Dict[str, "datasets.Dataset"]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **kwargs,
    ):
        # We override the init to accept and store a custom data_collator.
        # The original GRPOTrainer and DPOTrainer do not accept this argument.
        super(DPOTrainer, self).__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs,
        )
        self.data_collator = data_collator

        # The parent class's `_prepare_dataset` method, which we don't use,
        # might have been called. We ensure `_signature_columns` is None to avoid
        # issues with column removal, as our collator handles the data structure.
        self._signature_columns = None

    def get_train_dataloader(self) -> "torch.utils.data.DataLoader":
        """Overrides the default method to use our custom data collator."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=self._get_train_sampler(),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset: Optional["datasets.Dataset"] = None) -> "torch.utils.data.DataLoader":
        """Overrides the default method to use our custom data collator for evaluation."""
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=self._get_eval_sampler(eval_dataset),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def concatenated_forward(
        self, model: "torch.nn.Module", batch: Dict[str, Union[List, "torch.LongTensor"]]
    ) -> Tuple["torch.FloatTensor", "torch.FloatTensor", "torch.FloatTensor", "torch.FloatTensor"]:
        """Overrides the DPO forward pass to work with the RMT model's `segments` input."""
        chosen_outputs = model(segments=batch["chosen_segments"], labels=batch["chosen_labels"])
        rejected_outputs = model(segments=batch["rejected_segments"], labels=batch["rejected_labels"])

        chosen_logps = chosen_outputs.loss.detach() * -1
        rejected_logps = rejected_outputs.loss.detach() * -1

        chosen_logits = chosen_outputs.logits
        rejected_logits = rejected_outputs.logits

        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

if __name__ == '__main__':
    args = parser.parse_args()
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)
    set_seed(args.seed)

    timeout = timedelta(seconds=20 * 1800)
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                                         kwargs_handlers=[accelerate.InitProcessGroupKwargs(timeout=timeout)])
    from accelerate.logging import get_logger
    logger = get_logger('')

    logger.info(f'num processes: {accelerator.num_processes}')
    logger.info(f'mixed precision: {accelerator.mixed_precision}')

    if args.output_dir is None:
        logger.warning('output_dir is not set: config, logs and checkpoints will not be saved.')

    if not args.from_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    if args.tokenizer_for_chat_template is not None:
        it_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_for_chat_template, trust_remote_code=True)
        tokenizer.chat_template = it_tokenizer.chat_template
    if args.padding_side is not None:
        tokenizer.padding_side = args.padding_side
    logger.info(f'preparing dataset for {args.task_name}')

    if args.dataset_name is not None:
        hf_dataset = datasets.load_dataset(args.dataset_name)
        train_dataset = hf_dataset["train"]
        valid_dataset = hf_dataset["valid"]
        if "test" in hf_dataset:
            test_dataset = hf_dataset["test"]
        else:
            test_dataset = None
    else:
        dataset_path = os.path.join(args.dataset_dir, args.task_name)
        train_dataset = datasets.load_from_disk(os.path.join(dataset_path, "train"))
        valid_dataset = datasets.load_from_disk(os.path.join(dataset_path, "valid"))
        if os.path.exists(os.path.join(dataset_path, "test")):
            test_dataset = datasets.load_from_disk(os.path.join(dataset_path, "test"))
        else:
            test_dataset = datasets.load_from_disk(os.path.join(dataset_path, "valid"))

    if args.max_cot_steps is not None:
        train_dataset = train_dataset.filter(lambda x: x['cot_len'] <= args.max_cot_steps)
        valid_dataset = valid_dataset.filter(lambda x: x['cot_len'] <= args.max_cot_steps)
        test_dataset = test_dataset.filter(lambda x: x['cot_len'] <= args.max_cot_steps)
        logger.info(f"Filtered ds sizes: {len(train_dataset), len(valid_dataset), len(test_dataset)}")

    def create_preference_data(sample, task_name):
        prompt = sample['task']
        delim = ">> <<" if 'gsm8k' in task_name else ' + '

        # Chosen data is the ground truth
        chosen_cot_segments = split_cot(sample['cot'], by=delim)
        chosen_label = sample['labels']

        # Rejected data is synthetically generated (e.g., truncated CoT)
        if len(chosen_cot_segments) > 1:
            rejected_cot_segments = chosen_cot_segments[:-1]
            rejected_label = "some wrong answer"
        else:
            rejected_cot_segments = ["I don't know."]
            rejected_label = ""

        return {"prompt": prompt, "chosen_cot_segments": chosen_cot_segments, "chosen_label": chosen_label,
                "rejected_cot_segments": rejected_cot_segments, "rejected_label": rejected_label}

    train_dataset = train_dataset.map(create_preference_data, fn_kwargs={'task_name': args.task_name})
    valid_dataset = valid_dataset.map(create_preference_data, fn_kwargs={'task_name': args.task_name})
    logger.info("Converted SFT dataset to preference dataset format.")

    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if 'gpt2' in args.from_pretrained:
        think = tokenizer.encode('????')
        bos = tokenizer.encode('////')
        ans = tokenizer.encode('!!!!')
    elif 'SmolLM' in args.from_pretrained:
        bos = [tokenizer.bos_token_id]
        think = tokenizer.encode("<issue_start>")
        ans = tokenizer.encode("<issue_closed>")

    eos = [tokenizer.eos_token_id]
    if 'gsm8k' in args.task_name:
        delim = ">> <<"
    elif 'multiplication' in args.task_name:
        delim = ' + '
    else:
        raise NotImplementedError(f"Unknown task name {args.task_name}")

    def collate_fn(batch):
        def prepare_segments_for_sample(prompt_tokens, cot_segments, label_str):
            label_tokens = tokenizer.encode(label_str, add_special_tokens=False)
            cot_segment_tokens = tokenizer.batch_encode_plus(cot_segments, add_special_tokens=False)['input_ids']

            segments = []
            segments.append(make_segment(bos + prompt_tokens + think, loss=False))
            for segment in cot_segment_tokens:
                segments.append(make_segment(bos + segment + think, loss=True))
            segments.append(make_segment(bos + label_tokens + eos, loss=True))
            return segments

        chosen_segments_batch = []
        rejected_segments_batch = []

        for sample in batch:
            prompt_tokens = tokenizer.encode(sample['prompt'], add_special_tokens=False)
            chosen_segments = prepare_segments_for_sample(prompt_tokens, sample['chosen_cot_segments'], sample['chosen_label'])
            chosen_segments_batch.append(chosen_segments)
            rejected_segments = prepare_segments_for_sample(prompt_tokens, sample['rejected_cot_segments'], sample['rejected_label'])
            rejected_segments_batch.append(rejected_segments)

        max_len_chosen = max(len(s) for s in chosen_segments_batch) if chosen_segments_batch else 0
        max_len_rejected = max(len(s) for s in rejected_segments_batch) if rejected_segments_batch else 0
        num_segments = max(max_len_chosen, max_len_rejected)

        def pad_and_collate_segments(segments_batch, target_num_segments):
            for segments in segments_batch:
                if len(segments) < target_num_segments:
                    segments.extend([make_segment(eos, loss=False)] * (target_num_segments - len(segments)))

            batch_segments = []
            for i in range(target_num_segments):
                input_ids = [s[i]['input_ids'] for s in segments_batch]
                attention_mask = [s[i]['attention_mask'] for s in segments_batch]
                labels = [s[i]['labels'] for s in segments_batch]

                input_ids = pad_sequence(input_ids, batch_first=True, padding_value=id_pad_value)
                attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
                labels = pad_sequence(labels, batch_first=True, padding_value=-100)

                batch_segment = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
                batch_segments.append(batch_segment)

            full_labels = torch.cat([s['labels'] for s in batch_segments], dim=1)
            return batch_segments, full_labels

        chosen_segments, chosen_labels = pad_and_collate_segments(chosen_segments_batch, num_segments)
        rejected_segments, rejected_labels = pad_and_collate_segments(rejected_segments_batch, num_segments)

        return {
            "chosen_segments": chosen_segments,
            "chosen_labels": chosen_labels,
            "rejected_segments": rejected_segments,
            "rejected_labels": rejected_labels,
            # The DPO trainer needs these flat lists for logging, even if they are not used in the model forward pass
            "prompt": [s['prompt'] for s in batch],
            "chosen": [s['chosen_label'] for s in batch],
            "rejected": [s['rejected_label'] for s in batch],
        }

    # define model
    # TODO: move model building to separate function
    model_cls = get_cls_by_name(args.model_cls)
    logger.info(f'Using model class: {model_cls}')

    # if args.use_adapter:
    #     raise NotImplementedError('Adapter is not supported for RMT-v4')
    #     model_cfg = AutoConfig.from_pretrained(args.from_pretrained)
    #     model_cfg.use_parallel_adapter = args.use_adapter
    #     model_cfg.parallel_adapter_mode = 'ffn'
    #     model_cfg.adapter_bottleneck_dim = args.adapter_bottleneck_dim
    #     model_cfg.adapter_dropout = args.adapter_dropout
    #     model_cfg.adapter_scale = args.adapter_scale
    #     model = model_cls(config=model_cfg)
    #     logger.info(f'Loading pretrained model: {args.from_pretrained}')
    #     base_model = model_cls.from_pretrained(args.from_pretrained, use_safetensors=False)
    #     model.load_state_dict(base_model.state_dict(), strict=False)
    #     del base_model
    #     logger.info('Added adapters')
    # else:
    #     if args.from_pretrained is None and args.model_cfg is not None:
    #         model_cfg = AutoConfig.from_pretrained(args.model_cfg)
    #         model = model_cls.from_config(model_cfg)
    #     elif args.from_pretrained is not None:
    #         logger.info(f'Loading pretrained model: {args.from_pretrained}')
    #         if "Qwen" in args.from_pretrained or "Llama" in args.from_pretrained:
    #             model = model_cls.from_pretrained(args.from_pretrained,
    #                                               attn_implementation="flash_attention_2",
    #                                               trust_remote_code=True)
    #         else:
    #             model = model_cls.from_pretrained(args.from_pretrained)

    # add LoRA adapters
    if args.use_lora:
        raise NotImplementedError('LoRA is not supported for RMT-v4')
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_attn_dim,
            lora_alpha=args.lora_attn_alpha,
            lora_dropout=args.lora_dropout
            )
        model = get_peft_model(model, peft_config)
        logger.info('Added LoRA, trainable parameters with LoRA only:')
        model.print_trainable_parameters()
    # load cpt of backbone model
    if args.backbone_cpt:
        raise NotImplementedError('Backbone cpt is not supported for RMT-v4')
        if 'bin' in args.backbone_cpt:
            backbone_cpt = args.backbone_cpt
        else:
            backbone_cpt = os.path.join(args.backbone_cpt, "model_best.pth")
        cpt = torch.load(backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'], strict=True)
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')
    if args.num_mem_tokens is not None:
        config = RMTConfig(num_mem_tokens=args.num_mem_tokens, 
                           max_n_segments=10,
                           think_token_id=think[0],
                           answer_token_id=ans[0],
                           bos_token_id=bos[0],
                           eos_token_id=eos[0],
                           d_mem=args.d_mem,
                           wrap_pos=args.wrap_pos,
                           correction=not args.no_correction,
                           layers_attr=args.layers_attr,
                           attend_to_previous_input=args.attend_to_previous_input,
                           segment_size=args.segment_size,
                           k2=args.k2,
                           return_all_logits=False,
                           answer_loss_weight=args.answer_loss_weight
                           )
        model = RMTForReasoning(config)
        # load cpt of rmt
        if args.model_cpt:
            if "safetensors" in args.model_cpt:
                print(model)
                from safetensors.torch import load_model
                load_model(model, args.model_cpt, device="cuda:0")
            else:
                if ".bin" in args.model_cpt:
                    model_cpt = args.model_cpt
                elif "model_best" in os.listdir(args.model_cpt):
                    model_cpt = os.path.join(args.model_cpt, "model_best", "pytorch_model.bin")
                else:
                    dir_files = os.listdir(args.model_cpt)
                    checkpoint_dir = [el for el in dir_files if "checkpoint-" in el][0]
                    model_cpt = os.path.join(args.model_cpt, checkpoint_dir, "pytorch_model.bin")
                cpt = torch.load(model_cpt, map_location='cpu')
                model.load_state_dict(cpt, strict=False)
            logger.info(f'Loaded RMT state dict from: {args.model_cpt}')
            logger.info(f'Trainable parameters after adding RMT/ARMT: {[n for n, p in model.named_parameters() if p.requires_grad]}')
    if args.add_lora_to_armt:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_attn_dim,
            lora_alpha=args.lora_attn_alpha,
            lora_dropout=args.lora_dropout
            )
        # add LoRA only to the inner model
        model.memory_cell.model = get_peft_model(model.memory_cell.model, peft_config)
        logger.info('Added LoRA, trainable parameters with LoRA only:')
        model.memory_cell.model.print_trainable_parameters()
    if args.freeze_model_weights:
        for n, p in model.named_parameters():
            if 'memory' not in n and 'lora' not in n and 'adapter' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        logger.info('Frozen model weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')
    if args.tune_only_memory:
        for n, p in model.named_parameters():
            if 'memory_cell.memory' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        logger.info('Frozen model weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')
    if args.tune_only_armt:
        for n, p in model.named_parameters():
            if 'memory_cell.memory' not in n and 'W_mq' not in n \
                    and 'W_mk' not in n and 'W_mv' not in n and 'W_mb' not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        logger.info('Frozen model weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')

    # fix the not-contiguous error
    def make_contiguous(module):
        with torch.no_grad():
            for param in module.parameters():
                param.set_(param.contiguous())
    make_contiguous(model)
    training_args_dict = {key: value for key, value in vars(args).items() if hasattr(GRPOConfig('.'), key)}

    training_args_dict['remove_unused_columns'] = False
    training_args_dict['save_safetensors'] = False
    training_args_dict['bf16'] = True
    training_args_dict['label_names'] = ['labels']
    training_args_dict['eval_strategy'] = 'steps'
    if training_args_dict.get('per_device_train_batch_size') == 1:
        training_args_dict['per_device_eval_batch_size'] = training_args_dict.get('per_device_train_batch_size')
    else:
        training_args_dict['per_device_eval_batch_size'] = training_args_dict.get('per_device_train_batch_size') // 2
    training_args_dict['eval_accumulation_steps'] = 32
    if args.d_mem is None:
        # for now, gradient checkpointing doesn't supported for ARMT
        training_args_dict['gradient_checkpointing'] = True
        training_args_dict['gradient_checkpointing_kwargs'] = {'use_reentrant': False}
    training_args_dict['log_level'] = 'debug'
    training_args_dict['load_best_model_at_end'] = args.early_stopping_patience != -1

    # training_args_dict['dataset_kwargs'] = {"skip_prepare_dataset": True}

    # if args.num_mem_tokens is not None:
    #     # fix max_seq_length warning
    #     training_args_dict["max_seq_length"] = args.segment_size

    training_args = GRPOConfig(**training_args_dict)

    # The compute_accuracy and custom optimizer/scheduler are not directly compatible
    # with the GRPOTrainer's preference-based evaluation and internal optimizer setup.
    # They are removed for this example.

    trainer = RMT_GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )
    logger.info(f"Trainer Gradient Checkpointing Enabled: {getattr(trainer.args, 'gradient_checkpointing', False)}")
    if args.early_stopping_patience != -1:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )
        trainer.add_callback(early_stopping)
    # start_metrics = trainer.evaluate()
    # logger.info(f"Metrics of initial model: {start_metrics}")
    if not args.validate_only:
        trainer.train(resume_from_checkpoint=args.checkpoint)
