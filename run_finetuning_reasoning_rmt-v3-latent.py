import logging
import os
from pathlib import Path
from datetime import timedelta
import torch
import numpy as np
import datasets
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, set_seed
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
# from lm_experiments_tools.dataset_preprocessing import load_and_preprocess_task, combine_datasets
# from lm_experiments_tools.instruction_utils import mask_non_completion, mask_non_completion_multi

from torch.nn.utils.rnn import pad_sequence

import accelerate
from peft import get_peft_model, LoraConfig, TaskType

# import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402

# from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run  # noqa: E402
from lm_experiments_tools.utils import get_cls_by_name

from utils.reasoning import make_segment, split_cot

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')


# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
# torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...
# torch.cuda.set_device(hvd.local_rank())

parser = HfArgumentParser(SFTConfig)
# parser.add_argument('--task_names', type=str, help="Task names, separated by : , wikitext:arxiv: ...")
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
parser.add_argument('--n_latent', type=int, default=0, help='number of latent reasoning steps in between segments')

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
# parser.add_argument('--segment_ordering', type=str,help='????', default='regular',
#                     choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
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


if __name__ == '__main__':
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)
    set_seed(args.seed)

    # workaround with setting bigger tiomeout for NCCL (useful for big dataset, to avoid timeout at tokenization)
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
    # Prepare datasets
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
        # first, we segment each sample into task, cot steps and labels
        segments_batch = []
        for sample in batch:
            task, lab, cot = sample['task'], sample['labels'], sample['cot']
            task_tokens = tokenizer.encode(task, add_special_tokens=False)
            labels_tokens = tokenizer.encode(lab, add_special_tokens=False)
            if getattr(args, 'use_cot', False):
                cot_segments = split_cot(cot, by=delim)
            else:
                cot_segments = [cot]
            cot_segment_tokens = tokenizer.batch_encode_plus(cot_segments, add_special_tokens=False)['input_ids']

            segments = []
            segments.append(make_segment(bos + task_tokens + think, loss=False))
            for _ in range(args.n_latent):
                segments.append(make_segment(bos + think, loss=False))
            for segment in cot_segment_tokens[:-1]:
                segments.append(make_segment(bos + segment + think, loss=True))
                for _ in range(args.n_latent):
                    segments.append(make_segment(bos + think, loss=False))
            segments.append(make_segment(bos + cot_segment_tokens[-1] + ans, loss=True))
            for _ in range(args.n_latent):
                segments.append(make_segment(bos + ans, loss=False))

            segments.append(make_segment(bos + labels_tokens + eos, loss=True))
            segments_batch.append(segments)

        # if some samples have less segments than others, we pad them with empty segments
        num_segments = max(len(segments) for segments in segments_batch)
        for segments in segments_batch:
            if len(segments) < num_segments:
                segments.extend([make_segment(eos, loss=False)] * (num_segments - len(segments)))

        # prepare segments for the whole batch
        batch_segments = []
        for i in range(num_segments):
            input_ids = [s[i]['input_ids'] for s in segments_batch]
            attention_mask = [s[i]['attention_mask'] for s in segments_batch]
            labels = [s[i]['labels'] for s in segments_batch]
            labels_mask = [s[i]['labels_mask'] for s in segments_batch]

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=id_pad_value)
            attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
            labels = pad_sequence(labels, batch_first=True, padding_value=-100)
            labels_mask = pad_sequence(labels_mask, batch_first=True, padding_value=False)

            batch_segment = {'input_ids': input_ids,
                             'attention_mask': attention_mask,
                             'labels_mask': labels_mask,
                             'labels': labels
                             }
            batch_segments.append(batch_segment)
        full_labels = torch.cat([s['labels'] for s in batch_segments], dim=1)
        return {"segments": batch_segments, 'labels': full_labels}

    # define model
    # TODO: move model building to separate function
    model_cls = get_cls_by_name(args.model_cls)
    logger.info(f'Using model class: {model_cls}')

    if args.use_adapter:
        model_cfg = AutoConfig.from_pretrained(args.from_pretrained)

        model_cfg.use_parallel_adapter = args.use_adapter
        model_cfg.parallel_adapter_mode = 'ffn'
        model_cfg.adapter_bottleneck_dim = args.adapter_bottleneck_dim
        model_cfg.adapter_dropout = args.adapter_dropout
        model_cfg.adapter_scale = args.adapter_scale

        model = model_cls(config=model_cfg)

        logger.info(f'Loading pretrained model: {args.from_pretrained}')
        base_model = model_cls.from_pretrained(args.from_pretrained, use_safetensors=False)

        model.load_state_dict(base_model.state_dict(), strict=False)
        del base_model
        logger.info('Added adapters')
    else:
        # TODO: fix if for Qwen and Llama
        if not args.from_pretrained:
            model_cfg = AutoConfig.from_pretrained(args.model_cfg)
            model = model_cls.from_config(model_cfg)
        else:
            logger.info(f'Loading pretrained model: {args.from_pretrained}')
            if "Qwen" in args.from_pretrained or "Llama" in args.from_pretrained:
                model = model_cls.from_pretrained(args.from_pretrained,
                                                  attn_implementation="flash_attention_2",
                                                  torch_dtype=torch.bfloat16,
                                                  trust_remote_code=True)
            else:
                model = model_cls.from_pretrained(args.from_pretrained)     # use_safetensors=False)
    if args.use_lora:
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
        backbone_cpt = os.path.join(args.backbone_cpt, "model_best.pth")
        cpt = torch.load(backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'], strict=True)
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')
    # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None:
        memory_cell_cls = get_cls_by_name(args.memory_cell_cls)
        recurrent_wrapper_cls = get_cls_by_name(args.recurrent_wrapper_cls)
        logger.info(f'Wrapping in: {memory_cell_cls} and {recurrent_wrapper_cls}')
        mem_cell_args = dict(
            base_model=model,
            num_mem_tokens=args.num_mem_tokens,
        )
        # additional parameters for ARMT model
        if args.d_mem is not None:
            mem_cell_args['d_mem'] = args.d_mem
            mem_cell_args['wrap_pos'] = args.wrap_pos
            mem_cell_args['correction'] = not args.no_correction
            # mem_cell_args['use_lora'] = args.use_lora
        if args.layers_attr is not None:
            mem_cell_args['layers_attr'] = args.layers_attr
        if args.attend_to_previous_input:
            mem_cell_args['attend_to_previous_input'] = args.attend_to_previous_input
        cell = memory_cell_cls(**mem_cell_args)
        model = recurrent_wrapper_cls(cell,
                                      segment_size=args.segment_size,
                                      max_n_segments=args.max_n_segments,
                                      vary_n_segments=args.vary_n_segments,
                                      k2=args.k2,
                                      attend_to_previous_input=args.attend_to_previous_input,
                                      return_all_logits=False,
                                      answer_loss_weight=args.answer_loss_weight
                                      )
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
        # print(model)
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
    training_args_dict = {key: value for key, value in vars(args).items() if hasattr(SFTConfig('.'), key)}

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

    training_args_dict['dataset_kwargs'] = {"skip_prepare_dataset": True}

    if args.num_mem_tokens is not None:
        # fix max_seq_length warning
        # training_args_dict["max_seq_length"] = args.sample_size
        model.to(torch.bfloat16)
    training_args = SFTConfig(**training_args_dict)

    def compute_accuracy(eval_pred):
        preds = eval_pred.predictions.argmax(axis=-1)[:, :-1]
        labels = eval_pred.label_ids[:, 1:]

        labels_masks = labels > 0
        preds_full = [p[m] for p, m in zip(preds, labels_masks)]
        labels_full = [lab[m] for lab, m in zip(labels, labels_masks)]

        special_tokens = {ans[0], bos[0]}
        acc_cot, acc_ans = [], []
        for lab_tokens, pred_tokens in zip(labels_full, preds_full):
            ans_start_index = max(i for i, x in enumerate(lab_tokens) if x == ans[0])

            pred_cot_tokens = pred_tokens[:ans_start_index].tolist()
            lab_cot_tokens = lab_tokens[:ans_start_index].tolist()

            cot_correct = [p == l for p, l in zip(pred_cot_tokens, lab_cot_tokens) if l not in special_tokens]
            acc_cot.append(all(cot_correct))

            pred_ans_tokens = pred_tokens[ans_start_index:].tolist()
            lab_ans_tokens = lab_tokens[ans_start_index:].tolist()

            ans_correct = [p == l for p, l in zip(pred_ans_tokens, lab_ans_tokens) if l not in special_tokens]
            acc_ans.append(all(ans_correct))

        return {'accuracy_cot': np.mean(acc_cot), 'accuracy_ans': np.mean(acc_ans)}

    def lr_lambda(current_step):
        if current_step < training_args.warmup_steps:
            return current_step / training_args.warmup_steps
        if args.lr_scheduler_type == "linear":
            decay_factor = (training_args.max_steps - current_step) / (training_args.max_steps - training_args.warmup_steps)
            return max(args.min_lr / training_args.learning_rate, decay_factor)
        elif args.lr_scheduler_type == "constant":
            return 1.0
        else:
            raise ValueError("Unsupported lr_scheduler_type")

    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_accuracy,
        optimizers=(optimizer, scheduler)
    )
    logger.info(f"Trainer Gradient Checkpointing Enabled: {trainer.args.gradient_checkpointing}")
    if args.early_stopping_patience != -1:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        )
        trainer.add_callback(early_stopping)
    # start_metrics = trainer.evaluate()
    # logger.info(f"Metrics of initial model: {start_metrics}")
    if not args.validate_only:
        trainer.train(resume_from_checkpoint=args.checkpoint)
