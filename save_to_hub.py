from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from modeling_rmt.language_modeling import MemoryCell
from modeling_rmt.experimental import RecurrentWrapperNoSegmentationGenerate


device = 'cuda'
model_name = "HuggingFaceTB/SmolLM2-135M"
checkpoint_path = "../../models/RMT_SmolLM2-135M/cot/checkpoint-3300/pytorch_model.bin"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
bos = [tokenizer.bos_token_id]
eos = [tokenizer.eos_token_id]
think = tokenizer.encode("<issue_start>")
ans = tokenizer.encode("<issue_closed>")

delim = ">> <<"


memory_cell = MemoryCell(
    model,
    num_mem_tokens=16
)

model = RecurrentWrapperNoSegmentationGenerate(memory_cell, 
                                             max_n_segments=10, 
                                             think_token_id=think[0],
                                             answer_token_id=ans[0],
                                             bos_token_id=bos[0],
                                             eos_token_id=eos[0]
                                             )

model.load_state_dict(torch.load(checkpoint_path), strict=False)

model.push_to_hub("okashurin/RMT-SmolLM2-135M")