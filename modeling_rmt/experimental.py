import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class MemoryCellSmart(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens):
        super().__init__()
        self.model = base_model
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim = getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

    def set_memory(self, input_shape):
        return self.memory.repeat(input_shape[0], 1, 1)
    
    def put_tensor_by_mask(self, inputs_embeds, memory_state, mem_mask):
        bs, N, H = inputs_embeds.shape

        for i in range(bs):
            inputs_embeds[i, mem_mask[i]] = memory_state[i]
        
        return inputs_embeds
    
    def extract_tensor_by_mask(self, outputs, mask):
        bs, N, H = outputs.shape
        M = mask.sum(dim=1)[0].item()

        extracted = outputs[mask]
        return extracted.view(bs, M, H)

    def process_input(
            self,
            input_ids,
            memory_state,
            write_mem,
            read_mem_mask=None,
            write_mem_mask=None,
            **kwargs
        ):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        else:
            raise ValueError("inputs_embeds is not supported for memory cells") # test "if"

        if read_mem_mask is not None:
            inputs_embeds = self.put_tensor_by_mask(inputs_embeds, memory_state, read_mem_mask)
        if write_mem and write_mem_mask is not None:
            inputs_embeds = self.put_tensor_by_mask(inputs_embeds, memory_state, write_mem_mask)
        
        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        seg_kwargs['attention_mask'] = kwargs['attention_mask']
        seg_kwargs['output_hidden_states'] = True
        return seg_kwargs

    def forward(
            self,
            input_ids,
            memory_state=None,
            text_mask=None,
            read_mem_mask=None,
            write_mem_mask=None,
            **kwargs
        ):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(
            input_ids,
            memory_state,
            write_mem=True,
            read_mem_mask=read_mem_mask,
            write_mem_mask=write_mem_mask,
            **kwargs
        )
        out = self.model(**seg_kwargs)
        out, new_memory_state = self.process_output(
            out,
            text_mask=text_mask,
            write_mem_mask=write_mem_mask,
            **kwargs
        )

        return out, new_memory_state

    def generate(
        self,
        input_ids,
        memory_state,
        attention_mask=None,
        read_mem_mask=None,
        write_mem_mask=None,
        **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(
            input_ids,
            memory_state,
            attention_mask=attention_mask,
            write_mem=False,
            read_mem_mask=read_mem_mask,
            write_mem_mask=write_mem_mask,
        )
        
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'], attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        return out

    def process_output(self, model_outputs, text_mask, write_mem_mask, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = self.extract_tensor_by_mask(model_outputs.hidden_states[-1], write_mem_mask)
            out['logits'] = self.extract_tensor_by_mask(model_outputs.logits, text_mask)

            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [self.extract_tensor_by_mask(lh, text_mask)
                                        for lh in model_outputs.hidden_states]

            if kwargs.get('output_attentions'):
                print(model_outputs['attentions'].shape)
                out['attentions'] = model_outputs['attentions']
        else:
            memory_state = None
            out = model_outputs

        return out, memory_state


class RecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs

    def process_outputs(self, cell_outputs, segments, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        proxy_out = {}
        for seg_num, segment in enumerate(segments):
            cell_out = cell_outputs[seg_num]

            full_logits = cell_out.logits

            labels = segment.get('labels')
            if labels is not None:
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = full_logits[..., :-1, :].contiguous()
                flat_labels = shift_labels.view(-1)
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))

                loss_fct = CrossEntropyLoss()
                labels_mask = segment.get('labels_mask')
                if labels_mask is not None:
                    shift_mask = labels_mask[..., :-1].contiguous()

                    flat_labels = flat_labels[shift_mask.view(-1)]
                    flat_logits = flat_logits[shift_mask.view(-1)]

                    if labels_mask.sum() == 0:
                        loss_value = 0
                    else:
                        loss_value = loss_fct(flat_logits, flat_labels)

                proxy_out[f'loss_{seg_num}'] = loss_value
            else:
                proxy_out[f'loss_{seg_num}'] = 0

            segment_keys = ['loss']
            if kwargs.get('output_attentions'):
                segment_keys.append('attentions')
            if kwargs.get('output_hidden_states'):
                segment_keys.append('hidden_states')

            for key, value in cell_out.items():
                if any([sk in key for sk in segment_keys]):
                    proxy_out[f'{key}_{seg_num}'] = value

        num_segments = len(segments)
        out['loss'] = sum([proxy_out[f'loss_{seg_num}'] for seg_num in range(num_segments)]) / num_segments
        out['logits'] = torch.cat([cell_out.logits for cell_out in cell_outputs], dim=1)

        return out
    
    def forward(self, segments, labels, output_attentions=None, output_hidden_states=None):
        memory_state = None

        cell_outputs = []
        for seg_num, segment in enumerate(segments):
            cell_out, memory_state = self.memory_cell(
                input_ids=segment['input_ids'],
                attention_mask=segment['attention_mask'],
                text_mask=segment['text_mask'],
                read_mem_mask=segment['read_mem_mask'],
                write_mem_mask=segment['write_mem_mask'],
                memory_state=memory_state,
                output_hidden_states=True,
                )
            cell_outputs.append(cell_out)
            self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(
            cell_outputs,
            segments,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        return out

    def generate(self, segments, **kwargs):
        memory_state = None

        for seg_num, segment in enumerate(segments):
            cell_out, memory_state = self.memory_cell(
                input_ids=segment['input_ids'],
                attention_mask=segment['attention_mask'],
                text_mask=segment['text_mask'],
                read_mem_mask=segment['read_mem_mask'],
                write_mem_mask=segment['write_mem_mask'],
                memory_state=memory_state,
                output_hidden_states=True
            )

        generated_segments = []
        for seg_num in range(len(segments), self.rmt_config.get("max_n_segments", 32)):
            output_ids, memory_state = self.generate_segment(memory_state=memory_state, **kwargs)
            generated_segments.append(output_ids)

            if self.all_done(generated_segments):
                break

        return generated_segments

    def generate_segment(self, memory_state, **kwargs):
        input_ids, read_mem_mask = self.get_bos_tensor(memory_state)
        attention_mask = torch.ones_like(input_ids).bool()

        generated = self.memory_cell.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_state=memory_state,
            read_mem_mask=read_mem_mask,
            eos_token_id=[
                self.rmt_config['eos_token_id'],
                self.rmt_config['think_token_id'],
                self.rmt_config['answer_token_id']
            ],
            **kwargs
        )

        # Update memory state from generation
        fwd_inputs = torch.cat((input_ids, generated, memory_state), dim=1)
        _, memory_state = self.memory_cell(input_ids=fwd_inputs, memory_state=memory_state)

        return generated, memory_state

    def get_bos_tensor(self, memory_state):
        mem = self.rmt_config["mem_token_id"]
        bos = self.rmt_config["bos_token_id"]
        mem_tensor = torch.tensor([mem] * self.memory_cell.num_mem_tokens).reshape(1, -1)
        mem_tensor = mem_tensor.repeat(memory_state.shape[0], 1)
        bos_tensor = torch.tensor([bos] * memory_state.shape[0]).reshape(-1, 1)
        input_tensor = torch.cat([mem_tensor, bos_tensor], dim=1)

        read_mem_mask = torch.zeros(input_tensor.shape, dtype=torch.bool)
        read_mem_mask[:, :self.memory_cell.num_mem_tokens] = True
        return input_tensor.to(memory_state.device), read_mem_mask.to(memory_state.device)

    def all_done(self, generated_segments):
        eos = self.rmt_config['eos_token_id']
        bs = generated_segments[0].shape[0]
        have_eos = [any([eos in seg[i] for seg in generated_segments]) for i in range(bs)]
        all_done = all(have_eos)
        return all_done

    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
                or seg_num + k2 > max_n_segments:
            return memory_state

        memory_state = memory_state.detach()
        return memory_state

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.memory_cell.model, "gradient_checkpointing_enable"):
            return self.memory_cell.model.gradient_checkpointing_enable(*args, **kwargs)


class RecurrentWrapperNoSegmentation(RecurrentWrapper):
    def forward(self, segments, labels=None, output_attentions=None, output_hidden_states=None):
        memory_state = None

        cell_outputs = []
        for seg_num, segment in enumerate(segments):
            cell_out, memory_state = self.memory_cell(
                input_ids=segment['input_ids'],
                attention_mask=segment['attention_mask'],
                memory_state=memory_state,
                output_hidden_states=True
            )
            cell_outputs.append(cell_out)
            memory_state = self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(cell_outputs, segments,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states)
        return out

    def generate(self, segments, **generate_kwargs):
        # raise NotImplementedError("Generation not implemented for this wrapper.")
        memory_state = None
        for seg_num, segment in enumerate(segments[:-1]):
            cell_out, memory_state = self.memory_cell(
                input_ids=segment['input_ids'],
                attention_mask=segment['attention_mask'],
                memory_state=memory_state,
                output_hidden_states=True
            )

        final_segment = segments[-1]
        out = self.memory_cell.generate(**final_segment, memory_state=memory_state, **generate_kwargs)

        return out

    def process_outputs(self, cell_outputs, segments, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        proxy_out = {}
        for seg_num, segment in enumerate(segments):
            cell_out = cell_outputs[seg_num]

            full_logits = cell_out.logits

            labels = segment.get('labels')
            if labels is not None:
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = full_logits[..., :-1, :].contiguous()
                flat_labels = shift_labels.view(-1)
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))

                loss_fct = CrossEntropyLoss()
                labels_mask = segment.get('labels_mask')
                if labels_mask is not None:
                    shift_mask = labels_mask[..., :-1].contiguous()

                    flat_labels = flat_labels[shift_mask.view(-1)]
                    flat_logits = flat_logits[shift_mask.view(-1)]

                    if labels_mask.sum() == 0:
                        loss_value = 0
                    else:
                        loss_value = loss_fct(flat_logits, flat_labels)

                proxy_out[f'loss_{seg_num}'] = loss_value
            else:
                proxy_out[f'loss_{seg_num}'] = 0

            segment_keys = ['loss']
            if kwargs.get('output_attentions'):
                segment_keys.append('attentions')
            if kwargs.get('output_hidden_states'):
                segment_keys.append('hidden_states')

            for key, value in cell_out.items():
                if any([sk in key for sk in segment_keys]):
                    proxy_out[f'{key}_{seg_num}'] = value

        num_segments = len(segments)
        out['loss'] = sum([proxy_out[f'loss_{seg_num}'] for seg_num in range(num_segments)]) / num_segments
        out['logits'] = torch.cat([cell_out.logits for cell_out in cell_outputs], dim=1)

        return out

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.memory_cell.model, "gradient_checkpointing_enable"):
            return self.memory_cell.model.gradient_checkpointing_enable(*args, **kwargs)



class RecurrentWrapperNoSegmentationWeighLoss(RecurrentWrapperNoSegmentation):
    def process_outputs(self, cell_outputs, segments, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        proxy_out = {}
        for seg_num, segment in enumerate(segments):
            cell_out = cell_outputs[seg_num]

            full_logits = cell_out.logits

            labels = segment.get('labels')
            if labels is not None:
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = full_logits[..., :-1, :].contiguous()
                flat_labels = shift_labels.view(-1)
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))

                loss_fct = CrossEntropyLoss()
                labels_mask = segment.get('labels_mask')
                if labels_mask is not None:
                    shift_mask = labels_mask[..., :-1].contiguous()

                    flat_labels = flat_labels[shift_mask.view(-1)]
                    flat_logits = flat_logits[shift_mask.view(-1)]

                    if labels_mask.sum() == 0:
                        loss_value = 0
                    else:
                        loss_value = loss_fct(flat_logits, flat_labels)

                if seg_num == len(segments) - 1:
                    answer_weight = self.rmt_config.get("answer_loss_weight", 1)
                    loss_value *= answer_weight

                proxy_out[f'loss_{seg_num}'] = loss_value
            else:
                proxy_out[f'loss_{seg_num}'] = 0

            segment_keys = ['loss']
            if kwargs.get('output_attentions'):
                segment_keys.append('attentions')
            if kwargs.get('output_hidden_states'):
                segment_keys.append('hidden_states')

            for key, value in cell_out.items():
                if any([sk in key for sk in segment_keys]):
                    proxy_out[f'{key}_{seg_num}'] = value

        num_segments = len(segments)
        out['loss'] = sum([proxy_out[f'loss_{seg_num}'] for seg_num in range(num_segments)]) / num_segments
        out['logits'] = torch.cat([cell_out.logits for cell_out in cell_outputs], dim=1)
        # print(out.keys(), out.loss)

        return out

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.memory_cell.model, "gradient_checkpointing_enable"):
            return self.memory_cell.model.gradient_checkpointing_enable(*args, **kwargs)


from transformers import StoppingCriteria

class StopOnSpecialTokenCriteria(StoppingCriteria):
    def __init__(self, special_token_ids):
        self.special_token_ids = set(special_token_ids)

    def __call__(self, input_ids, scores, **kwargs):
        last_token = input_ids[0, -1].item()
        return last_token in self.special_token_ids


class RecurrentWrapperNoSegmentationGenerate(RecurrentWrapperNoSegmentation):
    def forward(self, segments, labels, output_attentions=None, output_hidden_states=None):
        memory_state = None

        cell_outputs = []
        for seg_num, segment in enumerate(segments):
            cell_out, memory_state = self.memory_cell(
                input_ids=segment['input_ids'],
                attention_mask=segment['attention_mask'],
                memory_state=memory_state,
                output_hidden_states=True,
                )
            cell_outputs.append(cell_out)
            self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(cell_outputs, segments,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states)
        return out

    def generate(self, segments, **kwargs):
        memory_state = None

        for seg_num, segment in enumerate(segments):
            cell_out, memory_state = self.memory_cell(
                input_ids=segment['input_ids'],
                attention_mask=segment['attention_mask'],
                memory_state=memory_state,
                output_hidden_states=True
            )

        generated_segments = []
        for seg_num in range(len(segments), self.rmt_config.get("max_n_segments", 32)):
            output_ids, memory_state = self.generate_segment(memory_state=memory_state, **kwargs)
            generated_segments.append(output_ids)

            if self.all_done(generated_segments):
                break

        return generated_segments

    def generate_segment(self, memory_state, **kwargs):
        input_ids = self.get_bos_tensor(memory_state)
        attention_mask = torch.ones_like(input_ids).bool()

        generated = self.memory_cell.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_state=memory_state,
            eos_token_id=[
                self.rmt_config['eos_token_id'],
                self.rmt_config['think_token_id'],
                self.rmt_config['answer_token_id']
            ],
            **kwargs
        )

        # Update memory state from generation
        fwd_inputs = torch.cat((input_ids, generated), dim=1)
        _, memory_state = self.memory_cell(input_ids=fwd_inputs, memory_state=memory_state)

        return generated, memory_state

    def get_bos_tensor(self, memory_state):
        bos = self.rmt_config["bos_token_id"]
        bos_tensor = torch.tensor([bos] * memory_state.shape[0]).reshape(-1, 1)
        return bos_tensor.to(memory_state.device)

    def all_done(self, generated_segments):
        eos = self.rmt_config['eos_token_id']
        bs = generated_segments[0].shape[0]
        have_eos = [any([eos in seg[i] for seg in generated_segments]) for i in range(bs)]
        all_done = all(have_eos)
        return all_done

    def make_custom_stopping_criteria(self):
        return [StopOnSpecialTokenCriteria([self.rmt_config['think_token_id'], self.rmt_config['answer_token_id']])]
