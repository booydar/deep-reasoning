import importlib
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModelForCausalLM
# from lm_experiments_tools.utils import get_cls_by_name


def get_cls_by_name(name: str) -> type:
    """Get class by its name and module path.

    Args:
        name (str): e.g., transfomers:T5ForConditionalGeneration, modeling_t5:my_class

    Returns:
        type: found class for `name`
    """
    module_name, cls_name = name.split(':')
    return getattr(importlib.import_module(module_name), cls_name)


class RMTConfig(PretrainedConfig):
    model_type = "rmt"

    def __init__(self,
                 base_model_name="HuggingFaceTB/SmolLM2-135M",
                 num_mem_tokens=16,
                 max_n_segments=10,
                 think_token_id=None,
                 answer_token_id=None,
                 bos_token_id=None,
                 eos_token_id=None,
                 memory_cell_cls='modeling_rmt.language_modeling:MemoryCell',
                 recurrent_wrapper_cls='modeling_rmt.experimental:RecurrentWrapperNoSegmentationGenerate',
                 **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_mem_tokens = num_mem_tokens
        self.max_n_segments = max_n_segments
        self.think_token_id = think_token_id
        self.answer_token_id = answer_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.memory_cell_cls = memory_cell_cls
        self.recurrent_wrapper_cls = recurrent_wrapper_cls

    def get(self, attr: str, default=None):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return default


class RMTForReasoning(PreTrainedModel):
    config_class = RMTConfig

    def __init__(self, config: RMTConfig, **kwargs):
        super().__init__(config, **kwargs)
        from transformers import AutoConfig, AutoModelForCausalLM
        base_config = AutoConfig.from_pretrained(config.base_model_name)
        base_model = AutoModelForCausalLM.from_config(base_config)

        memory_cell_cls = get_cls_by_name(config.memory_cell_cls)
        recurrent_wrapper_cls = get_cls_by_name(config.recurrent_wrapper_cls)

        self.rmt_config = config
        memory_cell = memory_cell_cls(base_model, num_mem_tokens=config.num_mem_tokens)
        self.rmt = recurrent_wrapper_cls(
            memory_cell,
            max_n_segments=config.max_n_segments,
            think_token_id=config.think_token_id,
            answer_token_id=config.answer_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id
        )

    def forward(self, *args, **kwargs):
        return self.rmt(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.rmt.generate(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        try:
            return super().load_state_dict(state_dict, strict, assign)
        except RuntimeError:
            print("Failed to load state, retrying with RMT loader.")
            self.rmt.load_state_dict(state_dict, strict=True, assign=assign)
            print("Success!")

    def save_pretrained(
        self,
        save_directory: str,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory.

        This method extends the default `save_pretrained` to also save the source code
        of dynamically loaded classes like the memory cell and recurrent wrapper,
        allowing the model to be loaded with `trust_remote_code=True`.
        """
        import os
        import shutil
        import inspect
        from pathlib import Path

        # First, let the parent class do its thing. This will save weights, config,
        # and the source file for RMTForReasoning itself because it's registered.
        super().save_pretrained(save_directory, **kwargs)

        # Now, find and copy the source code for other custom classes from the config.
        module_names_to_copy = set()
        for attr in ["memory_cell_cls", "recurrent_wrapper_cls"]:
            class_string = getattr(self.config, attr, None)
            if class_string and ":" in class_string:
                module_name, _ = class_string.split(":", 1)
                module_names_to_copy.add(module_name)

        for module_name in module_names_to_copy:
            try:
                module = importlib.import_module(module_name)
                source_path = inspect.getsourcefile(module)
                
                if source_path is None:
                    print(f"Warning: Could not find source for module '{module_name}'. Skipping.")
                    continue

                # Determine destination path, preserving module structure.
                relative_path = module_name.replace(".", os.sep) + ".py"
                dest_path = os.path.join(save_directory, relative_path)

                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(source_path, dest_path)
                print(f"Saved custom code file for '{module_name}': {dest_path}")

                # Create any necessary __init__.py files to make packages importable
                path_parts = module_name.split('.')
                for i in range(1, len(path_parts)):
                    package_path = os.path.join(save_directory, *path_parts[:i])
                    init_file = os.path.join(package_path, "__init__.py")
                    if not os.path.exists(init_file):
                        Path(init_file).touch()
            except Exception as e:
                print(f"Warning: Could not save source for module '{module_name}'. Error: {e}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, *args, **kwargs):
        from transformers.utils.hub import cached_file, HfHubHTTPError
        import torch

        if config is None:
            config = RMTConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        model = cls(config, *args, **kwargs)

        state_dict = None
        try:
            weights_path = cached_file(pretrained_model_name_or_path, "model.safetensors", **kwargs)
            from safetensors.torch import load_file
            state_dict = load_file(weights_path, device="cpu")
        except (OSError, HfHubHTTPError):
            try:
                weights_path = cached_file(pretrained_model_name_or_path, "pytorch_model.bin", **kwargs)
                state_dict = torch.load(weights_path, map_location="cpu")
            except (OSError, HfHubHTTPError):
                print(f"Warning: Could not find weights for {pretrained_model_name_or_path}. "
                      f"The model is initialized randomly.")

        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)

        return model

# Register the custom model and config with the Auto-classes.
# This allows users to load it automatically with `AutoModelForCausalLM.from_pretrained(...)`
AutoConfig.register("rmt", RMTConfig)
AutoModelForCausalLM.register(RMTConfig, RMTForReasoning)
