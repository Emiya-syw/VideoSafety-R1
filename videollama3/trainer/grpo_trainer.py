# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import sys
sys.path.append('./')
sys.path.append('../')
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import is_conversational
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from trl import GRPOConfig
# torch-related packages
# NOTE: torch must be imported before transformers. Otherwise, `Segmentation fault (core dumped)` will occur.
import torch
import transformers
from packaging import version
from torch.utils.data import Dataset
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
# from train_grpo_vidlm3 import LazySupervisedDataset, DataCollatorWithFlatteningForSupervisedDataset, DataCollatorForSupervisedDataset
from packaging import version
from dataclasses import dataclass, field
from typing import Optional

from videollama3.constants import (
    NUM_FRAMES, DEFAULT_IMAGE_TOKEN,
    STREAM_START_TOKEN, STREAM_END_TOKEN)
from videollama3.mm_utils import load_images, load_video
from videollama3.model import *
from videollama3.videollama3_trainer import find_all_linear_names
from videollama3.model.alarm_tokens import configure_alarm_tokens
from videollama3.model.processor import Videollama3Processor
import copy
if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def int_with_none(value):
    if value == 'None':
        return None
    return int(value)

@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model checkpoint for weights initialization."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Specific model version to use. It can be a branch name, a tag name, or a commit id."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": "Whether to use PEFT for training."},
    )
    lora_enable: bool = False
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA R value."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    lora_modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Model layers to unfreeze & train."},
    )
    lora_task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "Task type to pass for LoRA (use 'SEQ_CLS' for reward modeling)."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, "
            "instead of the original default value of `lora_alpha/r`."
        },
    )
    use_dora: bool = field(
        default=False,
        metadata={
            "help": "Enable Weight-Decomposed Low-Rank Adaptation (DoRA). This technique decomposes the updates of "
            "the weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
            "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
            "especially at low ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a "
            "bigger overhead than pure LoRA, so it is recommended to merge weights for inference."
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8 bit precision for the base model. Works only with LoRA."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4 bit precision for the base model. Works only with LoRA."},
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type.", "choices": ["fp4", "nf4"]},
    )
    use_bnb_nested_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use nested quantization."},
    )

    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    # Connector Arguments
    mm_projector_type: Optional[str] = field(default='linear')
    pretrain_mm_projector: Optional[str] = field(default=None)
    # Vision tower Arguments
    vision_encoder: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_attn_implementation: Optional[str] = field(default="flash_attention_2")
    # freeze_vision_encoder: bool = field(default=False, metadata={"help": "Whether to freeze the Vision Encoder."})
    # Token downsampling Arguments
    use_token_compression: Optional[bool] = field(default=False)
    use_flash_loss: Optional[bool] = field(default=False)

    # Training learning rate Arguments
    vision_encoder_lr: Optional[float] = None
    mm_projector_lr: Optional[float] = None
    llm_lr: Optional[float] = None
    # Training Data Arguments
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )

    # Loading Arguments
    is_multimodal: bool = False
    fps: Optional[int] = field(default=None)
    max_frames: Optional[int_with_none] = field(default=None)
    # Preprocess Arguments
    image_merge_size: Optional[int] = field(default=1)
    video_merge_size: Optional[int] = field(default=1)
    mm_max_length: Optional[int] = field(default=10240)
    image_aspect_ratio: str = 'square'
    use_batch_flattening: bool = field(default=True, metadata={"help": "Whether to flatten the in-batch sequences of variable lengths."})
    dataset_cache_dir: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default="videollama3", metadata={"help": "Model type selected in the list: " + ", ".join(VLLMs.keys())})
    # bf16: bool = field(
    #     default=False,
    #     metadata={
    #         "help": (
    #             "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
    #             " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
    #         )
    #     },
    # )
    # fp16: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    # )

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        if hasattr(self.lora_target_modules, "__len__") and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]

    
class VidLM3GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        model_args = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
            
        self.script_args = script_args
        self.model_args = model_args
        self.args = args

        print(args.device)
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            # torch_dtype = model_init_kwargs.get("torch_dtype")
            torch_dtype = model_args.torch_dtype
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            self.torch_dtype = torch_dtype
            # print(torch_dtype)
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "videollama3" or "VideoLLaMA3" in model_id:

                bnb_model_from_pretrained_args = {}
                if model_args.bits in [4, 8]:
                    from transformers import BitsAndBytesConfig
                    bnb_model_from_pretrained_args.update(dict(
                        # device_map={"": model_args.device},
                        # BUG: High version transformers report error:
                        # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time
                        # load_in_4bit=model_args.bits == 4,
                        # load_in_8bit=model_args.bits == 8,
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=model_args.bits == 4,
                            load_in_8bit=model_args.bits == 8,
                            llm_int8_skip_modules=["mm_projector"],
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False,
                            bnb_4bit_compute_dtype=torch_dtype,
                            bnb_4bit_use_double_quant=model_args.double_quant,
                            bnb_4bit_quant_type=model_args.quant_type, # {'fp4', 'nf4'}
                            bnb_4bit_quant_storage=torch_dtype,
                        )
                    ))

                config = VLLMConfigs[model_args.model_type].from_pretrained(model_args.model_name_or_path)

                config._attn_implementation = attn_implementation
                config.use_token_compression = model_args.use_token_compression
                config.use_flash_loss = model_args.use_flash_loss

                # special token
                config.learnable_tokens = model_args.learnable_tokens
                config.multi_task = False
                if model_args.vision_encoder is not None:
                    config.vision_encoder = model_args.vision_encoder
                    model = VLLMs[model_args.model_type].from_pretrained(
                        model_args.model_name_or_path,
                        config=config,
                        torch_dtype=torch_dtype,
                        do_sample=True,
                        **bnb_model_from_pretrained_args
                    )
                    if 'mixtral' in model_args.model_type:
                        import deepspeed
                        deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
                else:
                    model = transformers.LlamaForCausalLM.from_pretrained(
                        model_args.model_name_or_path,
                        config=config,
                        torch_dtype=torch_dtype,
                        do_sample=True,
                        **bnb_model_from_pretrained_args
                    )
                model.config.use_cache = False
                if model_args.freeze_backbone:
                    model.model.requires_grad_(False)

                if model_args.bits in [4, 8]:
                    from peft import prepare_model_for_kbit_training
                    model.config.torch_dtype=torch_dtype
                    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

                if args.gradient_checkpointing:
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    else:
                        def make_inputs_require_grad(module, input, output):
                            output.requires_grad_(True)
                        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

                if model_args.lora_enable:
                    from peft import LoraConfig, get_peft_model
                    lora_config = LoraConfig(
                        r=model_args.lora_r,
                        lora_alpha=model_args.lora_alpha,
                        target_modules=find_all_linear_names(model),
                        lora_dropout=model_args.lora_dropout,
                        bias=model_args.lora_bias,
                        task_type="CAUSAL_LM",
                    )
                    if model_args.bits == 16:
                        if model_args.bf16:
                            model.to(torch.bfloat16)
                        if model_args.fp16:
                            model.to(torch.float16)
                    rank0_print("Adding LoRA adapters...")
                    model = get_peft_model(model, lora_config)
            else:
                raise ValueError(
                    "No valid videollama3."
                )
                # model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        #self.ref_model = None
        # Reference model
        if is_deepspeed_zero3_enabled():
            if "videollama3" or "VideoLLaMA3" in model_id:
                if model_args.vision_encoder is not None:
                    # config.vision_encoder = model_args.vision_encoder
                    print('#'*5+'Creating ref'+'#'*5)
                    self.ref_model = VLLMs[model_args.model_type].from_pretrained(
                        model_args.model_name_or_path,
                        config=config,
                        torch_dtype=torch_dtype,
                        do_sample=True,
                        **bnb_model_from_pretrained_args
                    )
                    if 'mixtral' in model_args.model_type:
                        import deepspeed
                        deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
                else:
                    self.ref_model = transformers.LlamaForCausalLM.from_pretrained(
                        model_args.model_name_or_path,
                        config=config,
                        torch_dtype=torch_dtype,
                        do_sample=True,
                        **bnb_model_from_pretrained_args
                    )
                # self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                model_max_length=model_args.model_max_length,
                padding_side="right",
                use_fast=True,
            )
                # print(tokenizer.pad_token)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.unk_token

                if model_args.vision_encoder is not None:
                    tokenizer.add_tokens(
                        [DEFAULT_IMAGE_TOKEN, STREAM_START_TOKEN, STREAM_END_TOKEN],
                        special_tokens=True,
                    )
                    if model_args.learnable_tokens:
                        configure_alarm_tokens(
                            tokenizer,
                            models=[model, self.ref_model],
                        )

                    # initialize vision encoder + multi-modal projector
                    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=args.fsdp)

                    vision_encoder = model.get_vision_encoder()
                    vision_encoder.to(dtype=torch_dtype, device='cuda')

                    mm_max_length = model_args.mm_max_length
                    vision_encoder.image_processor.max_tokens = mm_max_length

                    mm_projector = model.get_mm_projector()
                    mm_projector.to(dtype=torch_dtype, device='cuda')

                    model_args.is_multimodal = True

                    model.config.tokenizer_padding_side = tokenizer.padding_side
                    model.config.tokenizer_model_max_length = tokenizer.model_max_length

                    if model_args.bits in [4, 8]:
                        model.get_model().mm_projector.to(dtype=torch_dtype, device='cuda')

                    # decoupled learning rate
                    model.config.llm_lr = model_args.llm_lr
                    model.config.vision_encoder_lr = model_args.vision_encoder_lr
                    model.config.mm_projector_lr = model_args.mm_projector_lr

                    if model.config.llm_lr is None:
                        for p in model.get_model().parameters():
                            p.requires_grad = False
                        for p in model.get_model().vision_encoder.parameters():
                            p.requires_grad = True
                        for p in model.get_model().mm_projector.parameters():
                            p.requires_grad = True

                    if model.config.vision_encoder_lr is None:
                        for p in model.get_model().vision_encoder.parameters():
                            p.requires_grad = False

                    if model.config.mm_projector_lr is None:
                        for p in model.get_model().mm_projector.parameters():
                            p.requires_grad = False

                    model.config.max_frames = getattr(model_args, 'max_frames', NUM_FRAMES)
                    model.config.image_aspect_ratio = model_args.image_aspect_ratio if 'avt' not in model_args.vision_encoder else 'avt'

                    # NOTE: complement model_args via model hyperparameters
                    # 1. acquire image size
                    model.config.image_size = model_args.image_size = vision_encoder.image_size
                    # 2. calculate the number of tokens in the image
                    model.config.image_token_length = model_args.image_token_length = mm_projector.cal_proj_size(vision_encoder.num_patches_per_side)
                    # 3. check if alignment
                    model.config.is_alignment = model_args.is_alignment = model_args.is_alignment = (
                        model.config.mm_projector_lr is not None and
                        model.config.llm_lr is None and
                        model.config.vision_encoder_lr is None
                    )
                    # 4. set spatial merge size as default
                    model.config.image_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
                    if self.ref_model is not None:
                        self.ref_model.config.image_token_index = model.config.image_token_index

                    vlprocessor = Videollama3Processor(
                        vision_encoder.image_processor,
                        tokenizer,
                        learnable_tokens=model_args.learnable_tokens,
                    )
                processing_class = vlprocessor
                pad_token_id = 151643
                # print(pad_token_id)
                # processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                # pad_token_id = processing_class.pad_token_id
        # data_module = make_supervised_data_module(vlprocessor=vlprocessor, data_args=script_args)
        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temporal = script_args.temporal
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
        )
        
        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.len_control = script_args.len_control
        self.beta = args.beta
        self.epsilon = args.epsilon
        self.epsilon_high = args.epsilon_high or args.epsilon

        # The paper's GRPO stage updates the LLM only. Alarm embeddings remain
        # active in the forward pass but retain the representation learned by AT-SFT.
        for parameter_name in ("visual_alarm_tokens", "textual_alarm_tokens"):
            parameter = getattr(model, parameter_name, None)
            if parameter is not None:
                parameter.requires_grad_(False)

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, **kwargs):
        """Compute token log-probabilities using the full multimodal context.

        Generation already consumes video features. Rebuilding multimodal
        embeddings here is essential: otherwise the policy gradient would be
        computed from text-only logits and would not train video-conditioned
        behavior or propagate through the alarm-token input path.
        """
        if kwargs.get("pixel_values") is not None:
            core_model = getattr(model, "module", model)
            prepared = core_model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=kwargs.get("attention_mask"),
                position_ids=kwargs.get("position_ids"),
                labels=input_ids,
                pixel_values=kwargs["pixel_values"],
                grid_sizes=kwargs.get("grid_sizes"),
                merge_sizes=kwargs.get("merge_sizes"),
                modals=kwargs.get("modals"),
                learnable_tokens=[
                    core_model.visual_alarm_tokens,
                    core_model.textual_alarm_tokens,
                ],
            )
            (
                _,
                attention_mask,
                position_ids,
                _,
                inputs_embeds,
                aligned_input_ids,
                _,
            ) = prepared
            logits = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits
            input_ids = aligned_input_ids
        else:
            logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Row-wise log-softmax avoids materializing an additional full-batch
        # vocabulary tensor, which is significant for long video prompts.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        

        prompts = [x["prompt"] for x in inputs]
        # print(prompts)
        # # prompts_text = [maybe_apply_chat_template(example, self.processing_class) for example in inputs]

        conversations = [x["conversations"] for x in inputs][0]
        conversations = copy.deepcopy(conversations)
        # print(conversations)
        modal = inputs[0]['data_type']
        # input_copy = copy.deepcopy(inputs[0]['prompt'])
        
        # input_copy = self.remove_none_from_data(input_copy)
        
        # if inputs[0]['data_type'] == 'image':
        #     input_copy[0]['content'][0]['image'] = os.getcwd() + "/Video-R1-data" + inputs[0]['path'][1:] 
        # elif inputs[0]['data_type'] == 'video':
        #     # input_copy[0]['content'][0]['video'] = os.getcwd() + "/Video-R1-data" + inputs[0]['path'][1:] 
        #     input_copy[0]['content'][0]['video'] = inputs[0]['path']
            
        # try:
        #     image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
        # except Exception as e:
        #     print(f"process_vision_info error, using fixed data, {e}")
        #     if inputs[0]['data_type'] == 'image':
        #         input_copy[0]['content'][0]['image'] = os.getcwd() + "/Video-R1-data" + '/Math/Multimath-300k/17ff4c7d14c388134de02381b1fc2824.png'
        #     elif inputs[0]['data_type'] == 'video':
        #         # input_copy[0]['content'][0]['video'] = os.getcwd() + "/Video-R1-data" + '/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_7nRmsEw7nsE.mp4'
        #         input_copy[0]['content'][0]['video'] = inputs[0]['path']
        #     image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
   
        # image_inputs = video_inputs
        # images = image_inputs
        # print(images)
        torch_dtype = self.torch_dtype
        args = self.args
        media_path = inputs[0]['path']
        if modal == "video":
            frames, timestamps = load_video(
                media_path,
                fps=self.model_args.fps,
                max_frames=self.model_args.max_frames,
            )
            images = [frames]
        elif modal == "image":
            images = load_images(media_path)
            timestamps = None
        else:
            images = None
            timestamps = None
        messages = []
        for conv in conversations:
            if conv["from"] == "human":
                # replace video tag to image tag for unified processing
                # conv["value"] = conv["value"].replace("<video>", "<image>" * len(images))
                chunks = conv["value"].split("<image>" if modal == 'image' else "<video>")
                messages.append({
                    "role": "user",
                    "content": []
                })

                for chunk_idx in range(1, 2 * len(chunks)):
                    if chunk_idx % 2 == 1:
                        chunk = chunks[chunk_idx // 2].strip()
                        messages[-1]["content"].append({"type": "text",  "text": chunk}) if chunk else None
                    else:
                        if modal == 'image':
                            messages[-1]["content"].append({"type": "image"})
                        elif modal == 'video':
                            messages[-1]["content"].append({"type": "video", "num_frames": len(images[0]), "timestamps": timestamps})
            else:
                messages.append({
                    "role": "assistant",
                    "content": conv['value']
                })

        # print(messages)

        if modal == 'video':
            merge_size = self.model_args.video_merge_size
        else:
            # image/text
            merge_size = self.model_args.image_merge_size

        data_dict = self.processing_class(
                images=images,
                text=messages,
                merge_size=merge_size,
                return_labels=False,
                return_tensors="pt",
            )
        # print(data_dict)
        if modal == 'text':
            unit_size = self.processing_class.image_processor.patch_size**2 * 3
            data_dict['pixel_values'] = torch.zeros(self.model_args.image_merge_size**2, unit_size) #.to(dtype=torch_dtype)
            data_dict['grid_sizes'] = torch.as_tensor([[1, self.model_args.image_merge_size, self.model_args.image_merge_size]])
            data_dict['merge_sizes'] = torch.as_tensor([self.model_args.image_merge_size])
        elif modal == 'image' or modal == 'video':
            assert len(data_dict['pixel_values']) > 0 and len(data_dict['grid_sizes']) > 0, f"Invalid image data: {data_dict['images']}, {data_dict['grid_thws']}"
        data_dict['modals'] = [modal]
        
        prompt_inputs = data_dict
        prompt_inputs['input_ids'] = prompt_inputs['input_ids'].unsqueeze(0).to(device='cuda')
        prompt_inputs['pixel_values'] = prompt_inputs['pixel_values'].to(dtype=torch_dtype, device='cuda')
        # print(prompt_inputs['input_ids'].shape) # [1,10498]
        # print(prompt_inputs['pixel_values'].dtype, prompt_inputs['pixel_values'].shape, prompt_inputs['pixel_values'].device)
        # prompt_inputs = self.processing_class(
        #     text=copy.deepcopy(messages),
        #     images=images,
        #     return_tensors="pt",
        #     padding=True,
        #     padding_side="left",
        #     add_special_tokens=False,
        # )
        # print(prompt_inputs)
        # prompt_inputs = super()._prepare_inputs(prompt_inputs)
        
        # # fix prompt_inputs["input_ids"] length issue
        # if self.max_prompt_length is not None:
        #     prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
        #     prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        # prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        # print(prompt_inputs)

        # if self.temporal and video_inputs:
        #     indices = torch.randperm(video_inputs[0].size(0))
        #     shuffled_video_inputs = [video_inputs[0][indices]]
        #     shuffled_prompt_inputs = self.processing_class(
        #         text=copy.deepcopy(prompts_text),
        #         images=image_inputs,
        #         videos=shuffled_video_inputs,
        #         return_tensors="pt",
        #         padding=True,
        #         padding_side="left",
        #         add_special_tokens=False,
        #     )
        #     shuffled_prompt_inputs = super()._prepare_inputs(shuffled_prompt_inputs)
        #     shuffled_prompt_ids, shuffled_prompt_mask = shuffled_prompt_inputs["input_ids"], shuffled_prompt_inputs["attention_mask"]
        #     if self.max_prompt_length is not None:
        #         shuffled_prompt_ids = shuffled_prompt_ids[:, -self.max_prompt_length :]
        #         shuffled_prompt_mask = shuffled_prompt_mask[:, -self.max_prompt_length :]
        
        # print(prompt_inputs)
        # print(self.generation_config)
        # Generate completions
        # generated_ids = model.generate(**prompt_inputs, max_new_tokens=512)
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
            prompt_ids = prompt_inputs['input_ids']
            prompt_ids_re = prompt_ids.repeat(prompt_completion_ids.shape[0], 1)
            prompt_ids = prompt_ids_re
            prompt_completion_ids = torch.cat((prompt_ids_re, prompt_completion_ids), 1)
            
            prompt_length = prompt_ids.size(1)
            # completion_ids = prompt_completion_ids
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            # prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            
        # completion_ids.cpu().numpy()
        # print(completion_ids)
        # print(prompt_completion_ids)
        # print(prompt_completion_ids.shape)
        # print(prompt_ids)
        # print(prompt_ids.shape)
        # print(completion_ids)
        # print(completion_ids.shape)
        
        print('path:', media_path)
        print('problem_id:', inputs[0]['problem_id'])       
        print('prompt_length:', prompt_length)
        print('completion_length:', completion_ids.shape)
        print('prompt_completion_length:', prompt_completion_ids.shape)
        torch.cuda.empty_cache()
        
        
        
        # Mask everything after the first EOS token
        # print(self.processing_class.eos_token_id) # 151645
        # is_eos = completion_ids == self.processing_class.eos_token_id
        # print(is_eos)
        end_token_ids = {
            self.processing_class.eos_token_id,
            self.processing_class.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        }
        is_eos = torch.zeros_like(completion_ids, dtype=torch.bool)
        for token_id in end_token_ids:
            if token_id is not None and token_id >= 0:
                is_eos |= completion_ids == token_id
        # print(is_eos)
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        # pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
        # image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)
        

        
        prompt_inputs.pop("input_ids")
        # prompt_inputs.pop("attention_mask")
        
        if inputs[0]['data_type'] == 'image':
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["grid_sizes"] = prompt_inputs["grid_sizes"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["merge_sizes"] = prompt_inputs["merge_sizes"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["modals"] = prompt_inputs["modals"] * len(prompt_completion_ids)
        # import pdb; pdb.set_trace()
        
        # print(prompt_inputs["pixel_values"].shape)
        # print(prompt_inputs["grid_sizes"].shape)
        if inputs[0]['data_type'] == 'video':
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1, 1)
            prompt_inputs["grid_sizes"] = prompt_inputs["grid_sizes"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["merge_sizes"] = prompt_inputs["merge_sizes"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["modals"] = prompt_inputs["modals"] * len(prompt_completion_ids)

            # prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            if 'second_per_grid_ts' in prompt_inputs:
                del prompt_inputs["second_per_grid_ts"]
                # prompt_inputs["second_per_grid_ts"] = torch.tensor(prompt_inputs["second_per_grid_ts"]).repeat(len(prompt_completion_ids), 1)
        # print(prompt_inputs)
        # print("pixel_values_shape_re: ", prompt_inputs["pixel_values"].shape)
        # print()
        
        
        # print(prompt_completion_ids)
        # print(prompt_inputs)
        per_token_logps = self._get_per_token_logps(
            model, prompt_completion_ids, **prompt_inputs
        )
        per_token_logps = per_token_logps[:, -completion_ids.size(1):]


        # print(per_token_logps.shape)
        # print(prompt_completion_ids)
        # print(prompt_inputs)
        # pdb.set_trace()
        torch.cuda.empty_cache()
        with torch.inference_mode():
            ref_model = self.ref_model if self.ref_model is not None else model
            ref_per_token_logps = self._get_per_token_logps(
                ref_model, prompt_completion_ids, **prompt_inputs
            )
            ref_per_token_logps = ref_per_token_logps[:, -completion_ids.size(1):]
        # print(ref_per_token_logps.shape)
        # Compute the KL divergence between the model and the reference model
        
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)  # 限制 x 的范围
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1
        
        if self.temporal and video_inputs:
            shuffled_completions = self.processing_class.batch_decode(shuffled_completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                shuffled_completions = [[{"role": "assistant", "content": shuffled_completion}] for shuffled_completion in shuffled_completions]
                
            # Compute the rewards
            shuffled_prompts = [prompt for prompt in prompts for _ in range(self.shuffled_num_generations)]
            shuffled_rewards_per_func = torch.zeros(len(shuffled_prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                shuffled_reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in shuffled_reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        shuffled_reward_kwargs[key].extend([example[key]] * self.shuffled_num_generations)
                shuffled_output_reward_func = reward_func(prompts=shuffled_prompts, completions=shuffled_completions, **shuffled_reward_kwargs)
                shuffled_rewards_per_func[:, i] = torch.tensor(shuffled_output_reward_func, dtype=torch.float32, device=device)

        
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
            
        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    # Repeat each value in the column for `num_generations` times
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        

        
        
        if self.temporal and video_inputs:
            temporal_rewards_per_func = rewards_per_func.clone()
            
            acc_mean = temporal_rewards_per_func[:, 0].mean()
            shuffled_acc_mean = shuffled_rewards_per_func[:, 0].mean()

            if acc_mean >= 0.8 * shuffled_acc_mean:
                mask = temporal_rewards_per_func[:, 0] > 0.1
                temporal_rewards_per_func[mask, 0] = temporal_rewards_per_func[mask, 0] + 0.3
                temporal_rewards = torch.tensor([1.0]).to('cuda')
            else:
                temporal_rewards = torch.tensor([0.0]).to('cuda')
        else:
            temporal_rewards =  torch.tensor([0.5]).to('cuda')
        
        # Sum the rewards from all reward functions
        if self.temporal and video_inputs:
            rewards = temporal_rewards_per_func.sum(dim=1)
        else:
            rewards = rewards_per_func.sum(dim=1)
    
        
        if self.len_control:
            mem_rewards = [0] * self.num_generations
            mask = rewards_per_func[:, 0] > 0.1
            lenth_list = completion_mask.sum(1)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
            #             if len(selected_indices) > 1 and len(selected_indices) < self.num_generations:
            # if len(selected_indices) > 1:
            #     selected_items = [(i, lenth_list[i]) for i in selected_indices]
            #     sorted_items = sorted(selected_items, key=lambda x: x[1], reverse=True)
            #     N = len(sorted_items)
            #     for rank, (idx, length) in enumerate(sorted_items):
            #         reward = 0.2 - 0.2 * (rank / N)
            #         rewards[idx] += reward
            #         mem_rewards[idx] = reward
            # for idx in range(len(lenth_list)):
            #     if lenth_list[idx] >= 512:
            #         rewards[idx] -= 0.5
                    
            if len(selected_indices) > 1:     
                for idx in selected_indices:
                    if 320 <= lenth_list[idx] <= 512:
                        rewards[idx] += 0.2
        
        print(rewards)
        print(completion_mask.sum(1))

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # if self.len_control and len(selected_indices) == self.num_generations:
        #     for idx in range(len(rewards)):
        #         advantages[idx] += (mem_rewards[idx] - 0.2) * 2

        # Stop-gradient creates pi_old for this sampled batch. With the default
        # single GRPO iteration, the ratio is numerically one on the first pass
        # but still carries the correct policy-gradient derivative.
        old_per_token_logps = per_token_logps.detach()
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.epsilon,
            1 + self.epsilon_high,
        )
        policy_objective = torch.minimum(
            ratio * advantages.unsqueeze(1),
            clipped_ratio * advantages.unsqueeze(1),
        )
        # Token-level clipped policy objective plus the reverse-KL estimator
        # against the frozen reference model.
        per_token_loss = -(policy_objective - self.beta * per_token_kl)
        # per_token_loss = -per_token_loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
       
            
        # import pdb
        # pdb.set_trace()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        
        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        
        num_devices = gathered_rewards.size(0) // self.num_generations 
        rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices
        
        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices
        
        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)
        
        if self.temporal:
            temporal_rewards_list = self.accelerator.gather_for_metrics(temporal_rewards)
            self._metrics["temporal_rewards"].append(self.accelerator.gather_for_metrics(temporal_rewards_list).mean().item())
        
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
