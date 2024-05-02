from collections.abc import Callable
from typing import Any

import torch
from accelerate import Accelerator
from datasets import Dataset
from .args import ScriptArgs
from .config import get_lora_config, get_ppo_config
from transformers import Adafactor
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from trl import PPOConfig
from trl import PPOTrainer
from trl import set_seed
# from trl import create_reference_model


def build_trainer(
    args: ScriptArgs,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    data_collator: Callable[..., Any] | None = None,
    **lora_kwargs: Any,
) -> tuple[PPOConfig, PPOTrainer]:
    """Build the PPO trainer.

    Args:
        args (ScriptArgs): The script arguments.
        tokenizer (AutoTokenizer): The tokenizer to use.
        dataset (Dataset): The dataset to use.
        lora_kwargs: Keyword arguments for the LoRA config.

    Returns:
        tuple[PPOConfig, PPOTrainer]: The PPO config & trainer objects.

    """
    config = get_ppo_config(args)
    lora_config = get_lora_config(**lora_kwargs)

    # Set seed before initializing value head for deterministic eval.
    set_seed(config.seed)

    current_device = Accelerator().local_process_index
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        load_in_8bit=True,
        device_map={'': current_device},
        peft_config=lora_config,
    )
    # ref_model = create_reference_model(model, num_shared_layers=6)
    ref_model = None

    optimizer, lr_scheduler = None, None
    if args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)

    trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=data_collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    return config, trainer